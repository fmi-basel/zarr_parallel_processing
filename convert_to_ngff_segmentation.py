import os, dataclasses, numcodecs, abc, time, dask
from aicsimageio import AICSImage
from aicsimageio.metadata.utils import OME
import numpy as np, cupy as cp
from pathlib import Path
import zarr
from typing import (Union, Iterable)
import warnings
from dask import array as da, bag, delayed
from dask.highlevelgraph import HighLevelGraph
import dask
from dask_cuda import LocalCUDACluster
#from rmm.allocators.cupy import rmm_cupy_allocator
#import rmm

import itertools
from pathlib import Path
import glob, zarr
from multiscales import Multimeta
from typing import Callable, Any
from collections import defaultdict

from skimage import restoration
from skimage.filters import threshold_otsu
import dask_image.ndfilters
from cellpose import core, models, metrics

from distributed import LocalCluster, Client
from joblib import delayed as jdel, Parallel, parallel_config
from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown()


def get_regions(array_shape,
                region_shape,
                as_slices = False
                ):
    assert len(array_shape) == len(region_shape)
    steps = []
    for i in range(len(region_shape)):
        size = array_shape[i]
        inc = region_shape[i]
        seq = np.arange(0, size, inc)
        if size > seq[-1]:
            seq = np.append(seq, size)
        increments = tuple([(seq[i], seq[i+1]) for i in range(len(seq) - 1)])
        tuples = tuple(tuple(item) for item in increments)
        if as_slices:
            slcs = tuple([slice(*item) for item in tuples])
            steps.append(slcs)
        else:
            steps.append(tuples)
    out = list(itertools.product(*steps))
    return out

def read_image(file_path: str):
    img = AICSImage(file_path)
    return img.get_image_dask_data()

def create_zarr_array(directory: str,
                   array_name: str,
                   shape: tuple,
                   chunks: tuple,
                   dtype: Any,
                   overwrite: bool = False,
                   ) -> zarr.Array:
    chunks = tuple(np.minimum(shape, chunks))
    if not isinstance(directory, zarr.Group):
        path = os.path.join(directory, array_name)
        dataset = zarr.create(shape=shape,
                              chunks=chunks,
                              dtype=dtype,
                              store=path,
                              dimension_separator='/',
                              overwrite=overwrite
                              )
    else:
        _ = directory.create(name = array_name,
                              shape = shape,
                              chunks = chunks,
                              dtype = dtype,
                              dimension_separator='/',
                              overwrite=overwrite
                              )
        dataset = directory[array_name]
    return dataset

def write_single_region(region: da.Array,
                dataset: str,
                region_slice: slice = None
                ):
    da.to_zarr(region,
               url = dataset,
               region = region_slice,
               compute=True,
               return_stored=True
               )
    return dataset

def write_regions_sequential(
                  image_regions: tuple,
                  region_slices: tuple,
                  dataset: zarr.Array
                  ):
    executor = get_reusable_executor(max_workers=n_jobs,
                                     kill_workers=True,
                                     context='loky')
    for region_slice, image_region in zip(region_slices, image_regions):
        executor.submit(write_single_region,
                         region=image_region,
                         dataset=dataset,
                         region_slice=region_slice
                         )
    return dataset

def write_regions(
                    image_regions: tuple,
                    region_slices: tuple,
                    dataset: zarr.Array,
                    client: Client = None
                    ) -> zarr.Array:
    if client is None:
        n_jobs = 4
    else:
        n_jobs = client.cluster.workers.__len__()
    client.cluster.scale(n_jobs)
    client.scatter(image_regions)
    client.scatter(region_slices)
    with parallel_config(backend = 'loky', n_jobs = n_jobs):
        with Parallel() as parallel:
            parallel(jdel(write_single_region)(region = image_region,
                                               region_slice = region_slice,
                                               dataset = dataset)
                     for image_region, region_slice in
                     zip(image_regions, region_slices)
                     )
    return dataset

def deconvolve_block(img, psf=None, iterations=20):
    # Pad PSF with zeros to match image shape
    pad_l, pad_r = np.divmod(np.array(img.shape) - np.array(psf.shape), 2)
    pad_r += pad_l
    psf = np.pad(psf, tuple(zip(pad_l, pad_r)), 'constant', constant_values=0)
    # Recenter PSF at the origin
    # Needed to ensure PSF doesn't introduce an offset when
    # convolving with image
    for i in range(psf.ndim):
        psf = np.roll(psf, psf.shape[i] // 2, axis=i)
    # Convolution requires FFT of the PSF
    psf = np.fft.rfftn(psf)
    # Perform deconvolution in-place on a copy of the image
    # (avoids changing the original)
    img_decon = np.copy(img)
    for _ in range(iterations):
        ratio = img / np.fft.irfftn(np.fft.rfftn(img_decon) * psf)
        img_decon *= np.fft.irfftn((np.fft.rfftn(ratio).conj() * psf).conj())
    return img_decon


import numpy as np


def gaussian_psf(shape, mean, cov):
    """
    Computes an n-dimensional Gaussian function over a grid defined by the given shape.

    Parameters:
        shape (tuple of int): Shape of the n-dimensional grid (e.g., (height, width, depth)).
        mean (float or list-like): Scalar or array-like representing the mean of the Gaussian.
                                   If scalar, it will be applied to all dimensions.
        cov (float or list-like): Scalar, 1D array, or 2D array representing the covariance.
                                  - If scalar, creates an isotropic Gaussian.
                                  - If 1D, creates a diagonal covariance matrix.
                                  - If 2D, used directly as the covariance matrix.

    Returns:
        np.ndarray: An n-dimensional array containing the Gaussian function values.
    """
    n = len(shape)
    if np.isscalar(mean):
        mean = np.full(n, mean)
    else:
        mean = np.asarray(mean)
    if mean.shape[0] != n:
        raise ValueError(f"Mean must match the number of dimensions ({n}).")
    if np.isscalar(cov):
        cov = np.eye(n) * cov
    elif np.ndim(cov) == 1:
        if len(cov) != n:
            raise ValueError(f"Covariance vector length must match the number of dimensions ({n}).")
        cov = np.diag(cov)
    elif np.ndim(cov) == 2:
        cov = np.asarray(cov)
        if cov.shape != (n, n):
            raise ValueError(f"Covariance matrix must be ({n}, {n}).")
    else:
        raise ValueError("Covariance must be a scalar, 1D array, or 2D matrix.")
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    coords = np.stack(grids, axis=-1)  # Shape: (*shape, n)
    flat_coords = coords.reshape(-1, n)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    if det_cov <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    norm_factor = 1 / (np.sqrt((2 * np.pi) ** n * det_cov))
    diff = flat_coords - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
    gaussian_values = norm_factor * np.exp(exponent)
    return gaussian_values.reshape(shape)



def richardson_lucy(img: da.Array,
                    psf: da.Array,
                    iterations: int = 20,
                    backend: str = 'cupy'
                    ):
    if backend == 'cupy':
        img = img.map_blocks(cp.asarray)
        psf = psf.map_blocks(cp.asarray)
    deconvolved = img.map_overlap(
                                    deconvolve_block,
                                    psf = psf,
                                    iterations = iterations,
                                    meta = img._meta,
                                    depth = tuple(np.array(psf.shape) // 2),
                                    boundary = "periodic"
                                )
    if backend == 'cupy':
        deconvolved = deconvolved.map_blocks(cp.asnumpy)
    return deconvolved



def to_ngff(arr: da.Array,
            output_path: str,
            region_shape: tuple = None,
            scale: tuple = None,
            units: tuple = None,
            client: Client = None
            ) -> zarr.Group:

    region_slices = get_regions(arr.shape, region_shape, as_slices = True)

    gr = zarr.open_group(output_path, mode='a')
    dataset = create_zarr_array(gr,
                                array_name = '0',
                                shape = arr.shape,
                                chunks = chunks,
                                dtype = arr.dtype,
                                overwrite = True
                                )

    meta = Multimeta()
    meta.parse_axes(axis_order='tczyx',
                    unit_list = units
                    )
    meta.add_dataset(path = '0',
                     scale = scale
                     )
    meta.to_ngff(gr)

    image_regions = [arr[region_slice] for region_slice in region_slices]
    if client is not None:
        client.scatter(region_slices)
        client.scatter(image_regions)

    write_regions(image_regions = image_regions,
                  region_slices = region_slices,
                  dataset = dataset,
                  client = client)
    return gr


def normalize_custom(img,
    lower=0.0,
    upper=400.0,
    background=100,
    percentile = False,
    percentile_lower = 1,
    percentile_upper = 99,
    ):

    if percentile:
        x01 = np.percentile(img, percentile_lower)
        x99 = np.percentile(img, percentile_upper)
        if x99 - x01 > 1e-3:
            img = (img - x01) / (x99 - x01)
        else:
            img[:] = 0
    else:
        img = np.asarray(img)
        img = img.astype(np.float32)
        img = img - background
        img[img<0] = 0
        img = (img - lower) / (upper - lower)

    return img

if __name__ == '__main__':

    chunks = (1, 1, 96, 128, 128)
    region_shape = (1, 1, 91, 554, 928)
    scale = (600, 1, 2, 0.406, 0.406)
    units = ('s', 'Channel', 'µm', 'µm', 'µm')
    psf = gaussian_psf((1, 1, 12, 16, 16), (1, 1, 6, 8, 8), (1, 1, 12, 16, 16))
    psf = da.from_array(psf, chunks = chunks)

    n_jobs = 4
    threads_per_worker = 1
    memory_limit = '5GB'

    input_tiff_path_mg = f"/path/mg/*"
    input_tiff_path_h2b = f"/path/H2B/*"

    output_zarr_path = f"/path/cellpose_seg.zarr"

    model_nuc = models.CellposeModel(gpu=True, model_type="CP")
    model_cell = models.CellposeModel(gpu=True, model_type="CPx")

    t0 = time.time()

    paths_mg = sorted(glob.glob(input_tiff_path_mg))
    paths_h2b = sorted(glob.glob(input_tiff_path_h2b))

    with LocalCluster(n_workers=n_jobs, threads_per_worker=threads_per_worker, memory_limit=memory_limit) as cluster:
        cluster.scale(n_jobs)
        with Client(cluster) as client:

            ### Read image collections
            imgs_mg = [read_image(path) for path in paths_mg]
            imgs_h2b = [read_image(path) for path in paths_h2b]

            ### Concatenate collections into a single dask array
            mg_merged = da.concatenate(imgs_mg, axis = 0) # concatenate along the time dimension
            h2b_merged = da.concatenate(imgs_h2b, axis = 0) # concatenate along the time dimension


            ### Process merged images
            t, c, z, y, x = h2b_merged.shape
            imgs_thres_mg = da.zeros(mg_merged.shape)
            imgs_thres_h2b = da.zeros(h2b_merged.shape)

            for tp in range(t):
                print(tp)
                img = normalize_custom(h2b_merged[tp, 0,:,:,:], lower = 0.0, upper = 800.0)
                masks, flows, styles = model_nuc.eval(img,
                                    channels = [0,0],
                                    diameter = 25.0,
                                    flow_threshold = 0.4,
                                    cellprob_threshold = 0.0,
                                    stitch_threshold = 0.5,
                                    do_3D = False,
                                    normalize = False,
                                    )
                imgs_thres_h2b[tp, 0,:,:,:] = masks

                img = normalize_custom(mg_merged[tp, 0,:,:,:], lower = 0.0, upper = 800.0)
                masks, flows, styles = model_cell.eval(img,
                                    channels = [0,0],
                                    diameter = 30.0,
                                    flow_threshold = 0.4,
                                    cellprob_threshold = 0.0,
                                    stitch_threshold = 0.5,
                                    do_3D = False,
                                    normalize = False,
                                    )
                imgs_thres_mg[tp, 0,:,:,:] = masks


            imgs_merged = da.concatenate((mg_merged, imgs_thres_mg, h2b_merged, imgs_thres_h2b), axis = 1) # concatenate along the channel dimension

            print(imgs_merged.shape)

            to_ngff(imgs_merged,
                    output_path = output_zarr_path,
                    region_shape = region_shape,
                    scale = scale,
                    units = units,
                    client = client
                    )
