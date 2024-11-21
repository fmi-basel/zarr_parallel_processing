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
from dask_image import ndfilters
from dask_cuda import LocalCUDACluster
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm

import itertools
from pathlib import Path
import glob, zarr
from zarr_parallel_processing.multiscales import Multimeta
from zarr_parallel_processing import utils

from typing import Callable, Any
from collections import defaultdict

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

def read_image(file_path: Path | str):
    img = AICSImage(file_path)
    return img.get_image_dask_data()

def create_zarr_array(directory: Path | str | zarr.Group,
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
                dataset: Path | str | zarr.Array,
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
                  dataset: zarr.Array,
                  **kwargs
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




# def threshold_local(img: da.Array)


def process_and_save_to_ngff(arr: da.Array,
                            output_path: str | Path,
                            region_shape: tuple = None,
                            scale: tuple = None,
                            units: tuple = None,
                            client: Client = None,
                            parallelize_over_regions = True,
                            func: Callable = utils.otsu,
                            **func_params
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
    # processed_regions = image_regions
    processed_regions = [func(reg, **func_params) for reg in image_regions]

    if client is not None:
        client.scatter(region_slices)
        client.scatter(image_regions)

    if not parallelize_over_regions:
        write_regions(image_regions = processed_regions,
                              region_slices = region_slices,
                              dataset = dataset,
                              client = client)
    else:
        write_regions_sequential(image_regions = processed_regions,
                                  region_slices = region_slices,
                                  dataset = dataset,
                                  client = client)
    return gr



if __name__ == '__main__':
    chunks = (1, 1, 48, 128, 128)
    region_shape = (1, 1, 91, 554, 928)
    scale = (600, 1, 2, 0.406, 0.406)
    units = ('s', 'Channel', 'µm', 'µm', 'µm')
    # psf = gaussian_psf((1, 1, 12, 16, 16), (1, 1, 6, 8, 8), (1, 1, 12, 16, 16))
    # psf = da.from_array(psf, chunks = chunks)

    block_size = (1, 1, 5, 9, 9)

    n_jobs = 4
    threads_per_worker = 2
    memory_limit = '8GB'

    input_tiff_path_mg = f"/home/oezdemir/data/original/franziska/crop/mG_View1/*"
    input_tiff_path_h2b = f"/home/oezdemir/data/original/franziska/crop/H2B_View1/*"

    output_zarr_path = f"/home/oezdemir/data/original/franziska/concat.zarr"

    t0 = time.time()

    paths_mg = sorted(glob.glob(input_tiff_path_mg))
    paths_h2b = sorted(glob.glob(input_tiff_path_h2b))



    # imgs_mg = [read_image(path) for path in paths_mg]
    # imgs_h2b = [read_image(path) for path in paths_h2b]
    #
    # ### Concatenate collections into a single dask array
    # mg_merged = da.concatenate(imgs_mg, axis=0)  # concatenate along the time dimension
    # h2b_merged = da.concatenate(imgs_h2b, axis=0)  # concatenate along the time dimension
    # imgs_merged = da.concatenate((mg_merged, h2b_merged), axis=1)  # concatenate along the channel dimension

    # processed_img = da.concatenate([otsu(img, return_thresholded=True) for img in imgs_merged], axis=0)


    with LocalCluster(processes=True,
                      nanny=True,
                      n_workers=n_jobs,
                      threads_per_worker=threads_per_worker,
                      memory_limit=memory_limit) as cluster:
        cluster.scale(n_jobs)
        with Client(cluster,
                    heartbeat_interval="120s",
                    timeout="600s",
                    ) as client:

            ### Read image collections
            imgs_mg = [read_image(path) for path in paths_mg]
            imgs_h2b = [read_image(path) for path in paths_h2b]

            ### Concatenate collections into a single dask array
            mg_merged = da.concatenate(imgs_mg, axis = 0) # concatenate along the time dimension
            h2b_merged = da.concatenate(imgs_h2b, axis = 0) # concatenate along the time dimension
            imgs_merged = da.concatenate((mg_merged, h2b_merged), axis = 1) # concatenate along the channel dimension

            ### Process merged images
            processed_img = imgs_merged
            # processed_img = ndfilters.threshold_local(imgs_merged, block_size=block_size, method='mean')
            # processed_img = ndfilters.gaussian_filter(imgs_merged, sigma = (0.4, 0.4, 1, 1, 1))
            # filtered = ndfilters.uniform_filter(imgs_merged, size = block_size)
            # processed_img = imgs_merged > filtered
            # processed_mg = da.concatenate([utils.mean_threshold(img, return_thresholded=True) for img in imgs_mg], axis = 0)
            # processed_h2b = da.concatenate([utils.mean_threshold(img, return_thresholded=True) for img in imgs_h2b], axis = 0)
            # processed_mg = da.concatenate([utils.otsu(img, bincount = 9, return_thresholded=True) for img in imgs_mg], axis = 0)
            # processed_h2b = da.concatenate([utils.otsu(img, bincount = 9, return_thresholded=True) for img in imgs_h2b], axis = 0)
            # processed_img = da.concatenate((processed_mg, processed_h2b), axis = 1) # concatenate along the channel dimension

            process_and_save_to_ngff(processed_img,
                    output_path = output_zarr_path,
                    region_shape = region_shape,
                    scale = scale,
                    units = units,
                    client = client,
                    parallelize_over_regions=False,
                    func = utils.otsu,
                    )





