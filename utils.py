import os, dataclasses, numcodecs, abc, time, dask
from aicsimageio import AICSImage
from aicsimageio.metadata.utils import OME
import numpy as np, cupy as cp
from pathlib import Path
import zarr
from typing import (Union, Iterable)
import warnings

import itertools
from pathlib import Path
import glob, zarr

from typing import Callable, Any
from collections import defaultdict
import dask.array as da



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

def rlu_deconvolve(image, psf, iterations=5, kernel_type='gaussian', verbose=False):
    # a good psf for my images: generate_gaussian_kernel((10, 7, 7), (3, 2, 2))
    """ Skimage's Richardson Lucy deconvolution with slight modifications.

        Parameters:
        -----------
        image: array
            An n-dimensional numpy array.
        psf: array or iterable
            Either a numpy array specifying the point spread function, or an iterable specifying the shape of the psf.
            If directly point spread function, it must have the same number of dimensions as 'image'.
            If an iterable, then 'kernel_type' must be specified to calculate the psf.
        iterations: int
            Number of iterations of the deconvolution
        kernel_type: str
            Either 'mean' or 'gaussian'. Conveniently calculates a point spread function.
            Ignored if psf is an array.

        Returns:
        --------
        deconvolved: array
            Array with same shape as 'image'."""
    cnv.cp_array(image)
    if hasattr(psf, 'nonzero'):
        assert image.ndim == psf.ndim, 'The image and psf must have the same number of dimensions.'
    elif np.isscalar(psf):
        psf = [psf] * image.ndim
    elif hasattr(psf, 'index'):
        if kernel_type == 'mean':
            psf = np.ones(psf) / np.prod(psf)
        elif kernel_type == 'gaussian':
            w = [i // 3 for i in psf]
            psf = generate_gaussian_kernel(psf, w)
        else:
            raise TypeError('kernel_type must be either of "mean" or "gaussian"')
    image = image.astype(float)
    psf = st._block_zeroes(psf)
    im_deconv = np.full(image.shape, 0.5)
    psf_mirror = psf[::-1, ::-1, ::-1]
    conv_im = fftconvolve(im_deconv, psf_mirror, mode='same')
    for i in range(iterations):
        conv_im = st._block_zeroes(conv_im)
        relative_blur = image / conv_im
        c = fftconvolve(relative_blur, psf, mode='same')
        im_deconv *= c
        conv_im = fftconvolve(im_deconv, psf_mirror, mode='same')
        if verbose:
            print('iteration: {}'.format(i))
    return im_deconv

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


def otsu(img, bincount=9, return_thresholded=True):
    hist, vals = da.histogram(img, bins = bincount, range = [img.min(), img.max()])  # calculate histogram
    binmids = 0.5 * (vals[1:] + vals[:-1])  # get the midpoints of the bins
    countsums0 = da.cumsum(hist)  # cumulative sum of the voxel counts in the bins, starting from the minimum
    countsums1 = da.cumsum(hist[::-1])[::-1]  # cumulative sum of the voxel counts in the bins, starting from the maximum
    binsums = binmids * hist  # sum of intensities within each bin
    valuesums0 = da.cumsum(binsums)  # cumulative sum of intensities, starting from the minimum
    valuesums1 = da.cumsum(binsums[::-1])[::-1]  # cumulative sum of intensities, starting from the maximum
    cummeans0 = valuesums0 / countsums0  # cumulative mean of intensities, starting from the minimum
    cummeans1 = valuesums1 / countsums1  # cumulative mean of intensities, starting from the maximum
    objective = countsums0[:-1] * countsums1[1:] * (cummeans0[:-1] - cummeans1[
                                                                     1:]) ** 2  # interclass variance equation. Frameshift ensures correct bin match
    argt = da.argmax(objective)  # index of the maximum interclass variation
    t = binmids[argt]  # threshold value corresponding to maximum interclass variation
    if return_thresholded:
        return img > t
    else:
        return t


def mean_threshold(img, return_thresholded=False):
    t = da.mean(img)
    if return_thresholded:
        return img > t
    else:
        return t


