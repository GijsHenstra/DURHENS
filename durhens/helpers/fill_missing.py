#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:31:01 2021

@author: gijshenstra
"""

import warnings

import rasterio
import numpy as np
import matplotlib.pyplot as plt

from helpers import boundingbox as bb

def array(img, sigma=6.0, truncate=20.0):
    """
    Inpaint np.nan values in an image.

    Replacing empty arrays cells by calculating neighbouring values by means of
    gaussian blur.

    Adjusted from: https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    Parameters
    ----------
    %(input)s
    img : ndarray
        Image to fill
    sigma : float, optional
        Standard deviation for Gaussian kernel.
    truncate : float, optional
        Truncate filter at this many sigmas.
    %(mode)s

    Returns
    -------
    img_fill : ndarray
        The filled input.
    """
    import scipy as sp
    import scipy.ndimage
    # print(f'try with sigma {sigma}')
    nan_mask = (np.isnan(img) | (np.abs(img) == np.inf) | np.greater(img, 1e24))
    # plt.imshow(np.moveaxis(nan_mask, 0, -1)[..., 0])
    # plt.title('nan_mask fill_missiing r46')
    # plt.show()

    if nan_mask.sum() == 0:
        # warnings.warn('No values are missing, returned original img')
        return img

    V = img.copy()
    V[nan_mask] = 0
    
    VV = sp.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = (0 * img.copy()) + 1
    W[nan_mask] = 0
    WW = sp.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    img_gaussian = VV / WW

    img_fill = img.copy()
    img_fill[nan_mask] = img_gaussian[nan_mask]

    return img_fill.copy()


def geotiff(src, save_path_filled=None, max_sigma=25):
    
    if save_path_filled is None:
        src_name = src.name
        idx = src_name.index('.')
        save_path_filled = src_name[:idx] + '_filled' + src_name[idx:]
    
    try:
        dsr = rasterio.open(save_path_filled)
        if not bb.equals(dsr, src):
            raise Exception('Wrong file found, bounding boxes is different')
            
    except Exception as e:
        print(e)
        
        if isinstance(src, str):
            dsr = rasterio.open(src)
        elif isinstance(src, rasterio.io.DatasetReader):
            dsr = src
        else:
            raise TypeError('Invalid type specified')
            
        arr = dsr.read(1)
        
        if dsr.nodata is not None:
            nodata_mask = arr == arr.dtype.type(dsr.nodata)
        else:
            nodata_mask = arr == 0
        
        _nodata_val = np.nan
        
        arr[nodata_mask] = _nodata_val
        
        arr_filled = arr.copy()
        sigma = 1.0
        while True:
            arr_filled = array(arr_filled.copy(), sigma=sigma)
            
            if not np.isnan(arr_filled).any() or sigma > max_sigma:
                # print(f'Break for sigma {sigma}')
                arr_filled = np.nan_to_num(arr_filled, nan=1.0)
                break
            
            if sigma < 5:
                sigma += 0.5
            elif sigma < 10:
                sigma += 2
            else:
                sigma += 5
                
        with rasterio.open(save_path_filled, "w", **dsr.meta) as dest:
            dest.write(arr_filled[np.newaxis, ...])
        
        dsr = rasterio.open(save_path_filled)
    
    return dsr