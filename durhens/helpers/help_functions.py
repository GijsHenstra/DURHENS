#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:47:17 2020

@author: gijshenstra

# Some docstrings were provided by a AI model called ChatGPT, with a knowledge cutoff at 2021, and edited by me
"""

import numpy as np
import math

import datetime
import time
import glob
import cv2
import numba
import scipy
import json
import os

from plot import plot2d

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import boundingbox as bb

from skimage.measure import block_reduce

# %% Random functions from runfunc


def get_data(locals_dict, var_str):
    if '[' in var_str:
        var_without_key = var_str.split('[')[0]
        key = var_str.split('[')[-1].split(']')[0].strip("'").strip('"')
        data = locals_dict[var_without_key][key].copy()
    else:
        data = locals_dict[var_str].copy()
        
    return data


def intensity_top_faces(array_2d, 
                        normalize=False,
                        max_intensity=1500, 
                        min_visibility=0.3):  
    """Alter 2d array by raising minimum visibility and/or normalizing the data.

    Parameters
    ----------
    array_2d : array-like
        Input array to be edited
    normalize : bool
        If True, normalize the data

    Returns
    -------
        Normalized or otherwise altered array.

    """

    if normalize:
        array_2d = array_2d / max_intensity

    if min_visibility > 0:
        if not normalize:
            min_visibility = min_visibility * max_intensity

        array_2d = min_visibility + (1 - min_visibility) * array_2d

    return array_2d


def intensity_all_faces(array_3d,
                        normalize=False,
                        max_intensity=1500, min_visibility=0.3):
    """
    Alter 3d array by raising minimum visibility and/or normalizing the data.
    
    Parameters
    ----------
    array_3d : array-like
        Input array to be edited.
    normalize : bool
        If True, normalize the data

    Returns
    -------
        Normalized or otherwise altered array.
    """

    if normalize:
        array_3d = array_3d / max_intensity

    if min_visibility > 0:
        if not normalize:
            min_visibility = min_visibility * max_intensity

        array_3d = min_visibility + (1 - min_visibility) * array_3d

    return array_3d


# def wind_degrees_to_faces(degr, speed, method='uniform'):
#     if method == 'uniform':
#         face_wind_speed = np.array([speed,
#                                      speed,
#                                      speed,
#                                      speed,
#                                      speed,
#                                      speed])

#     if method == 'shadow':
#         # to be inserted
#         pass

#     return face_wind_speed


def find_closest_stn(latitude, longitude, print_loc=True,
                     knmi_locs_path='data/knmi_station_locations') -> int:
    """
    Return integer index number of KNMI station closest to given coordinates.
    
    Parameters
    ----------
    latitude : float
        Latitudinal coordinate
    longitude : float
        Longitudinal coordinate
    print_loc : bool
        If True, print the found KNMI station in terminal
    knmi_locs_path : str
        Local file path of .txt file with knmi locations


    Returns
    -------
    int : The index of the closest KNMI station based on the distance to the 
        coordinates specified.
    
    """
    knmi_locs = pd.read_csv('data/knmi_station_locations.txt',
                            index_col=0, skipinitialspace=True)

    loc_latlong = [latitude, longitude]
    knmi_latlong = np.array(knmi_locs[['LAT', 'LONG']])
    dist = np.linalg.norm(loc_latlong - knmi_latlong, axis=1)

    closest_stn = knmi_locs['STN'].iloc[dist == dist.min()]

    if print_loc:
        print('Found closest KNMI station: ' + str(closest_stn.values[0])
              + ' in ' + closest_stn.index.values[0])

    return int(closest_stn[0])


def plot_dashboard(
        data,
        time_now,
        shape=(4, 1),
        axs=None,
        data_title=None,
        title=False,
        plot=False,
        alpha=0.7,
        hide_spines=[]):
    """Generate dashboard overview plot displaying data values over time.
    
    Parameters
    ----------
    data : dict
        Dictonary with multiple nested dictionaries from datasets to be plotted.
        The nested dicts must containing key ``data`` and 
            data : str
                Name of the data to be plotted
            min : (str, float), optional
                When a numer is specified, use it as a minimum value
            max : (0)
    time_now : 
        Current Time in simulation, at which a vertical line will be drawn in 
        the plot. 
    shape : tuple of ints
        Tuple with number of rows and cols of the columns.
    axs : matplotlib axes or None, default: None
        If None, make a new matplotlib plot according to the specified shape.
        If matplotlib axes are given, use those axes as the canvas for the data
        plots.
    data_title : str
        String to use as title of the dashboard plot
    title : bool
        To add the title of the subplots or not.
    plot : bool
        If True, not only save the plot but also plot it.
    alpha : float
        Opacity of the plot-lines of the data.
    hide-spines : list of strings
        List of strings referring to which spines need to be hidden.

    Returns
    -------
    axs : matplotlib axes
        The input data plotted in a gives set of axes or on a newly created plot
    """

    if axs is None:
        nrows, ncols = shape

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                sharex=True, edgecolor='g')

        axs = axs.ravel()

        if data_title is not None:
            axs[0].set_title(data_title)

    for i, sub_plot_data in enumerate(data):
        for sub_data in data[sub_plot_data]["data"]:
            plot_data = data[sub_plot_data]["data"][sub_data]
            if isinstance(plot_data, dict):

                min_data = plot_data[(
                    [string for string in plot_data if 'min' in string][0])].values
                max_data = plot_data[(
                    [string for string in plot_data if 'max' in string][0])].values
                axs[i].fill_between(np.arange(0, 1, 1 / len(min_data)),
                                    min_data,
                                    max_data,
                                    alpha=0.2, label=sub_data + " min / max")
            else:
                if isinstance(plot_data, pd.Series):
                    values = plot_data.values
                else:
                    values = plot_data

                try:
                    iter(values)
                    pass
                except BaseException:
                    values = np.array([values])

                if len(values) != 96:
                    values = np.concatenate(
                        [values, [np.nan] * (96 - len(values))])
                axs[i].plot(
                    np.arange(
                        0,
                        1,
                        1 / len(values)),
                    values,
                    label=sub_data,
                    alpha=alpha,
                    scalex=False)

        axs[i].set_xlim(0, 1)

        if title:
            axs[i].set_title(sub_plot_data)
        axs[i].margins(0.01, 0.15)
        for spine in hide_spines:
            axs[i].spines[spine].set_visible(False)

        axs[i].legend(prop={'size': 8})

        try:
            lims = data[sub_plot_data]["lims"]
            axs[i].set_ylim(lims)
        except Exception:
            pass

        axs[i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        axs[i].xaxis.set_major_locator(mpl.dates.HourLocator(interval=6))
        if i != len(data) - 1:
            axs[i].set_xticklabels([])
            axs[i].xaxis.set_ticks_position('none')
        axs[i].axvline(time_now, color='red', alpha=0.4)
        if (i + 1) % shape[0] == 0:
            ymin = 0
        else:
            ymin = -1

    if plot:
        plt.show(block=False)

    return axs


def create_circular_mask(h, w, center=None, radius=None):
    """Create a 2D circular mask of booleans with specified height and width.
    
    Parameters
    ----------
    h : Union[int, float]
        The height of the circle to be drawn
    w : Union[int, float]
        The width of the circle to be drawn
    center : tuple of 2 ints, optional
        the center point of the circle
    radius : Union[float, int], optional

    Returns
    -------
    mask : ndarray
        A 2D boolean array containting True inside a circle and False outside.
    
    References
    -------
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array"""
    # use the middle of the image
    if center is None:
        center = (int(w / 2), int(h / 2))

    # use the smallest distance between the center and image walls
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask


# @numba.njit(parallel=True, fastmath=True)
# def coo_multiplication(coo_arr, data, upper_triangular=False):
#     result = np.zeros((int(coo_arr[:, 0].max()) + 1, 6))
#     for idx in numba.prange(len(coo_arr)):
#         i1, i2, f1, f2, v = coo_arr[idx, :]
#         result[int(i1), int(f1)] += data[int(i2), int(f2)] * v
#         if upper_triangular:
#             result[int(i2), int(f2)] += data[int(i1), int(f1)] * v
#     return result

# @numba.njit(parallel=True)
# def coo_multiplication2(coo_arr, data, upper_triangular=False):
#     result = np.zeros((int(coo_arr[:, 0].max()) + 1, 6))
#     for idx in numba.prange(len(coo_arr)):
#         i1, i2, f1, f2, v = coo_arr[idx, :]
#         result[int(i1), int(f1)] += data[int(i2), int(f2)] * v
#         if upper_triangular:
#             result[int(i2), int(f2)] += data[int(i1), int(f1)] * v
#     return result

def get_mask_stats(masks):
    """
    Calculate statistics from a dict containing 2D boolean masks.
    
    Parameters
    ----------
    masks : dict of boolean arrays
        Dictionary containing a boolean mask per ground-use type.


    Returns
    -------
    build_frac : float
        The fraction of the ground that is used for buildings
    imperv_frac : float
        The fraction of the ground that is used for roads, concrete and other 
        paved ground types.
    """
    total_pixels = list(masks.values())[0].size
    build_frac = masks['buildings'].sum() / total_pixels
    imperv_frac = (masks['roads'].sum() +
                   masks['concrete'].sum()) / total_pixels

    return build_frac, imperv_frac


def get_hourly_dataframe_offline(path):
    """
    Load a dataframe from a local path and set the datetime as index.
    
    Parameters
    ----------
    path : str
        Local path to dataframe.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame read from the local drive. 
    """
    def date_parser(date, hh, subtract_hour=1):
        """Date parser that can subtract hours to correct for timezones."""
        return pd.Timestamp(date).replace(hour=int(hh) - subtract_hour)

    df = pd.read_csv(path, skipinitialspace=True, parse_dates=[
                     ['YYYYMMDD', 'HH']], date_parser=date_parser)
    df = df.rename(columns={"YYYYMMDD_HH": "Date"})
    df.set_index(["Date"], inplace=True)
    return df


def frontal_areas(
        idcs_dict,
        faces_idcs,
        area_map,
        wind_dir_deg,
        area_face=1,
        _print=False):
    """
    Calculate frontal area densities of trees and buildings and their combination.
    
    The frontal area density of buildings is an important indicator of wind 
    reductions in urban areas. Frontal areas are determined by calculating the 
    perpendicular surfaces towards the wind direction. 
    
    Parameters
    ----------
    idcs_dict : dict
        Dictionary that contains surface-types as keys and stores the indices
        per surface-type. 
    faces_idcs : nested list of ints
        Nested list that per cardinal orientation contains a list of indices 
        of the elements that have an outer face in that direction.
    area_map : float
        Number of elements in groud surface.
    wind_dir_deg : float
        Direction of the wind, in degrees.

    Returns
    -------
    front_dens_tree : float
        Frontal area density of trees.
    front_dens_building : float
        Frontal area density of buildings.
    front_dens_total : float
        Total frontal area density, so trees and buildings combined. 
    
    References
    ----------
    A standardized Physical Equivalent Temperature urban heat map at 1-m
    spatial resolution to facilitate climate stress tests in the Netherlands
    S. Koopmans, B.G. Heusinkveld and G.J. Steeneveld
    """
    frontal_area_trees = 0
    frontal_area_buildings = 0

    if 180 <= wind_dir_deg <= 360:
        trees_faces_count = len(
            np.intersect1d(
                idcs_dict['trees'],
                faces_idcs[0]))
        frontal_area_trees += trees_faces_count * area_face * \
            abs(math.sin(math.radians(wind_dir_deg)))**2

        buildings_faces_count = len(
            np.intersect1d(
                idcs_dict['buildings'],
                faces_idcs[0]))
        frontal_area_buildings += buildings_faces_count * \
            area_face * abs(math.sin(math.radians(wind_dir_deg)))**2

    if 0 <= wind_dir_deg <= 180:

        trees_faces_count = len(
            np.intersect1d(
                idcs_dict['trees'],
                faces_idcs[1]))
        frontal_area_trees += trees_faces_count * area_face * \
            abs(math.sin(math.radians(wind_dir_deg)))**2

        buildings_faces_count = len(
            np.intersect1d(
                idcs_dict['buildings'],
                faces_idcs[1]))
        frontal_area_buildings += buildings_faces_count * \
            area_face * abs(math.sin(math.radians(wind_dir_deg)))**2

    if (wind_dir_deg >= 270) or (wind_dir_deg <= 90):

        trees_faces_count = len(
            np.intersect1d(
                idcs_dict['trees'],
                faces_idcs[4]))
        frontal_area_trees += trees_faces_count * area_face * \
            abs(math.cos(math.radians(wind_dir_deg)))**2

        buildings_faces_count = len(
            np.intersect1d(
                idcs_dict['buildings'],
                faces_idcs[4]))
        frontal_area_buildings += buildings_faces_count * \
            area_face * abs(math.cos(math.radians(wind_dir_deg)))**2

    if 90 <= wind_dir_deg <= 270:

        trees_faces_count = len(
            np.intersect1d(
                idcs_dict['trees'],
                faces_idcs[5]))
        frontal_area_trees += trees_faces_count * area_face * \
            abs(math.cos(math.radians(wind_dir_deg)))**2

        buildings_faces_count = len(
            np.intersect1d(
                idcs_dict['buildings'],
                faces_idcs[5]))
        frontal_area_buildings += buildings_faces_count * \
            area_face * abs(math.cos(math.radians(wind_dir_deg)))**2

    front_dens_tree = frontal_area_trees / area_map
    front_dens_building = frontal_area_buildings / area_map
    
    front_dens_total = 0.6 * front_dens_building + 0.3 * front_dens_tree + 0.015

    return front_dens_tree, front_dens_building, front_dens_total


def duration(start):
    """
    Given a startpoint in time, return the duration in seconds until now as a string.
    
    Parameters
    ----------
    start : float
        Float of the start time

    Returns
    -------
    str : Difference in seconds between `start` time and now.
    """
    return str(datetime.timedelta(seconds=time.time() - start))

def _make_continuous(arr1, arr2):
    """
    Make a continuous array from the two input arrays and view it is its 
    original shape.
    """
    arr1_view = np.ascontiguousarray(arr1).view(
        [('', arr1.dtype)] * arr1.shape[1])

    try:
        arr2_view = np.ascontiguousarray(arr2).view(
            [('', arr2.dtype)] * arr2.shape[1])
    except IndexError:
        arr2_view = np.ascontiguousarray(arr2).view(
            [('', arr2.dtype)] * arr2.shape[0])
        
    return arr1_view, arr2_view

def intersecting_rows(arr1, arr2, assume_unique=True):
    """
    Find the intersection of rows between input arr1 and arr2.

    Parameters
    ----------
    arr1 : numpy.ndarray
        2D numpy array 
    arr2 : numpy.ndarray
        2D numpy array
    

    Returns
    -------    
    values_1isin2 : np.ndarray
        The values of `arr1` that are found in `arr2`
    idx_1isin2 : np.ndarray
        An array with elements from where rows of `arr1` are found in `arr2`.
    idx_2isin1 : np.ndarray
        An array with elements from where rows of `arr2` are found in `arr1`.
        
    """
    
    arr1_view, arr2_view = _make_continuous(arr1, arr2)

    values_1isin2, idx_1isin2, idx_2isin1 = np.intersect1d(arr1_view, arr2_view, assume_unique=assume_unique, return_indices=True)

    return values_1isin2, idx_1isin2, idx_2isin1


def isin_flattened(arr1, arr2, assume_unique=False):
    """
    Return a numpy array with bools of which values of arr1 are also in arr2.
    
    Parameters
    ----------
    ----------
    arr1 : np.ndarray
        2D numpy array 
    arr2 : np.ndarray
        1D or 2D numpy array

    Returns
    -------
    bools_1in2 : np.ndarray
    """

    arr1_view, arr2_view = _make_continuous(arr1, arr2)

    bools_1in2 = np.isin(arr1_view, arr2_view, assume_unique=assume_unique)

    return bools_1in2

@numba.njit
def make_2d(arraylist):  
    """From a list of xyz coordinates, make a 2d array with the xyz coordinates.
    
    Parameters
    ----------
    arraylist : list or ndarray
        Numpy array or nested list with coordinates
        

    Returns
    -------
    arr_2d : ndarray
        2D array that has all elements from `arraylist`, but then in one array.
        
    
    """
    n = len(arraylist)
    k = arraylist[0].shape[0]
    arr_2d = np.zeros((n, k))
    for i in range(n):
        arr_2d[i] = arraylist[i]
    return(arr_2d)


def idcs_to_booleanlist(size, idcs):
    """Return a list of booleans from a list of indices.
    
    Parameters
    ----------
    size : tuple of int
        Size of the resulting arraylist.
    idcs : array-like
        List or numpy array with indices of the locations at which the 
        resulting array must be True
    

    Returns
    -------
    mask_lst : np.ndarray
        1D Numpy array with True on the locations given by `idcs`.
    """
    mask_lst = np.zeros(size).astype(bool)
    if len(idcs) == 0:
        return mask_lst
    else:
        mask_lst[idcs] = True
        return mask_lst


# %% 2D numpy array functions


def interp_missing(z_arr, method='linear'):
    """
    Interpolate missing values in a 2d numyp array.
    
    Parameters
    ----------
    z_arr : np.ndarray
        A 2D numpy array with missing values that need to be filled in.
    method : {'linear', 'nearest', 'cubic'}
        Method to interpolate the input array and find it's missing values.

    Returns
    -------
    zs_interp : np.ndarray
        Completed array.
    """

    x_indx, y_indx = np.meshgrid(np.arange(0, z_arr.shape[1]),
                                 np.arange(0, z_arr.shape[0]))

    # mask all invalid values
    zs_masked = np.ma.masked_invalid(z_arr)

    # print('zs_masked',zs_masked.shape)

    # retrieve the valid, non-Nan, defined values
    valid_xs = x_indx[~zs_masked.mask]
    valid_ys = y_indx[~zs_masked.mask]
    valid_zs = zs_masked[~zs_masked.mask]

    # generate interpolated array of z-values
    zs_interp = scipy.interpolate.griddata(
        (valid_xs, valid_ys), valid_zs.ravel(), (x_indx, y_indx), method=method)

    return zs_interp

# %% Dictionary related functions


def save_dict(dct, path, indent=4):
    """
    Save a dict as JSON file.
    
    Parameters
    ----------
    dct : dict
        Input directory.
    path : str
        Local path to save the input dict to.
    indent : int
        Indentation when writing the dict to a file.
        """
    if not os.path.isfile(path):
        print("Created new file:", path)
    with open(path, 'w') as file:
        json.dump(dct, file, indent=indent)
 

def load_dict(path):
    """
    Load JSON file to dict.
    
    Parameters
    ----------
    path : str
        Local path from where to load a dictionary.

    Returns
    -------
    dct : dict
        Loaded dictionary from specified path.
    """

    while True:
        try:
            with open(path, 'r') as file:
                dct = json.load(file)
            print(path, 'loaded')
            break
        except FileNotFoundError:
            user_input = input(
                f'{path} not found, enter "new" to create a new file or press enter to retry...\n')
            if user_input == 'new':
                dct = {}
                save_dict(dct, path)
                break

    return dct


def get_key(dct, key, path, save=True, keytype=str, always_ask=False):
    """
    Get value from a dict, otherwise prompt to fill in the value by user input.
    
    Parameters
    ----------
    dct : dict
        Dictionary to load the `key` from, or to add it to if it can not be 
        found in the dict
    key : str
        Key to get from input dict or to add.
    path : str
        Local directory path for writing the dict when a new key is added.
    save : bool
        If True, save the dict when a new key is added.
    keytype : {str, bool}
        Type of the `key` to be loaded from data or prompted user input.
    always_ask : bool, default : False
        If True, always ask the user to define the value of `key`, if False, 
        try to load the `key` value, otherwise prompt it.

    Returns
    -------"""
    try:
        if always_ask:
            raise Exception
        value = dct[key]
    except (KeyError, Exception):
        if keytype is bool:
            while True:
                user_input = input(
                    "Enter True or False for " + str(key) + ":\n")
                if user_input == "True":
                    value = True
                    break
                if user_input == "False":
                    value = False
                    break
                print('Incorrect input')
        if keytype is not bool:
            user_input = input("Enter value for " + str(key) + ":\n")
            value = type(user_input)
        dct.update({key: value})
        if save:
            save_dict(dct, path)

    return value


# def set_key(dct, dct_update, path):
#     """
#     Update a key in a dict and save to specified path.
    
#     Parameters
#     ----------
#     dct : dict
#         Dictionary to be updated.
#     dct_update : tuple
#

#     Returns
#     -------"""
#     dct.update(dct_update)
#     save_dict(dct, path)





# @profile
# def split_subtract(arr, nr_splits, lenth_array):
#     """
    

#     Parameters
#     ----------
#     arr : TYPE
#         DESCRIPTION.
#     nr_splits : TYPE
#         DESCRIPTION.
#     lenth_array : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     split_subtr : TYPE
#         DESCRIPTION.

#     """

#     split_vals = np.array([lenth_array * (i) for i in range(nr_splits)])
#     split_loc = np.searchsorted(arr, split_vals[1:], side='left')
#     splitted = np.split(arr, split_loc)

#     split_subtr = np.subtract(splitted, split_vals[:, np.newaxis])

#     return split_subtr


# def round_to_half(number):
#     """Round a number to the closest half integer."""
#     return np.round(np.subtract(number,0.5))+0.5


# %% Simple numba fucntions

@numba.njit
def sum_nested_list_length(nested_list):
    # Initialize a variable to store the sum
    total_length = 0

    # Iterate through the outer list
    for lst in nested_list:
        # Iterate through the inner list
        total_length += len(lst)

    return total_length

@numba.njit
def list_to_2d_array(arraylist: list) -> np.ndarray:
    # Get the shape of the 2D array
    num_arrays = len(arraylist)
    array_len = arraylist[0].shape[0]
    # Create a 2D array with the correct shape
    array_2d = np.empty((num_arrays, array_len), dtype=arraylist[0].dtype)
    # Fill the 2D array with the 1D arrays
    for i in range(num_arrays):
        array_2d[i] = arraylist[i]
    return array_2d

# %% Simple numpy functions

def round_to_half(arr):
    """
    Round elements of array to the closest half-integer.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the values will be rounded to half-integers.

    Returns
    -------
    np.ndarray
        Rounded array.
    """
    return np.add(np.round(np.floor(arr)), 0.5)


def roundto(arr, r):
    """
    Round elements of array to a specified precision.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the values will be rounded to the desired precision.
    r : float
        Rounding precision, e.g. `0.1`, `0.5`, `2`.

    Returns
    -------
    np.ndarray
        Rounded array.
    """
    return np.array(np.multiply(np.round(np.divide(arr, r)), r))



def array2d_to_array3d(array2d, coo, dx=1, dy=1, dz=1, shape=None):  # data2d_to_data3d
    """
    Create a 3D array from a 2D array and coordinate array.
    
    Given a 2D array and a coordinate array, this function creates a 3D array
    by placing the elements of the 2D array at the positions specified by the 
    coordinate array and filling the remaining positions with zeros. The 
    spacing between the coordinates along the x, y, and z axis can be 
    specified by the dx, dy, and dz arguments respectively.
    
    Parameters
    ----------
    array2d : np.ndarray
        The 2D input array.
    coo : np.ndarray
        The coordinate array with shape (N, 3) where N is the number of elements
        in the 2D array, specifying the x, y, z indices of the elements in the 3D array
    dx : int, optional
        The spacing between coordinates along the x axis, by default 1
    dy : int, optional
        The spacing between coordinates along the y axis, by default 1
    dz : int, optional
        The spacing between coordinates along the z axis, by default 1
        
    Returns
    -------
    np.ndarray
        The resulting 3D array.
    """
    
    if shape is None:
        
        shape = (int(np.max(coo[:, 0])) + dx,
                           int(np.max(coo[:, 1])) + dy,
                           int(np.max(coo[:, 2])) + dz)
        
    array3d = np.zeros(shape)

    for i, (x, y, z) in enumerate(coo):
        array3d[int(x), int(y), int(z)] = array2d[i]

    return array3d


# %% Ray tracing functions

# @profile
def calc_rays_between_points(
        coords_from_el,
        coords_to_els,
        max_ray_length=100,
        _round_to_half=False):
    """
    Compute a 3D array of rays between 3D coordinates points.
    
    Given a set of start coordinates and a set of end coordinates, this function
    generates an array of rays that connects the start and end coordinates.
    Each ray is represented by an array of 3D coordinates that lie on the line
    segment between the start and end coordinates. The number of coordinates in
    each ray is determined by the maximum distance between the start and end
    coordinates, subject to the maximum allowed ray length.
    
    Parameters
    ----------
    coords_from_el : np.ndarray
        A 3D coordinate of shape (3,) representing the starting point of the rays
    coords_to_els : np.ndarray
        A 2D array of shape (N, 3) representing the end coordinates for N rays
    max_ray_length : int, optional
        Maximum length of the rays, by default 100
    _round_to_half : bool, optional
        Whether to round the coordinates to half integers, by default False
        
    Returns
    -------
    np.ndarray
        A 3D array of shape (N, max_ray_length, 3) representing the rays
    """
    longest_dist = math.ceil(np.max(abs(coords_from_el - coords_to_els)))

    ray_length = int(np.min([longest_dist, max_ray_length]))

    rays = np.linspace(start=np.array([coords_from_el[0], coords_from_el[1], coords_from_el[2]])[:, np.newaxis],
                       stop=[coords_to_els[:, 0], coords_to_els[:, 1], coords_to_els[:, 2]],
                       num=ray_length,
                       endpoint=False)

    if _round_to_half:
        rays = round_to_half(rays)

    return np.transpose(rays, axes=(2, 0, 1))


# @numba.njit
# def calc_rays_between_points_jit(
#         coords_from_el,
#         coords_to_els,
#         max_ray_length=100,
#         _round_to_half=False):
#     """
#     From the xyz coordinates of a single element, trace rays to a set of elements.
    
    
#     Parameters
#     ----------


#     Returns
#     -------"""
#     longest_dist = math.ceil(np.max(np.abs(coords_from_el - coords_to_els)))

#     ray_length = int(min([longest_dist, max_ray_length]))

#     rays = custom_linspace_arr(start=coords_from_el,
#                                stop=coords_to_els.T,
#                                num=ray_length)

#     if _round_to_half:
#         rays = np.add(np.floor(rays), 0.5)

#     return np.transpose(rays, axes=(2, 0, 1))


# @numba.jit
# def calc_rays_between_points_jit2(
#         coords_from_el,
#         coords_to_els,
#         max_ray_length=100,
#         _round_to_half=False):
#     """
#     From the xyz coordinates of a single element, trace rays to a set of elements.
    
#     Parameters
#     ----------


#     Returns
#     -------"""
#     longest_dist = math.ceil(np.max(np.abs(coords_from_el - coords_to_els)))

#     # ray_length = int(np.min([longest_dist, max_ray_length]))

#     # rays = np.linspace(start=np.array([coords_from_el[0], coords_from_el[1], coords_from_el[2]])[:, np.newaxis],
#     #                    stop=[coords_to_els[:, 0], coords_to_els[:, 1], coords_to_els[:, 2]],
#     #                    num=ray_length,
#     #                    endpoint=False)

#     # if _round_to_half:
#     #     rays = np.add(np.round(np.floor(rays)), 0.5)

#     return longest_dist


# @numba.njit(parallel=True)
# def custom_linspace_arr(start, stop, num, round_to_half=False):
#     """
#     Trace a ray between a startpoint and endpoint, returning an array.
    
#     Parameters
#     ----------


#     Returns
#     -------"""

#     arr = np.zeros((num, ) + stop.shape)

#     for j in range(stop.shape[-1]):
#         for i in range(3):

#             lower = start[i]
#             upper = stop[i, j]

#             for k in range(num):
#                 val = lower + (k * (upper - lower) / num)
#                 # if round_to_half:
#                 #     val
#                 arr[int(k), int(i), int(j)] = val

#     return arr


# @numba.njit
# def custom_linspace_lst(start, stop, num):
#     """
#     Trace a ray between a startpoint and endpoint, returning a list.
    
#     Parameters
#     ----------


#     Returns
#     -------"""

#     lst = []

#     for j in range(stop.shape[-1]):
#         sub_lst = []
#         for i in range(3):
#             lower = float(start[i])
#             upper = float(stop[i, j])

#             # print(f'lower: {type(lower)}, upper: {type(upper)}')
#             vals = []
#             for k in range(num):

#                 vals.append([lower + k * (upper - lower) / num])

#             sub_lst.append(vals)

#         lst.append(sub_lst)

#     # arr = np.array(lst)

#     return lst
    # return np.transpose(arr, axes=(2, 1, 0))


# %% File organisation functions

def load_pickled_list(path, name, length):
    # DOC missing
    data_load = np.load(path, allow_pickle=True)[name]
    data = [[] for _ in range(data_load.shape[0])]
    for i in range(data_load.shape[0]):
        data[i] = np.where(data_load[i])[0]
    
    load_factor = length / len(data)
    data_temp = [[] for _ in range(length)]
    if load_factor > 1:
        data_temp[::int(load_factor)] = data
    if load_factor <= 1:
        data_temp = data[::int(1 / load_factor)]
    
    data = data_temp
    
    return data


def save_image(data, fn, cmap="Greys"):
    """
    Save an image to a file.

    Parameters
    ----------
    data : numpy array
        The data for the image. Can be a 2D or 3D array.
    fn : str
        The filename to save the image to.
    cmap : str, optional
        The colormap to use for the image. Default is "Greys".

    Returns
    -------
    None"""

    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if len(data.shape) == 2:
        ax.imshow(data, cmap=cmap)
    if len(data.shape) == 3:
        ax.imshow(data)
    plt.savefig(fn, dpi=height)
    plt.close()


def images_to_vid(
        save_name,
        load_path='./',
        save_path='./',
        dates=None,
        post_label='',
        vid_duration=None,
        fps=None,
        sort=True,
        skip_ends=False):
    """
    Create a video from a series of images, that can be filtered by date and post label
    
    Parameters
    ----------
    save_name : str
        The name to save the video file as.
    load_path : str, optional
        The path to the location of the images. Default is './'
    save_path : str, optional
        The path to save the video to. Default is './'
    dates : list of str, optional
        A list of dates to filter images by. Default is None.
    post_label : str, optional
        A label to filter images by. Default is ''.
    vid_duration : int, optional
        The length of the video in seconds. If None it will be determined by the number of images.
    fps : int, optional
        The frames per second of the video. If None it will be determined by the number of images.
    sort : bool, optional
        If True, the images will be sorted. Default is True
    skip_ends : bool, optional
        If True, the first and last images will be skipped. Default is False
    
    Returns
    -------
    None
    """
    
    start = time.time()

    img_array = []

    if dates is not None:
        files = []
        for date in dates:
            file_loc = f'{load_path}{date}*{post_label}.png'
            files += glob.glob(file_loc)
    else:
        file_loc = f'{load_path}{date}*{post_label}.png'
        files = glob.glob(file_loc)
    if len(files) == 0:
        print(f'No .png files found at {load_path}. No .mp4 video made.')
        return
    if sort:
        files.sort()

    if skip_ends:
        files = files[1:-1]

    for filename in files:
        try:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        except AttributeError:
            print('Failed to read ' + filename)
            pass

    if vid_duration is not None:
        fps = len(img_array) / vid_duration

    if fps is None and vid_duration is None:
        fps = 15

    fps = int(min(30, max(8, fps)))
    print('fps set to ', fps)

    save_path += str(save_name)

    if dates is not None:
        dates.sort()
        save_path += ', ' + dates[0]
        save_path += ' to ' + dates[-1]
    save_path += ' - ' + post_label + '.mp4'

    out = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out and out.release()

    cv2.destroyAllWindows()
    print('Saving video to ' + save_path + ' took ' + duration(start))


def sizeof_fmt(num, suffix='B'):
    """Display filesize to be human readable, using binary prefixes.
    
    Parameters
    ----------
    num : int
        Integer of number of bytes
    suffix : str, optional
        Suffix after the binary prefix, defautl is 'B'
        

    Returns
    -------
    str : Human readable string of the object file size
    
    by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# %% Coordinates


def elevationmap_to_coo(elev_map, street_height, res):
    """
    Creates a coordinates list (COO) from a 2D elevation map.
    
    elev_map : numpy array
        The elevation map to convert.
    street_height : int
        The height of streets.
    res : int
        The resolution of the map.
        
    Returns
    -------
    coo : numpy array
        The 3D coordinates of the map.
    """

    coords_step = 1
    elev_map_round = roundto(elev_map / res, coords_step) + 1

    nr_blocks = int(np.sum(elev_map_round))
    coo = np.zeros((nr_blocks, 3))

    mesh_coord = np.add(np.meshgrid(range(0, len(elev_map_round)), range(
        0, len(elev_map_round))), coords_step / 2)

    i = 0
    for ix in range(len(elev_map_round)):
        for iy in range(len(elev_map_round)):
            for iz in range(int(elev_map_round[ix, iy])):
                if i < nr_blocks:
                    coo[i] = [mesh_coord[0][ix, iy], mesh_coord[1]
                              [ix, iy], (iz + 0.5) * coords_step]
                else:
                    coo = np.vstack(
                        coo,
                        [mesh_coord[0][ix, iy],
                         mesh_coord[1][ix, iy],
                         (iz + 0.5) * coords_step])

                i += 1
    return coo


def coo_to_ndarray(
        coo,
        interpolate_missing=False,
        plot=False,
        cmap='viridis',
        shape=None):
    """Convert COOrdinate list to 2d NumPy array. 
    
    Parameters
    ----------
    coo : numpy array
        The 3D coordinates to convert.
    interpolate_missing: bool, optional
        Whether to interpolate missing values. Defaults to False.
    plot : bool, optional
        Whether to plot the resulting array. Defaults to False.
    cmap : str
        The colormap to use when plotting. Defaults to 'viridis'.
    shape : tuple, optional
        The shape of the resulting array. Defaults to None.

    Returns
    -------
    z_array : numpy array
        The resulting 2D array.
    """
    if shape is None:
        shape = tuple(
            int(math.ceil(np.max(coo[:, :2]) + 0.5)) for _ in range(2))

    x = coo[:, 0].astype(int)
    y = coo[:, 1].astype(int)

    if coo.shape[1] == 2:
        z = np.ones(len(x), dtype=bool)
    elif coo.shape[1] == 3:
        z = coo[:, 2]
    else:
        raise Exception('Enter xy- or xyz coordinate list.')

    z_array = np.nan * np.empty(shape)
    z_array[y, x] = z

    if interpolate_missing:
        z_array = interp_missing(z_array, method='linear')

    if plot:
        cmap = mpl.cm.get_cmap(cmap)

        plt.imshow(z_array, cmap=cmap)
        plt.show(block=False)

    return z_array


@numba.njit(parallel=True, fastmath=True)
def coo_multiplication(coo_arr, vf_arr, data, nr_els, upper_triangular=False):
    """
    Calculate radiation transfer between faces of elements coupled by a view-factor.
    
    Calculates the radiation transfer between faces of elements and their 
    surrounding elements. The radiation transfer is dependend on their respective
    view-factors between the faces. The function is jitted for speed optimisation.
    
    Parameters
    ----------
    coo_arr : numpy.ndarray
        A 2D array of shape (N, 4), where N is the number of all faces of 
        the elements. Each row consists of i1, i2, f1, and f2. Where i is the 
        index of the element and f is the face of that element.
    vf_arr : numpy.ndarray
        A 1D array of shape (N,) containing the view-factors for each element in `coo_arr`.
    data : numpy.ndarray
        A 2D array of shape (nr_els, 6) containing data to be multiplied by the view factor.
    nr_els : int
        The number of elements between which radiation tranfer must be calculated.
    upper_triangular : bool
        If True, uses upper triangular matrix for the calculation. Default is False.

    Returns
    -------
    result : numpy.ndarray
        A 2D array of shape (nr_els, 6) containing the result of the radiation calculation.
    """
    result = np.zeros((nr_els, 6))
    for idx in range(len(coo_arr)):
        i1, i2, f1, f2 = coo_arr[idx, :]
        vf = vf_arr[idx]
        result[int(i1), int(f1)] += data[int(i2), int(f2)] * vf
        if upper_triangular:
            result[int(i2), int(f2)] += data[int(i1), int(f1)] * vf
    return result


def coo_reduce_accuracy(coords, data, reduce_acc, min_val_start=5e-03):
    """
    Remove enties in a coordinates list so that the sum of the accociated
    data differs a specified percentage from the original sum. 
    
    Repeatedly remove entries from `data` by only keeping entries that are 
    smaller than some `min_val`. Then, check if the difference between the
    original sum of `data` and the sum of the remaining entries of `data` 
    is smaller than the desired accuracy `reduce_acc`. If not, make `min_val` 
    smaller and try over. Repeat until the difference of the sums is smaller 
    than`reduce_acc`.

    Parameters
    ----------
    coords : np.ndarray
        2D Numpy array, a coordinates list
    data : np.ndarray
        1D Numpy array of data values corresponding to the coordinates.
    reduce_acc : float
        The percentage by which the sum of the data should differ from the 
        original sum after removing entries. Should be between 0 and 1.
    min_val_start : float, optional
        The minimum value for a data entry to be considered. Defaults to 5e-03.
    
    Returns
    -------
    coords_reduced : np.ndarray
        2D Numpy array where entries are deleted.
    data_reduced : Lnp.ndarray
        1D Numpy array where entries are deleted.
    diff : float
        The percentage by which the sum of the data differs from the original 
        sum after removing entries.
    """
    min_val = min_val_start
    begin_sum = data.sum()

    nnz = len(coords)

    while True:
        if nnz == 0:
            coords_reduced = coords
            data_reduced = data
            break

        _coo_sum = data * (data > min_val)
        diff = (begin_sum - _coo_sum.sum()) / begin_sum
        if diff < reduce_acc:
            coords_reduced = coords[(data > min_val)]
            data_reduced = data[(data > min_val)]
            print(
                'Omitted all View Factors smaller than %f decreased precision by %.3f %%' %
                (min_val, reduce_acc))
            print(
                'While %.1f percent of the non-zeros remain' %
                (len(coords_reduced) / nnz * 100))

            break
        min_val /= 2

    return coords_reduced, data_reduced, diff


# !!! deze werkt voor 1 xyz coord, niet zoals die andere functie voor lijst van xyz coordinaten van meerdere elementen
@numba.njit
def coo_to_idcs_jit(coo, idcs_3D, remove_negs=True):
    """
    Convert a 2D numpy array with 3D coordinates to a list of indices. 
    The function is jitted for speed optimisation.
    
    Parameters
    ----------
    coords : np.ndarray
        A 2D Numpy array of 3D coordinates.
    idcs_3D : numpy.ndarray, shape (X, Y, Z)
        A 3D array containing the indices corresponding to the given coordinates.
    remove_negs : bool, optional
        Indicates if negative indices should be removed from the resulting list. Defaults to True.

    Returns
    -------
    lst : list
        List of indices corresponding to the given coordinates.
    """
    lst = []
    for (x, y, z) in coo:
        idx = idcs_3D[x, y, z]
        if remove_negs:
            if idx >= 0:
                lst.append(idx)
        else:
            lst.append(idx)

    return lst


def coo_to_idcs(coo, idcs_3D, remove_negs=True):
    """
    Convert a 2D numpy array containing 3D coordinates to a list of indices. 
    
    Parameters
    ----------
    coo : np.ndarray
        A 2D Numpy array of 3D coordinates.
    idcs_3D : numpy.ndarray, shape (X, Y, Z)
        A 3D array containing the indices corresponding to the given coordinates.
    remove_negs : bool, optional
        Indicates if negative indices should be removed from the resulting list. Defaults to True.

    Returns
    -------
    lst : list
        List of indices corresponding to the given coordinates.
    """

    if np.shape(coo)[1] == 2: 
        idcs = idcs_3D[coo[:, 0], coo[:, 1], :]

    elif np.shape(coo)[1] == 3:  
        idcs = idcs_3D[coo[:, 0], coo[:, 1], coo[:, 2]]

    if remove_negs:
        idcs = idcs[idcs >= 0]

    return list(idcs)


def coos_to_idcs(coos, idcs_3D, remove_negs=True):
    """
    Convert multiple 2D numpy arrays containing 3D coordinates to a list of indices. 
    
    Parameters
    ----------
    coos : Union[np.ndarray, List[np.ndarray]]
        A nested numpy array or a list containing numpy arrays, containing 2D 
        arrays of 3D coordinates.
    idcs_3D : numpy.ndarray, shape (X, Y, Z)
        A 3D array containing the indices corresponding to the given coordinates.
    remove_negs : bool, optional
        Indicates if negative indices should be removed from the resulting list. Defaults to True.

    Returns
    -------
    lst : list
        List of indices corresponding to the given coordinates.
    """

    all_idcs = []
    for coo in coos:
        idcs = coo_to_idcs(coo=coo, idcs_3D=idcs_3D, remove_negs=remove_negs)

        all_idcs.append(list(idcs))

    return all_idcs


@numba.njit
def coo_mid_to_side_jit(coords_mid, face, dx=1, dy=1, dz=1, eps=1e-15):
    """
    Transpose the coordinates from the middle to the side of the element, 
    depending on the `face` of the element. Jitted for speed optimisation.
    
    Parameters
    ----------
    
    coords_mid : numpy.ndarray, shape (N, 3)
        Coordinates matrix in midpoint format.
    face : int
        The face for which the coordinates should be transposed.
        0 : west
        1 : east
        2 : bottom
        3 : top
        4 : south
        5 : north
    dx, dy, dz : float, optional
        The dimensions of the environmental cubes. Defaults to 1.
    eps : float, optional
        A small value to prevent division by zero. Defaults to 1e-15.
    
    Returns
    -------
    numpy.ndarray
        Coordinates transposed to the sides of the elements.
    """
    if face == 0:
        return np.add(coords_mid, np.array([-(dx / 2 + eps), 0, 0]))
    if face == 1:
        return np.add(coords_mid, np.array([+(dx / 2 + eps), 0, 0]))
    if face == 2:
        return np.add(coords_mid, np.array([0, 0, -(dz / 2 + eps)]))
    if face == 3:
        return np.add(coords_mid, np.array([0, 0, +(dz / 2 + eps)]))
    if face == 4:
        return np.add(coords_mid, np.array([0, -(dy / 2 + eps), 0]))
    if face == 5:
        return np.add(coords_mid, np.array([0, +(dy / 2 + eps), 0]))



def coords_all_adjacent_elements(coo_int, shape):
    """
    Find the coordinates of adjacent elements of given coordinates.
    
    Parameters
    ----------
    coo_int : np.ndarray of ints, shape (N,3)
        Coordinates of the elements.
    shape : Tuple[int, int, int]
        Shape of the 3D grid.
    
    Returns
    -------
    coords_ngbrs : nested np.ndarray of np.ndarrays of ints, shape (N,M,3)
        Per 3D coordinate in `coo_int`, generate a numpy array containing the 
        coordinates of all M adjacent elements. Gather the all arrays in one 
        nested numpy array.
    """

    coords_ngbrs = np.empty(len(coo_int), dtype=object)

    for i, coords in enumerate(coo_int):
        ngbrs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz or abs(dx) + abs(dy) + abs(dz) >= 2:
                        continue
                    X = coords[0] + dx
                    Y = coords[1] + dy
                    Z = coords[2] + dz
                    if (0 <= X < shape[0] and
                        0 <= Y < shape[1] and
                            0 <= Z < shape[2]):
                        ngbrs.append([X, Y, Z])

        coords_ngbrs[i] = np.array(ngbrs)
    return coords_ngbrs


def coo_to_idcs3D(coo_int, shape):
    """Convert a list with xyz coordinates to a 3d array containing the indices
    on the respective xyz coordinates.
    
    Parameters
    ----------
    coo_int : np.ndarray of ints, shape (N,3)
        Coordinates of the elements.
    shape : tuple of ints (int, int, int)
        Shape of the 3D grid.
        
    Returns
    -------
    idcs_3D : np.ndarray of ints, shape `shape`
        A 3D array containing the indices corresponding to the given coordinates.
    """
    idcs_3D = -1 * np.ones(shape, dtype=int)

    for i, (x, y, z) in enumerate(coo_int):
        idcs_3D[x, y, z] = i

    return idcs_3D


# %% Mask functions

def mask_modulus(shape, mod):
    mask = np.zeros(shape, dtype=bool)
    mask[mod::mod,
         mod::mod] = True
    
    return mask

# TODO doc
def make_sensor_mask(
        masks,  
        masks_keys,
        shape, 
        shrink_pixels=1,
        initial_mask=None,
        add_modulus_sensors=False,
        ): 
    

    masks_sum = np.sum([masks[key] for key in masks_keys], axis=0).astype(bool)
    masks_sum_shrink, sensor_xy = shrink_mask(masks_sum, pixels=shrink_pixels, out_shape=shape)

    mask_result = masks_sum_shrink
    
    if initial_mask is not None:
        mask_result *= initial_mask

    sensor_xy = sensor_xy.T
    
    if add_modulus_sensors:

        # keep sensors once every X gridcells
        mask_coarse = mask_modulus(shape=shape, mod=3)
        mask_fine = mask_modulus(shape=shape, mod=1)
            
        mask_result *= (mask_fine + mask_coarse)
        
    
        
    return mask_result, sensor_xy


def interpolate_missing(array):
    """Interpolate missing values in 2D array using Scipy's interpolate.griddata method.

    Parameters
    ----------
    array : np.ndarray
        2D array containing missing values.

    Returns
    -------
    array_result : np.ndarray
        2D array with missing values interpolated.
    """
    
    array[array == np.max(array)] = np.nan
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    array_result = scipy.interpolate.griddata(
        (x1, y1), newarr.ravel(), (xx, yy), method='cubic')
    mask = np.isnan(array_result)
    array_result[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])

    return array_result


def shrink_mask(mask, pixels=1, out_shape=None):
    """
    Shrink a binary mask by a specified number of pixels.

    Parameters
    ----------
    mask : numpy ndarray
        2D binary mask.
    pixels : int, optional
        Number of pixels to shrink the mask.
    out_shape : tuple, optional
        Output shape of the mask.

    Returns
    -------
    mask : np.ndarray of bools
        The shrunken mask
    trues_locs : np.ndarray 
        Locations of the trues in the mask.
    """
    if pixels == 0:
        trues_locs = np.where(mask)
        print('Enter value greater than 0.')
        return mask, trues_locs
    for i in range(pixels):
        locs_initial_trues = np.where(mask)
        trues_not_at_border = (np.not_equal(locs_initial_trues,
                                            0) * np.not_equal(locs_initial_trues,
                                                              int(mask.shape[0] - (i + 1)))).all(0)
        loc_trues = np.array(locs_initial_trues)[:, trues_not_at_border]

        mask_crop = (mask[tuple(loc_trues + np.array([[0], [1]]))]
                     * mask[tuple(loc_trues + np.array([[0], [-1]]))]
                     * mask[tuple(loc_trues + np.array([[1], [0]]))]
                     * mask[tuple(loc_trues + np.array([[-1], [0]]))])

        trues_locs = np.array(loc_trues)[:, mask_crop]
        if out_shape is None:
            out_shape = np.shape(mask)
        new_mask = np.zeros(out_shape).astype(bool)
        new_mask[tuple(trues_locs)] = True

        mask = new_mask

    return mask, trues_locs


def reduce_res_mask(masks, reduce_res, shape):

    """
    Reduce the resolution of a multiple of masks in a dict to a lower resolution.

    Parameters
    ----------
    masks : dict
        A dictionary containing the masks to be reduced.
    reduce_res : int
        The factor by which the resolution of the masks should be reduced.
    shape : tuple
        The shape of the reduced masks.

    Returns
    -------
    dict
        A dictionary containing the reduced resolution masks.
    """
    chararray = np.chararray(shape)
    chararray[:] = ''
    counts = np.zeros(shape)

    for item in masks:
        mask = masks[item]

        counts_temp = block_reduce(mask, (reduce_res, reduce_res), func=np.sum)
        chararray = np.where(counts_temp > counts, item, chararray)
        counts = np.where(counts_temp > counts, counts_temp, counts)

    # initialize low resolution masks dictionary
    masks_lr = {}
    for item in masks:
        masks_lr[item] = chararray == item

    return masks_lr


# technically not correct because also upscaling can be done
def reduce_res_mean(reduce_res, *args):
    """
    Reduce the resolution of multiple arrays by taking the mean of each block of pixels.

    Parameters
    ----------
    reduce_res : int
        The factor by which the resolution of the arrays should be reduced.
    *args : np.ndarray
        The arrays to be reduced in resolution.

    Returns
    -------
    tuple
        A tuple of the reduced resolution arrays.
    """
    rst = []
    for ds in args:
        ds_lowres = block_reduce(ds, (reduce_res, reduce_res), np.mean)
        
        if ds.dtype == 'bool':
            ds_lowres = np.round(ds_lowres)
        rst.append(ds_lowres.astype(ds.dtype))
    return tuple(rst)


# %% RGB map functions

def arr_to_rgb(data, cmap=plt.cm.get_cmap('viridis'),
               mask=None, colorbar_lims=None):
    """
    Convert a 2D array to an RGB image using a colormap.
    
    Parameters
    ----------
    data : numpy ndarray
        The 2D array to be converted to an RGB image.
    cmap : matplotlib colormap, optional
        The colormap to use for the conversion. The default is `plt.cm.get_cmap('viridis')`.
    mask : numpy ndarray, optional
        A boolean mask of the same shape as `data` to be applied to the data before conversion.
    colorbar_lims : tuple, optional
        The limits of the colorbar. The default is None, which uses the minimum and maximum values of the data.
    
    Returns
    -------
    tuple
        A tuple of the RGB image and the colorbar used for the conversion.
    """

    mask = ~np.ma.masked_invalid(data).mask
    if mask.any():
        data = np.ma.masked_array(data, ~mask)
    if colorbar_lims is None:
        colorbar_lims = tuple((np.min(data), np.max(data)))

    norm = mpl.colors.Normalize(vmin=colorbar_lims[0], vmax=colorbar_lims[1])

    colorbar = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm)

    img = cmap(norm(data))[:, :, :3]

    return img, colorbar


def img_to_threshold_mask(image):
    """Set a threshold and mask pixels below the threshold value as False and the rest as True.
    """
    (_, threshold) = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    mask = threshold.astype(bool)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return mask, contours


def img_to_masks(
        img,
        map_colors,
        load_dir='mask.png',
        showfig=False,
        max_dist=10):
    """
    Convert an RGB image to a dictionary of 2D boolean masks, where each mask 
    corresponds to a specific surface type.
    
    Parameters
    ----------
    img : np.ndarray
        The RGB image to be converted to masks.
    map_colors : dict
        A dictionary of surface types and their corresponding RGB color codes.
    load_dir : str, optional
        The directory to save the updated mask image, if any corrections are 
        made. The default is 'mask.png'.
    showfig : bool, optional
        Whether to display the image and masks during the conversion process. 
        The default is False.
    max_dist : int, optional
        The maximum distance between an unknown color and a known color for 
        the unknown color to be considered a match. The default is 10.
    
    Returns
    -------
    masks : dict
        A dictionary of 2D boolean masks, where each mask corresponds to a 
        specific surface type.
    """

    # rgb colors to be used in the map
    map_colors_rgb = {k: (np.array(list(mpl.colors.to_rgb(v)))
                          * 255).astype(int) for k, v in map_colors.items()}

    if showfig:
        plt.imshow(img)
        plt.show(block=False)

    masks = {}

    for surf_type in map_colors:
        mask = (img == map_colors_rgb[surf_type]).all(2)
        masks.update({surf_type: mask})

    # check if the sum of all masks already covers the entire map surface
    fully_identified = (np.array(list(masks.values())).sum(0)).all()

    if not fully_identified:

        valid_colors = np.array(list(map_colors_rgb.values()))
        img_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

        unidentified_colors = img_colors[[
            not any(np.equal(valid_colors, color).all(1)) for color in img_colors]]

        # for all colors that are not yet identiefied, find the closest colors
        # if the color is not close enough to the list of valid color codes,
        # ask for user input to define the surface type
        for color in unidentified_colors:
            mask = (img == color).all(2)

            distance_colors = np.linalg.norm(valid_colors - color, axis=1)

            # if color is close to a known color, replace it by that color
            if any(distance_colors < max_dist):
                surf_type = list(map_colors_rgb.keys())[
                    distance_colors.argmin()]
                print(
                    f'Made a small correction to the color of the {surf_type}-mask.')
                plt.imshow(mask, cmap=mpl.colors.ListedColormap(
                    ['white', list(color / 255) + [1, ]]))
                plt.show(block=False)

            # ask for user input when color is not close to the list of defined
            # colors
            else:
                plt.imshow(mask, cmap=mpl.colors.ListedColormap(
                    ['white', list(color / 255) + [1, ]]))
                plt.show(block=False)
                surf_type = ''
                while True:
                    surf_type = input(
                        "Enter surface type of displayed surface: ")
                    if surf_type in list(map_colors_rgb.keys()):
                        break
                    else:
                        print('Invalid surfaces type entered, use one of: ' +
                              str(list(map_colors_rgb.keys())))

            masks[surf_type] += mask

        print('Updating mask .png file to fix off colors.')
        plot2d.masks2img(
            masks,
            map_colors,
            plot=True,
            save_dir=load_dir)

    return masks


def img_to_masks_sat(img, palette):
    """
    Convert an RGB image to a dictionary of 2D boolean masks using a color palette.

    Parameters
    ----------
    img : np.ndarray
        The RGB image to be converted to masks.
    palette : dict
        A dictionary of surface types and their corresponding RGB color codes.

    Returns
    -------
    masks : dict
        A dictionary of 2D boolean masks, where each mask corresponds to a specific surface type.
    """

    if not img.shape[0] == img.shape[1]:
        img = np.moveaxis(img, 0, -1)

    rgb_distance = np.stack([np.linalg.norm(img - rgb, axis=2)
                            for rgb in palette.values()], axis=2)

    arr_colors = np.array(list(palette.keys()))[
        tuple([np.argmin(rgb_distance, axis=2)])]

    masks = {}
    for item in palette.keys():
        masks.update({item: arr_colors == item})

    return masks


# %% 2D Numpy array functions


# def get_lims(loc_mid, gridsize):
#     """
#     Get limits of bounding box around a mid-point for a given gridsize."""
#     xlims = np.array(
#         [loc_mid[0] - gridsize / 2, loc_mid[0] + gridsize / 2]).astype(int)
#     ylims = np.array(
#         [loc_mid[1] - gridsize / 2, loc_mid[1] + gridsize / 2]).astype(int)

#     return xlims, ylims


# def get_datalims(loc_mid, gridsize):
#     """?"""
#     data_xlims = np.array(
#         [loc_mid[0] * 2 - gridsize, loc_mid[0] * 2 + gridsize]).astype(int)
#     data_ylims = np.array(
#         [loc_mid[1] * 2 - gridsize, loc_mid[1] * 2 + gridsize]).astype(int)

#     return data_xlims, data_ylims

# def preprocess_ds(ds_raw,xlims,ylims,water_level = 0, save_2Dmap = False, save_3Dmap = False):
#     # find values where no data is recorded
#     nodata_value = np.max(ds_raw)

#     # set nodata value (i.e. waterbodies) to water level
#     ds = np.where(ds_raw == nodata_value, water_level, ds_raw)
#     ds_crop = ds[ylims[0]:ylims[1], xlims[0]:xlims[1]]

#     return ds, ds_crop


def crop_center(arr, cropx, cropy):
    """
    Crop 2D array to desided size.
    
    Parameters
    ----------
    arr : np.ndarray
        The array to be cropped.
    cropx : int
        The width of the cropped array.
    cropy : int
        The height of the cropped array.

    Returns
    -------
    numpy ndarray
        The cropped array.
    """
    cropx = int(cropx)
    cropy = int(cropy)
    x, y = arr.shape[-2:]
    startx = (x // 2) - (cropx // 2)
    starty = (y // 2) - (cropy // 2)
    if len(arr.shape) == 2:
        return arr[starty:starty + cropy, startx:startx + cropx]
    if len(arr.shape) == 3:
        return arr[:, starty:starty + cropy, startx:startx + cropx]


def rotate_crop(rot_angle, gridsize, res, *args):
    """
    Rotate and crop input 2D or 3D arrays or dictionary of 2D arrays.

    Parameters
    ----------
    rot_angle : float
        The angle by which the input arrays will be rotated, in degrees.
    gridsize : float
        The size of the grid, in pixels.
    res : float
        The resolution of the input arrays, in pixels per meter.
    *args : np.ndarray or dict
        The input arrays or dictionary of arrays to be rotated and cropped.

    Returns
    -------
    tuple
        Tuple of rotated and cropped input arrays or dictionary of rotated and cropped arrays."""
    rst = []
    for data in args:

        if isinstance(data, np.ndarray):

            data_rot = scipy.ndimage.rotate(data.astype(
                float), rot_angle, reshape=False)
            data_crop = crop_center(data_rot, gridsize / res, gridsize / res)

            rst.append(np.round(data_crop).astype(data_crop.dtype))

        if isinstance(data, dict):
            dct = {}
            for item in data:
                data_rot = scipy.ndimage.rotate(data[item].astype(
                    float), rot_angle, reshape=False)
                data_crop = crop_center(
                    data_rot, gridsize / res, gridsize / res)
                dct.update({str(item): np.round(
                    data_crop).astype(data_crop.dtype)})
            rst.append(dct)

    return tuple(rst)


# %%

def masks_to_idcs_dict(idcs_3D, types_dict, faces_idcs):  # identify_els
    """
    Make a dict containing per environmental type all indices of the elements 
    from that type.
    
    Parameters
    ----------
    idcs_3D : numpy array
        3D array of element indices.
    types_dict : dict
        Dictionary of 2D masks, where the keys are the environmental type and 
        the values are the corresponding 2D boolean masks.
    faces_idcs : list
        List of indices of elements that are faces of the 3D grid.

    Returns
    -------
    dict
        A dictionary that per enviromental type contains a list of element indices.
    """
    idcs_dict = {}
    for datatype in types_dict:
        mask = types_dict[datatype]
        mask_coords = np.argwhere(np.transpose(mask))
        idcs = coo_to_idcs(coo=mask_coords, idcs_3D=idcs_3D)
        if datatype == "buildings":
            # highest elements of building are roofs
            idcs_roofs = np.array(
                list(set(idcs).intersection(faces_idcs[3])), dtype=int)
            # building elements that are not roofs thus are walls
            idcs_walls = np.array(
                list(set(idcs).difference(idcs_roofs)), dtype=int)

            idcs_dict.update({'roofs': idcs_roofs})
            idcs_dict.update({'walls': idcs_walls})

        idcs_dict.update({datatype: idcs})

    return idcs_dict


# find_street_building
def identify_street_and_building(coo, street_height, faces_list, dz):
    """
    Identify the elements that belong to the street and the buildings from a 
    3D coordinate array.
    
    Parameters
    ----------
    coo : ndarray
        2D array of 3D coordinates of the elements.
    street_height : float
        Height of the street level
    faces_list : ndarray
        List of faces corresponding to the elements in
    dz : float
        The size of the grid cell along the z-axis
        
    Returns
    -------
    els_buildings : ndarray
        Indices of elements that belong to buildings.
    els_street : ndarray
        Indices of elements that belong to the street.
    """
    els_buildings = np.where(np.greater(coo[:, 2], street_height + dz / 2))[0]
    els_street = np.where(np.logical_and(
        coo[:, 2] == street_height + dz / 2, faces_list[:, 2] == 1))[0]

    return els_buildings, els_street


def identify_els_outer_faces(coo, street_height, dx=1, dy=1, dz=1):
    """

    Identify the outer faces of elements in a 3D array from a list of coordinates (COO) and also return a list that per direction (east, west, bottom, etc.) contains a list of booleans to indicate that an element has an outer face in that direction
    
    Face 0: West
    Face 1: East
    Face 2: Bottom
    Face 3: Top
    Face 4: South
    Face 5: North
        
    Parameters
    ----------
    coo : numpy array, shape (N,3)
        Coordinates of the elements in 3D space.
    street_height : float
        Height of the street level in the 3D space.
    dx : float
        Size of elements in x-axis.
    dy : float
        Size of elements in y-axis.
    dz : float
        Size of elements in z-axis.
        
    Returns
    -------
    faces_list : np.array, shape (N,6)
        A numpy boolean array for all N elements with 6 columns (west, east, 
        bottom, top, south, north). Values are True where elements have outer 
        faces in that face direction.
    outer_els_bools : numpy array
        Boolean array indicating which elements have at least one outer face 
        and thus are the outer elements.
    """

    faces_list = np.ones((len(coo), 6)).astype(int).astype(bool)
    faces_list[:, 2] = False

    for i_face in [0, 1, 2, 3, 4, 5]:
        if i_face == 0:
            coo_compare = np.add(coo, [-dx, 0, 0])
        if i_face == 1:
            coo_compare = np.add(coo, [+dx, 0, 0])
        if i_face == 2:
            coo_compare = np.add(coo, [0, 0, -dz])
        if i_face == 3:
            coo_compare = np.add(coo, [0, 0, +dz])
        if i_face == 4:
            coo_compare = np.add(coo, [0, -dy, 0])
        if i_face == 5:
            coo_compare = np.add(coo, [0, +dy, 0])

        els_at_side = intersecting_rows(coo_compare, coo)[1]
        faces_list[els_at_side, i_face] = False

    outer_els_bools = faces_list.any(1)

    # side 1
    faces_list[np.where(coo[:, 0] == coo[:, 0].min()), 0] = False
    # side 2
    faces_list[np.where(coo[:, 0] == coo[:, 0].max()), 1] = False

    # side 5
    faces_list[np.where(coo[:, 1] == coo[:, 1].min()), 4] = False
    # side 6
    faces_list[np.where(coo[:, 1] == coo[:, 1].max()), 5] = False

    return faces_list, outer_els_bools


def calc_faces_idcs(coo, faces_list):
    """
    Generate a numpy array containing per orientation the indices of elements 
    having a face in that orientation.
    
    Parameters
    ----------
    coo : numpy array, shape (N,3)
        Coordinates of the elements in 3D space.
    
    faces_list : np.array, shape (N,6)
        A numpy boolean array for all N elements with 6 columns (west, east, 
        bottom, top, south, north). Values are True where elements have outer 
        faces in that face direction.

    Returns
    -------
    faces_idcs : nested np.ndarray of np.ndarrays of ints
        Numpy array with 6 numpy array objects containing indices of the 
        elements with faces in that direction.
        
        Face 0: West
        Face 1: East
        Face 2: Bottom
        Face 3: Top
        Face 4: South
        Face 5: North
        """

    faces_idcs = np.array(
        [np.where(faces_list[:, 0] == True)[0],
         np.where(faces_list[:, 1] == True)[0],
         np.where(faces_list[:, 2] == True)[0],
         np.where(faces_list[:, 3] == True)[0],
         np.where(faces_list[:, 4] == True)[0],
         np.where(faces_list[:, 5] == True)[0]],
        dtype=object)

    return faces_idcs


# def remove_inner_elements(faces_list, coo, els_buildings,
#                           els_street, street_height, dz):
#     """Delete elements from a list that are inside an object body.
    
    
#     Parameters
#     ----------


#     Returns
#     -------"""
#     inner_elements = np.where(~faces_list[:, 1:].any(axis=1))[0]
#     faces_list = np.delete(faces_list, inner_elements, axis=0)
#     coo = np.delete(coo, inner_elements, axis=0)
#     before = len(els_buildings)

#     els_buildings, els_street = identify_street_and_building(
#         coo, street_height, faces_list, dz)
#     print(
#         'removed',
#         before -
#         len(els_buildings),
#         'elements from els_buildings')

#     return faces_list, coo, els_buildings, els_street


# def exposed_faces_sun(azimuth):
#     if 0 <= azimuth < math.pi / 2:
#         return np.array([2, 3, 4, 6])

#     if math.pi / 2 <= azimuth < math.pi:
#         return np.array([2, 3, 4, 5])

#     if math.pi <= azimuth < math.pi * 3 / 2:
#         return np.array([1, 3, 4, 5])

#     if math.pi * 3 / 2 <= azimuth < 2 * math.pi:
#         return np.array([1, 3, 4, 6])


# %% Fill missing
