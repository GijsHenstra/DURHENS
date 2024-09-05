#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:48:29 2020

@author: x
"""

import numpy as np
import math
import itertools
import time

import numba
import multiprocessing as mp
import tqdm

from helpers import help_functions as hf

# import line_profiler
# profile = line_profiler.LineProfiler()

# %% View factor matrix calculation function

# @profile
def calc_matrix_mp(
        envir_from,
        envir_to,
        min_vf_value,
        res=1,
        skip_bottom=False,
        max_dist=None,
        multiprocessing='starmap',
        processes=mp.cpu_count(),
        chunksize=None):
    """
    Calculates the view factor matrix using multiprocessing.

    Parameters
    ----------
    envir_from : dict
        Dictionary containing the coordinates, indices of the elements and 
        indices of the faces of the emitting environment.
    envir_to : dict
        Dictionary containing the coordinates, indices of the elements and 
        indices of the faces of the receiving environment.
    res : int, optional
        The resolution (length of the side of a square) of the elements.
    skip_bottom : bool, optional
        Skip bottom face for the emitting environment. Default is False.
    max_dist : int, optional
        Maximum distance of a ray. If None, do not cut off rays. Default is None.
    multiprocessing : str, optional
        Method of multiprocessing to use, e.g. 'starmap' or 'map'.
    processes : int, optional
        Number of processes to use, defaults to the number of CPU cores.
    chunksize : int, optional
        Chunk size for 'map' method.

    Returns
    -------
    coo_triu_arr : np.ndarray
        A 2D upper triangular sparse array of shape (N, 4), where N is the
        number of all faces of the elements. Each row consists of i1, i2, f1,
        and f2. Where i is the index of the element and f is the face of that
        element.
    vf_arr : np.ndarray
        A 1D array of shape (N,) containing the view-factors for each element 
        in `coo_arr`.

    """
    # if shapes match, the reciprocical properties of the view factors can be
    # used.
    reciprocity = (envir_from['coo'].shape == envir_to['coo'].shape)

    coo_from_int = envir_from['coo'].astype(int)
    coo_to_int = envir_to['coo'].astype(int)

    longest_dist = int(max(max(np.linalg.norm(envir_from['coo'], axis=1)),
                           max(np.linalg.norm(envir_to['coo'], axis=1))))

    if max_dist is not None:
        ray_length = int(min([longest_dist, max_dist]))
    else:
        ray_length = longest_dist

    # skip face 3 (bottom face), it is always adjacent to an onther element at
    # that face.
    if skip_bottom:
        faces = np.array([0, 1, 3, 4, 5])
    if not skip_bottom:
        faces = np.array([0, 1, 2, 3, 4, 5])

    # expand coordinate list
    coo_from_exp = np.array([coo_from_int[:, 0],
                             coo_from_int[:, 0] + res,
                             coo_from_int[:, 1],
                             coo_from_int[:, 1] + res,
                             coo_from_int[:, 2],
                             coo_from_int[:, 2] + res], dtype='i').T

    coo_to_exp = np.array(
        [coo_to_int[:, 0],
         coo_to_int[:, 0] + res,
         coo_to_int[:, 1],
         coo_to_int[:, 1] + res,
         coo_to_int[:, 2],
         coo_to_int[:, 2] + res], dtype='i').T

    A = res**2

    args_lst = []
    start = time.time()

    dtype = np.uint16 if len(
        envir_from['coo']) < np.iinfo(
        np.uint16).max else np.uint32

    # prepare all arguments to put in the pool
    for face_from in faces:
        faces_idcs_opp = [i for i in faces if i != face_from]

        for face_to in faces_idcs_opp:
            if (face_to > face_from if reciprocity else True):
                els_from = envir_from['faces_idcs'][face_from]
                
                args_lst += ((envir_from['coo'],
                              envir_to['coo'],
                              face_from,
                              face_to,
                              el_from,
                              envir_to['faces_idcs'][face_to],
                              envir_to['idcs_3D'],
                              ray_length,
                              coo_from_exp,
                              coo_to_exp,
                              A,
                              min_vf_value,
                              dtype) for el_from in els_from)

    start = time.time()

    # several different options to perform view factor calculation in parallel,
    # using jitted or non-jitted functions.
    if multiprocessing == 'map':
        with mp.Pool(processes=processes) as pool:
            vf_coo_triu_lst = list(
                pool.imap(
                    _vf_el_to_face_args,
                    tqdm.tqdm(
                        args_lst,
                        total=len(args_lst))))

    elif multiprocessing == 'map_unordered':
        with mp.Pool(processes=processes) as pool:
            vf_coo_triu_lst = list(
                tqdm.tqdm(
                    pool.imap_unordered(
                        _vf_el_to_face_args,
                        args_lst),
                    total=len(args_lst),
                ))

    elif multiprocessing == 'starmap':
        with mp.Pool(processes=processes) as pool:
            coo_triu_lst, vf_lst = zip(
                *pool.starmap(
                    func=vf_el_to_face, iterable=tqdm.tqdm(
                        args_lst, total=len(args_lst)), chunksize=chunksize))

    elif multiprocessing == 'starmap3':
        with mp.Pool(processes=processes) as pool:
            vf_coo_triu_lst = pool.starmap(func=vf_el_to_face, iterable=args_lst)

    elif multiprocessing == 'starmap_async':
        with mp.Pool(processes=processes) as pool:
            vf_coo_triu_lst = pool.starmap_async(
                func=vf_el_to_face, iterable=tqdm.tqdm(
                    args_lst, total=len(args_lst))).get()

    elif multiprocessing == 'apply':
        with mp.Pool(processes=processes) as pool:
            for args in args_lst:
                vf_coo_triu_lst = pool.apply(
                    func=vf_el_to_face, args=tqdm.tqdm(
                        args, total=len(args_lst)))

    elif multiprocessing == 'apply_async':
        with mp.Pool(processes=processes) as pool:
            vf_coo_triu_lst = pool.apply_async(
                vf_el_to_face, args=tqdm.tqdm(
                    args_lst, total=len(args_lst))).get()

    # using non-jitted functionsperform serialized computations on the view
    # factors between elements and gather in a list
    elif multiprocessing == 'serial':
        coo_triu_lst = []
        vf_lst = []
        for args in tqdm.tqdm(args_lst):
            coo_triu, vf = vf_el_to_face(*args)
            
            coo_triu_lst.append(coo_triu)
            vf_lst.append(vf)
            

    # using jitted functions, perform serialized computations of the view
    # factors between elements and gather in a list
    elif multiprocessing == 'serial_jit':
        vf_coo_triu_lst = vf_el_to_face_par(args_lst)

    else:
        raise NotImplementedError()

    print(f'Calculating the view factors took {time.time()-start} s')

    start = time.time()
    print('Start gathering the VF elements in one upper triangular COO-array')

    # remove empty items from list
    coo_triu_lst = [l for l in coo_triu_lst if l is not None]
    vf_lst = [l for l in vf_lst if l is not None]

    if not coo_triu_lst:
        return  np.array([]), np.array([])

    coo_triu_arr = np.concatenate(coo_triu_lst)
    vf_arr = np.concatenate(vf_lst)
    print(f'Concatenating the list for return took {time.time()-start} s')

    return coo_triu_arr, vf_arr

# %% Wrappers for vf calculation


def _vf_el_to_face_args(args):
    """Wrapper for parsing argument to function `vf_el_to_face`."""

    (coo_from, coo_to, face_from, face_to, el_from, els_to_all,
     idcs_3D, max_dist, coo_from_exp, coo_to_exp, A, vf_el_to_face) = args

    return vf_el_to_face(
        coo_from,
        coo_to,
        face_from,
        face_to,
        el_from,
        els_to_all,
        idcs_3D,
        max_dist,
        coo_from_exp,
        coo_to_exp,
        A,
        vf_el_to_face)


@numba.njit
def vf_el_to_face_par(args_lst):

    vf_coo_triu_lst = []
    for i in range(len(args_lst)):
        args = args_lst[i]
        vf_coo_triu_lst.append(vf_el_to_face(*args))

    return vf_coo_triu_lst

#%% VF calculation

#!!! Could be named different, confusing with `calc_vf_el2face`
@numba.njit 
def vf_el_to_face(
        coo_from,
        coo_to,
        face_from,
        face_to,
        el_from,
        els_to_all,
        idcs_3D,
        ray_length,
        coo_from_exp,
        coo_to_exp,
        A,
        min_vf_value,
        dtype):
    """Filter the visible faces and then compute vf's between one element and 
    all surrounding elements using the function `calc_vf_el_to_els`.

    Parameters
    ----------
    coo_from : np.ndarray, shape (N, 3)
        An array with the coordinates of all the elements from which one to
        get the coordinates from element `el_from`.
    coo_to : np.ndarray, shape (M, 3)
        An array with the coordinates of all the elements. Check which of
        these elements are visible with `calc_visible_els_seen_from_an_el`.
    face_from : int
        Index of the face of the element that is being used as the origin of
        radiation.
    face_to : int
        Index of the face of the elements that is being used as the target of 
        radiation.
    el_from : int
        Index of the element that is being used as the source of radiation.
    els_to_all : np.ndarray
        Indices of the elements that are being used as targets to
        calculate the view factors to. Must first be filtered to only check
        which of them are visible from element `el_from`.
    idcs_3D : np.ndarray
        A 3D array containing the indices corresponding to the given coordinates.
    ray_length : int
        Maximum distance for rays to be traced.
    coo_from_exp : np.ndarray, shape (N, 6)
        Expanded coordinates of all the elements in the environment.
    coo_to_exp : np.ndarray
        Expanded coordinates of all the elements in the environment.
    A : float
        Area of the source element's face.
    dtype : numpy data type
        Data type used to store the output.

    Returns
    -------
    row_els : np.ndarray
        Sparse COO array with information about the source and target elements 
        and their faces.
    view_factors : np.ndarray
        Array with the computed view factors between the source element and all 
        the target elements.
    """

    els_to_filtered = calc_visible_els_seen_from_an_el(
        coo_from=coo_from,
        coo_to=coo_to,
        face_from=face_from,
        face_to=face_to,
        el_from=el_from,
        els_to_all=els_to_all,
        idcs_3D=idcs_3D,
        ray_length=ray_length)

    # if el_from == 1349:
    #     print('els_to_filtered', els_to_filtered)

    if len(els_to_filtered) == 0:
        # raise ValueError('No elements remain when filtering.')
        return None, None

    nr_els_to = len(els_to_filtered)

    coords_from = transform_coords_els_to_faces(
        coords_els=coo_from_exp[el_from, :],
        face=face_from
    )

    coords_to = transform_coords_els_to_faces(
        coords_els=coo_to_exp[np.array(els_to_filtered), :],
        face=face_to)

    # find out if the axes are parallel or perpendicular and
    # what their common axes are
    bool_parallel, common_axis = find_common_axis(coords_from, coords_to[0])

    view_factors = calc_vf_el_to_els(
        coords_from=coords_from,
        coords_to=coords_to,
        bool_parallel=bool_parallel,
        common_axis=common_axis,
        A=A)
    
    keep_list = view_factors > min_vf_value

    coo = np.array([[el_from] * nr_els_to,
                    els_to_filtered,
                    [face_from] * nr_els_to,
                    [face_to] * nr_els_to]).T.astype(dtype)
    
    coo = coo[keep_list]
    view_factors = view_factors[keep_list]
    
    # if el_from == 1349:
    #     print('view_factors', view_factors)
    #     print('coo', coo)

    return coo, view_factors


@numba.njit
def calc_vf_el_to_els(coords_from, coords_to, bool_parallel, common_axis, A):
    """Calculates the view factors for all elements in `coords_to` that are 
    visible from a single element represented by coords_from. The function uses
    a jit-compiled version of the view factor calculation, for speed optimisation.

    Parameters:
    -----------
    coords_from : array_like
        Coordinates of the origin element in the form [x, x, y, y, z, z]
    coords_to : array_like
        A 2D array of coordinates of the target elements in the form 
        [[x, x, y, y, z, z], [x, x, y, y, z, z], ...]
    bool_parallel : boolean
        A boolean indicating whether the common axis is parallel. If False, the
        faces are perpendicular.
    common_axis : int
        An int indicating which axis is the shared axis either parallel or 
        perpendicular axis.
    A : float
        The area of one face of the element

    Returns:
    --------
    view_factors : np.ndarray
        A 1D array of view factors for all visible elements in coords_to
    """
    B_summation = np.zeros(len(coords_to))

    for l in numba.prange(2):
        for k in numba.prange(2):
            for j in numba.prange(2):
                for i in numba.prange(2):
                    if bool_parallel:
                        if common_axis == 0:
                            z = np.abs(coords_from[0] - coords_to[:, 0])
                            B = calc_B_parallel(
                                coords_from[i + 2], coords_from[j + 4], coords_to[:, k + 2], coords_to[:, l + 4], z)
                        if common_axis == 1:
                            z = np.abs(coords_from[2] - coords_to[:, 2])
                            B = calc_B_parallel(
                                coords_from[i], coords_from[j + 4], coords_to[:, k], coords_to[:, l + 4], z)
                        if common_axis == 2:
                            z = np.abs(coords_from[4] - coords_to[:, 4])
                            B = calc_B_parallel(
                                coords_from[i], coords_from[j + 2], coords_to[:, k], coords_to[:, l + 2], z)
                    elif not bool_parallel:
                        if common_axis == 0:
                            B = calc_B_perpendicular(np.abs(
                                coords_from[i + 4] - coords_to[:, i + 4]), coords_from[j], coords_to[:, l], np.abs(coords_to[:, k + 2] - coords_from[k + 2]))
                        if common_axis == 1:
                            B = calc_B_perpendicular(np.abs(
                                coords_from[i + 4] - coords_to[:, i + 4]), coords_from[j + 2], coords_to[:, l + 2], np.abs(coords_to[:, k] - coords_from[k]))
                        if common_axis == 2:
                            B = calc_B_perpendicular(np.abs(
                                coords_from[i] - coords_to[:, i]), coords_from[j + 4], coords_to[:, l + 4], np.abs(coords_to[:, k + 2] - coords_from[k + 2]))

                    B_summation += (-1)**(i + j + k + l) * B

    # !!! explain abs
    view_factors = (1 / (2 * np.pi * A)) * np.abs(B_summation)

    return view_factors


@numba.njit(fastmath=True)
def calc_B_perpendicular(x, y, eta, ksi):
    """
    Calculate B-factor for the view factor calculation of perpendicular faces.
    
    Note that inputs `x` and `y` not necessarily have to be the the coordinates
    of the x-axis and y-axis respectively. They are just the coordinates of the
    corner points of the faces that are perpendicular to each other. If the 
    shared axis for instance is the y-axis, `x` represents the x-axis and `y`
    represents the z-axis.
    
    This function is jitted using numba's njit decorator and fastmath mode enabled.

    Parameters
    ----------
    x : float
        coordinate on 1st axis of the face to calculate the view-factors from.
    y : float
        coordinate on 2nd axis  f the face to calculate the view-factors from.
    eta : array_like
        coordinates on 1st axis of the faces to calculate the view-factors to.
    ksi : array_like
        coordinates on 2nd axis of the faces to calculate the view-factors to.

    Returns
    -------
    B : np.ndarray
        B-factors for the view factor calculation or perpendicular faces.
        
    Reference
    ---------
    http://webserver.dmt.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf    
    
    """

    eps = 1e-16
    x = np.add(x, eps)

    C_squared = np.power(x, 2) + np.power(ksi, 2)
    C = np.sqrt(C_squared)
    D = (y - eta) / C
    D_squared = np.power(D, 2)

    B = (y - eta) * C * np.arctan(D) - (C_squared / 4) * \
        (1 - D_squared) * np.log(C_squared * (1 + D_squared))

    return B


@numba.njit(fastmath=True)
def calc_B_parallel(x, y, ksi, eta, z):
    """Calculate B-factor for the view factor calculation of parallel faces.

    Note that inputs `x` and `y` not necessarily have to be the the coordinates
    of the x-axis and y-axis respectively. They are just the coordinates of the
    corner points of the faces that are parallel to each other. If the shared 
    axis for instance is the y-axis, `x` represents the x-axis and `y`
    represents the z-axis.

    This function is jitted using numba's njit decorator and fastmath mode enabled.

    Parameters
    ----------
    x : float
        Coordinate on 1st axis of the face to calculate the view-factors from.
    y : float
        Coordinate on 2nd axis  o the face to calculate the view-factors from.
    eta : array_like
        Coordinates on 1st axis of the faces to calculate the view-factors to.
    ksi : array_like
        Coordinates on 2nd axis of the faces to calculate the view-factors to.
    z : array_like
        Distance between the origin face and the target faces on the 3rd axis,
        perpendicular to the faces.

    Returns
    -------
    B : np.ndarray
        B-factors for the view factor calculation of parallel faces.
        
    Reference
    ---------
    http://webserver.dmt.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf
    """

    eps = 1e-16

    x = np.where(x == ksi, x + eps, x)
    y = np.where(y == eta, y + eps, y)

    u = np.abs(x - ksi)
    u_sq = np.power(u, 2)
    v = np.abs(y - eta)

    z_sq = np.power(z, 2)
    v_sq = np.power(v, 2)
    p = np.sqrt(u_sq + z_sq)
    q = np.sqrt(v_sq + z_sq)

    B = (v * p * np.arctan(v / p) + u * q *
         np.arctan(u / q) - z_sq / 2 * np.log(u_sq + v_sq + z_sq))

    # remove the entries where distance z is 0
    B = np.where(z == 0, 0, B)

    return B


# %% Faces and elements visibility filtering

@numba.njit
def calc_visible_els_seen_from_an_el(
        coo_from,
        coo_to,
        face_from,
        face_to,
        idcs_3D,
        el_from,
        els_to_all,
        ray_length,
        dx=1,
        dy=1,
        dz=1):
    """Determine which of the elements of `els_to_all` are visible to `el_from`.

    By tracing rays from the coordinates of `el_from` and `els_to_all`, find if
    the coordinates of the rays do not overlap with coordinates of other
    elements. If the coordinates of a ray between element i1 and element i2 do
    not overlap with the coordinates of the environment, that means that i2 is
    visible. Function is jitted to improve computation speed.

    Parameters
    ----------
    coo_from : np.ndarray, shape (N, 3)
        An array with the coordinates of all the elements from which one to
        get the coordinates from element `el_from`.
    coo_to : np.ndarray, shape (M, 3)
        An array with the coordinates of all the elements. Check which of
        these elements are visible with `calc_visible_els_seen_from_an_el`.
    face_from : int
        Index of the face of the element that is being used as the origin.
    face_to : int
        Index of the face of the elements that is being used as the target.
    idcs_3D : np.ndarray
        A 3D array containing the indices corresponding to the given coordinates.
    el_from : int
        Index of the element that is being used as the source of radiation.
    els_to_all : np.ndarray
        Indices of the elements that are being used as targets to
        calculate the viewfactors to. Must first be filtered to only check
        which of them are visible from element `el_from`.
    ray_length : int
        Maximum distance for rays to be traced.
    dx : int, optional
        The grid spacing in the x-direction, by default 1
    dy : int, optional
        The grid spacing in the y-direction, by default 1
    dz : int, optional
        The grid spacing in the z-direction, by default 1

    Returns
    -------
    visible_els : list of ints
        A list of integers representing the indices of the visible elements as
        seen from element `el_from`.
    """

    
    # filter the faces away that the face of `el_from`, so that no backwards
    # rays will be traced that will anyway not be visible.
    
    els_to = filter_faces_behind(
        coo_from=coo_from,
        coo_to=coo_to,
        face_from=face_from,
        face_to=face_to,
        el_from=el_from,
        els_to_all=els_to_all,
        max_dist=ray_length)
    
    if els_to.size == 0:
        return list(els_to)

    coords_from_side = hf.coo_mid_to_side_jit(
        coords_mid=coo_from[el_from], face=face_from, dx=dx, dy=dy, dz=dz)
    coords_to_side = hf.coo_mid_to_side_jit(
        coords_mid=coo_to[els_to], face=face_to, dx=dx, dy=dy, dz=dz)

    visible_els = []

    for i in numba.prange(len(coords_to_side)):
        start = coords_from_side
        stop = coords_to_side

        ray = np.zeros((ray_length, 3))

        # loop over x, y, z
        for j in range(3):

            lower = start[j]
            upper = stop[i, j]

            for k in range(ray_length):
                val = lower + (k * (upper - lower) / ray_length)
                ray[int(k), int(j)] = val

        # do not include the startpoint itself in the ray
        ray = ray[1:, :]
        ray_empty = np.empty_like(ray) 

        ray_min_half = np.subtract(ray, 0.5)
        ray_rounded = np.round(ray_min_half, 0, ray_empty)
        ray_int = ray_rounded.astype(np.int64)

        idcs_at_ray_coords = hf.coo_to_idcs_jit(coo=ray_int,
                                                idcs_3D=idcs_3D,
                                                remove_negs=False)

        el_visible = True

        # per ray, check if the coordinates of the overlap with coordinates of the
        # environmtent.
        for idx in idcs_at_ray_coords:
            if idx >= 0:
                el_visible = False

        if el_visible:
            visible_els.append(els_to[i])

    visible_els = list(visible_els)

    return visible_els


@numba.njit
def filter_faces_behind(
        coo_from,
        coo_to,
        face_from,
        face_to,
        el_from,
        els_to_all,
        max_dist=None):
    """This function filters the elements in `els_to_all` that are located
    behind a given element `el_from`, based on the coordinates `coo_from` and
    `coo_to`, and the faces `face_from` and `face_to`.

    Parameters
    ----------
    coo_from : numpy.ndarray
        The coordinates of the element from which visibility is calculated.
    coo_to : numpy.ndarray
        The coordinates of the elements for which visibility is checked.
    face_from : int
        The face of the element from which visibility is calculated.
    face_to : int
        The face of the elements for which visibility is checked.
    el_from : int
        The element from which visibility is calculated.
    els_to_all : numpy.ndarray
        The elements for which visibility is checked.
    max_dist : float, optional
        Maximum distance of visibility, by default None

    Returns
    -------
    els_result : list of ints
        A list of integers representing the indices of the elements that are
        not behind the element `el_from`

    """
    if face_from == 0:
        els_targ_in_front = els_to_all[np.where(
            np.less(coo_to[els_to_all, 0], coo_from[el_from, 0]))[0]]
    if face_from == 1:
        els_targ_in_front = els_to_all[np.where(np.greater(
            coo_to[els_to_all, 0], coo_from[el_from, 0]))[0]]
    if face_from == 2:
        els_targ_in_front = els_to_all[np.where(
            np.less(coo_to[els_to_all, 2], coo_from[el_from, 2]))[0]]
    if face_from == 3:
        els_targ_in_front = els_to_all[np.where(np.greater(
            coo_to[els_to_all, 2], coo_from[el_from, 2]))[0]]
    if face_from == 4:
        els_targ_in_front = els_to_all[np.where(
            np.less(coo_to[els_to_all, 1], coo_from[el_from, 1]))[0]]
    if face_from == 5:
        els_targ_in_front = els_to_all[np.where(np.greater(
            coo_to[els_to_all, 1], coo_from[el_from, 1]))[0]]
        
    # if el_from == 1349:
    #     print('els_targ_in_front', els_targ_in_front)

    if face_to == 0:
        els_targ_in_front2 = els_targ_in_front[np.where(np.greater(
            coo_to[els_targ_in_front, 0], coo_from[el_from, 0]))[0]]
    if face_to == 1:
        els_targ_in_front2 = els_targ_in_front[np.where(
            np.less(coo_to[els_targ_in_front, 0], coo_from[el_from, 0]))[0]]
    if face_to == 2:
        els_targ_in_front2 = els_targ_in_front[np.where(np.greater(
            coo_to[els_targ_in_front, 2], coo_from[el_from, 2]))[0]]
    if face_to == 3:
        els_targ_in_front2 = els_targ_in_front[np.where(
            np.less(coo_to[els_targ_in_front, 2], coo_from[el_from, 2]))[0]]
    if face_to == 4:
        els_targ_in_front2 = els_targ_in_front[np.where(np.greater(
            coo_to[els_targ_in_front, 1], coo_from[el_from, 1]))[0]]
    if face_to == 5:
        els_targ_in_front2 = els_targ_in_front[np.where(
            np.less(coo_to[els_targ_in_front, 1], coo_from[el_from, 1]))[0]]
        
    # if el_from == 1349:
    #     print('els_targ_in_front2', els_targ_in_front2)
    

    # max length of the ray, items further apart are not conscidered in the
    # view factor calculation
    if max_dist is not None:
        els_targ_in_front_and_closeby = []
        for el_to in els_targ_in_front2:
            if np.linalg.norm(coo_from[el_from] - coo_to[el_to]) < max_dist:
                els_targ_in_front_and_closeby.append(el_to)
        els_result = np.array(els_targ_in_front_and_closeby)

    else:
        els_result = els_targ_in_front
        
    # if el_from == 1349:
    #     print('els_result', els_result)

    return els_result


# %% Supportive functions

@numba.njit
def transform_coords_els_to_faces(coords_els, face, dx=1, dy=1, dz=1):
    """Transform coordinates that cover the corner points of en entire block
    element to coordinates that cover the corner points of a specified face.

    The coordinates are in the form of [[x, x, y, y, z, z], [x, x, y, y, z, z], ...]. 
    So for example, a cubic element with coordinates [0, 1, 0, 1, 0, 1], 
    transformed to a west-oriented face (negative-x-axis-direction), 
    becomes [0, 0, 0, 1, 0, 1].
    """

    if face == 0:
        coords_faces = coords_els - np.array([0, dx, 0, 0, 0, 0])
    if face == 1:
        coords_faces = coords_els + np.array([dx, 0, 0, 0, 0, 0])
    if face == 2:
        coords_faces = coords_els - np.array([0, 0, 0, 0, 0, dz])
    if face == 3:
        coords_faces = coords_els + np.array([0, 0, 0, 0, dz, 0])
    if face == 4:
        coords_faces = coords_els - np.array([0, 0, 0, dy, 0, 0])
    if face == 5:
        coords_faces = coords_els + np.array([0, 0, dy, 0, 0, 0])

    return coords_faces




@numba.njit
def find_common_axis(coords_from, coords_to):
    """
    Find the common axis between the origin face and target face's coordinates.
    Also return a boolean indicating if the two elements are parallel or not.

    Parameters
    ----------
    coords_from : np.ndarray
        The coordinates of the origin face, in the form [x, x, y, y, z, z].
    coords_to : np.ndarray
        The coordinates of the target element, in the form [x, x, y, y, z, z].

    Returns
    -------
    bool_parallel : bool
        Boolean indicating if the faces are parallel.
    common_axis : int
        Integer number indicating the found common axis. Can be the axis over
        which the faces are parallel or the axis over which they are
        parallel.
    """

    orig_el_equal = [
        (coords_from[0] == coords_from[1]),
        (coords_from[2] == coords_from[3]),
        (coords_from[4] == coords_from[5])]

    targ_el_equal = [
        (coords_to[0] == coords_to[1]), 
        (coords_to[2] == coords_to[3]), 
        (coords_to[4] == coords_to[5])]

    same_axis_bools = [orig_el_equal[i] == targ_el_equal[i]
                       for i in range(len(orig_el_equal))]

    if False not in same_axis_bools:
        bool_parallel = True
        common_axis = int([i for i, b in enumerate(orig_el_equal) if b][0])
    else:
        bool_parallel = False
        common_axis = int([i for i, b in enumerate(same_axis_bools) if b][0])

    return bool_parallel, common_axis
