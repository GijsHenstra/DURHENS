 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:39:19 2020

@author: gijshenstra
"""

import math
import numpy as np
import numba

from helpers import help_functions as hf

# import line_profiler
# profile = line_profiler.LineProfiler()

def include_shadowside(_exposed_els, _dark_els):
    """Exclude elements from the list of `_dark_els` if they also occur in 
    the list of `_exposed_els`."""
    
    dark_els = set(_dark_els).difference(_exposed_els)
    
    return list(dark_els)


def sunlit_elements_rt(
        envir_from,
        envir_to,
        sun_pos_t,
        bool_include_shadowside=True,
        shadow_method='rt',
        COO_check=None):
    """
    Given the position of the sun, determine which elements in the environment are sunlit.

    Parameters
    ----------
    envir_from : dict
        Dictionary containing data of environment from which to trace rays.
    envir_to : dict
        Dictionary containing data of environment to which to trace rays.
    sun_pos_t : Dict[str, float]
        Dictionary containing the azimuth and zenith angles of the sun.
    include_shadowside : bool, optional
        Whether to include the shadow side of the elements in the output, by default True.
    shadow_method : str, optional
        The method to use for tracing rays, either 'rt' or 'rt_up', by default 'rt'.
    COO_check : Optional[np.ndarray], optional
        Coordinates of the elements in the environment, by default None.

    Returns
    -------
    List[int]
        List of indices of the sunlit elements in the environment.
    """

    zen_rad = sun_pos_t['zen_rad']
    azi_rad = sun_pos_t['azi_rad']

    if zen_rad > math.pi / 2:
        return []
    
    if zen_rad == math.pi / 2:
        zen_rad -= 0.0000001
        
    if azi_rad == math.pi / 2:
        azi_rad -= 0.0000001

    dz_max = int(envir_from['idcs_3D'].shape[2] + 3) 
    dxy_max = -dz_max * math.tan(zen_rad)
    dx_max = dxy_max * math.sin(azi_rad)
    dy_max = dxy_max * -math.cos(azi_rad)

    max_ray_length = int(np.hypot(
        (math.tan(math.radians(85)) * dz_max), dz_max))  # height dependent

    ray_length = int(np.hypot(
        (math.tan(zen_rad) * dz_max), dz_max))
    
    max_diff = min([int(np.ceil(np.max(np.abs([dx_max, dy_max, dz_max])))) + 1, ray_length]) 


    ray_z = np.linspace(0, dz_max, max_diff + 2)[:max_ray_length]
    ray_z = np.linspace(0, dz_max, max_diff + 1)[:max_ray_length]
    ray_z = np.linspace(0, dz_max, max_diff)[:max_ray_length]
    
    ray_xy = -ray_z * math.tan(zen_rad)
    ray_x = ray_xy * math.sin(azi_rad)
    ray_y = ray_xy * -math.cos(azi_rad)

    # Trace rays from the ground up to see which elements are lit and which not
    if shadow_method == 'rt_up':
        sunlit_elements = raytrace_upwards(
            envir_from=envir_from,
            envir_to=envir_to,
            ray_x=ray_x,
            ray_y=ray_y,
            ray_z=ray_z)

    # trace rays down from higher elements
    # seems to be the faster method
    elif shadow_method == 'rt':
        sunlit_elements = raytrace_downwards(
            envir_from=envir_from,
            envir_to=envir_to,
            sun_pos_t=sun_pos_t,
            bool_include_shadowside=bool_include_shadowside,
            ray_x=ray_x,
            ray_y=ray_y,
            ray_z=ray_z,
            dz_max=dz_max)

    else:
        raise ValueError('Invalid argument for raytrace method')

    return sunlit_elements


def raytrace_upwards(envir_from, envir_to, ray_x, ray_y, ray_z):
    """
    Trace rays from the ground up to see which elements are lit and which not.
    
    Parameters
    ----------
    envir_from : dict
        A dictionary containing information about the environment the rays are 
        being traced from.
    envir_to : dict
        A dictionary containing information about the environment the rays are 
        being traced to.
    ray_x : np.ndarray
        An array containing the x-coordinate of the rays being traced.
    ray_y : np.ndarray
        An array containing the y-coordinate of the rays being traced.
    ray_z : np.ndarray
        An array containing the heights of the rays being traced.
        
    Returns
    -------
    sunlit_elements : list of int
        A list of indices of the elements that are lit by the rays.
    """
    
    sunlit_elements = []

    max_ray = np.flip(np.vstack([-ray_x[1:], -ray_y[1:], ray_z[1:]]).T, 0)

    start_points = envir_from['coo']
    rays = np.tile(start_points[:, np.newaxis, :],
                   (1, len(max_ray), 1)) + max_ray

    rays_int = rays.astype(int)

    for i, ray in enumerate(rays_int):
        pre_filter_idcs = ray.T
        validity_point = np.array(
            [
                np.logical_and(
                    0 <= pre_filter_idcs[i],
                    pre_filter_idcs[i] < envir_to["idcs_3D"].shape[i]) for i in range(
                    len(
                        envir_to["idcs_3D"].shape))]).all(0)
        filtered_idcs = pre_filter_idcs[:, validity_point]

        idcs_found = envir_to["idcs_3D"][tuple(filtered_idcs.tolist())]
        sunny_bool = not (idcs_found > 0).any()

        if sunny_bool:
            sunlit_elements.append(i)

    return sunlit_elements

def raytrace_downwards(
        envir_from,
        envir_to,
        sun_pos_t,
        bool_include_shadowside,
        ray_x,
        ray_y,
        ray_z,
        dz_max):
    """Trace rays down from higher elements to see which elements are sunlit.
    
    The concept is that one single ray will be traced. This is the longest possible
    ray, given highest point and the given  and that that ray will be 
    transformed to an array to include all coordinates that are under the ray.
    
    
    Parameters
    ----------
    envir_from : dict
        A dictionary containing information about the environment the rays are 
        being traced from.
    envir_to : dict
        A dictionary containing information about the environment the rays are 
        being traced to.
    sun_pos_t : dict
        A dictionary containing the azimuth and zenith angle of the sun in radians.
    bool_include_shadowside : bool
        Whether to exclude elements that have at least one sun-facing element 
        from the `dark_els`, so that they potentially can be added to the 
        resulting `sunlit_elements`
    ray_x : np.ndarray
        An array containing the x-coordinate of the rays being traced.
    ray_y : np.ndarray
        An array containing the y-coordinate of the rays being traced.
    ray_z : np.ndarray
        An array containing the heights of the rays being traced.
    dz_max : int
        The maximum depth of the rays being traced.
    
    Returns
    -------
    sunlit_elements : np.ndarray
        An array containing the indices of the elements that are sunlit.
    """
    
    # Determine which of the face-orientations is sunlit and which are not
    exposed_faces, dark_faces = exposed_faces_sun(zen_deg=sun_pos_t['zen_deg'], azi_deg=sun_pos_t['azi_deg'])
    
    _exposed_els = np.concatenate(envir_to['faces_idcs'][exposed_faces]) 
    _dark_els = np.concatenate(envir_to['faces_idcs'][dark_faces])
    
    # if elements have faces in both sun-lit and non-sun-lit directions, 
    # exclude them from the list of dark elements.
    if bool_include_shadowside:
        dark_els_shadowside = include_shadowside(_exposed_els, _dark_els)
    else:
        dark_els_shadowside = _dark_els

    max_ray = np.flip(np.vstack([ray_x[1:], ray_y[1:], -ray_z[1:]]).T, 0) # can be recoded

    ray_downtriangle = np.empty((0, 3))
    for i, coords in enumerate(max_ray):
        ray_downtriangle_temp = np.vstack([coords[0] * np.ones(i + 2),
                                           coords[1] * np.ones(i + 2),
                                           np.linspace(coords[2], -dz_max, i + 2)]).T
        ray_downtriangle = np.vstack(
            (ray_downtriangle, ray_downtriangle_temp))

    shadow_causing_els = np.where(envir_from['faces_lst'][:, dark_faces].any(
        1) & envir_from['faces_lst'][:, 3])[0]
    
    if len(shadow_causing_els) == 0:
        return np.arange(len(envir_from["coo"]))

    # add the elements that are missing in the overlap between the dark
    # faces
    ngbrs = np.concatenate(np.array(envir_from['idcs_ngbrs'])[
                           shadow_causing_els]).astype(int)
    ngbrs_unique, counts = np.unique(ngbrs, return_counts=True)
    overlapping_ngbrs = ngbrs_unique[counts > 1]
    shadow_causing_els = list(
        set(shadow_causing_els).union(overlapping_ngbrs))

    start_points = envir_from["coo"][np.array(shadow_causing_els)]
    
    
    eps = 0.0001
    
    
    start_points += [0, 0, -eps]
    
    if 0 in dark_faces:
        start_points += [-eps, 0, 0]
    if 1 in dark_faces:
        start_points += [eps, 0, 0]
        
    if 4 in dark_faces:
        start_points += [0, -eps, 0]
    if 5 in dark_faces:
        start_points += [0, eps, 0]
    
    

    if len(start_points) > 0:
        
        shadow_coords = get_shadow_coords(
            ray_downtriangle, 
            start_points,
            envir_to['idcs_3D'])
        
        shadow_coords = np.round(shadow_coords).astype(np.int64)  
        dark_els_by_raytrace = hf.coo_to_idcs_jit(
            coo=shadow_coords.astype(int),
            idcs_3D=envir_to['idcs_3D'])

        dark_els = np.append(dark_els_shadowside, dark_els_by_raytrace)

        all_elements = np.arange(len(envir_to['coo']))
        # define all elements that are not dark to the sunlit elements
        sunlit_elements = list(set(all_elements).difference(dark_els))
        
    else:
        sunlit_elements = np.arange(len(envir_to['coo']))

    return sunlit_elements


@numba.njit(fastmath=True, parallel=True)
def get_shadow_coords(ray_downtriangle, start_points, idcs_3D_to):
    """Return coordinates where there is shadow, using the down triangle ray 
    and the starting points of the rays.
    
    Given the coordinates of the rays of the sunlight, the starting points of 
    the rays and the 3D indices of the target environment, the function returns 
    the coordinates of the points that are in shadow.
    
    It uses numba for faster execution and parallelization.
    
    Parameters
    ----------
    ray_downtriangle : numpy array of shape (n,3)
        The coordinates of the rays of the sunlight.
    start_points : numpy array of shape (n,3)
        The starting points of the rays.
    idcs_3D_to : numpy.ndarray, shape (x,y,z)
        A 3D array containing the indices corresponding to the given coordinates.
        
    Returns
    -------
    shadow_coords_arr : numpy array of shape (m,3)
        The coordinates of the points that are in shadow.
    """
    # make an empty list that will result in a nested list of shadow coordinates
    shadow_coords_lst = []

    start_points = np.subtract(start_points, np.array([0.5, 0.5, 0.5])) 

    for start_point in start_points:
        ray_from_startpoint = trace_ray_from_point(start_point, ray_downtriangle, idcs_3D_to)
        for coords in ray_from_startpoint:
            shadow_coords_lst.append(coords) 

    shadow_coords_arr = hf.list_to_2d_array(shadow_coords_lst)

    return shadow_coords_arr


@numba.njit(fastmath=True)
def trace_ray_from_point(start_point, ray_downtriangle, idcs_3D_to):
    """Add the `ray_downtriangle` to the `start_point` and omit coordinates
    that are outside of the boundaries."""
    
    # ray_from_startpoint = np.add(ray_downtriangle, start_point).astype(np.int64)
    ray_from_startpoint = np.add(ray_downtriangle, start_point)
    
    ray_from_startpoint_int = np.empty_like(ray_from_startpoint, dtype=np.int64)
    np.round(ray_from_startpoint, 0, ray_from_startpoint_int)
    # ray_from_startpoint_int = ray_from_startpoint_int.astype(int)
    
    # crop coordinates that are outside of the bounds
    ray_from_startpoint_int = ray_from_startpoint_int[
        np.where(((ray_from_startpoint_int[:, 0] >= 0) & 
                  (ray_from_startpoint_int[:, 0] < idcs_3D_to.shape[0]) &
                  (ray_from_startpoint_int[:, 1] >= 0) &
                  (ray_from_startpoint_int[:, 1] < idcs_3D_to.shape[1]) &
                  (ray_from_startpoint_int[:, 2] >= 0) &
                  (ray_from_startpoint_int[:, 2] < idcs_3D_to.shape[2])))]

    return ray_from_startpoint_int


@numba.njit
def exposed_faces_sun(zen_deg: float, azi_deg: float):
    """
    Determines the exposed and dark faces based on the position of the sun. 
    
    Parameters
    ----------
    

        
    Returns
    -------
    exposed_faces : list of ints
        Indices of the orientations that are exposed to the sun
    dark_faces : list of ints
        Indices of the orientations that are not exposed to the sun
    """
    if 0 <= zen_deg <= 90:
        exposed_faces = [3]

        if 0 <= azi_deg <= 180:
            exposed_faces.append(1)

        if 90 <= azi_deg <= 270:
            exposed_faces.append(5)

        if 180 <= azi_deg <= 360:
            exposed_faces.append(0)

        if ((270 <= azi_deg <= 360) or (0 <= azi_deg <= 90)):
            exposed_faces.append(4)

    else:
        exposed_faces = [np.int64(x) for x in range(0)]

    dark_faces = [
        face for face in [
            0,
            1,
            3,
            4,
            5] if face not in exposed_faces]

    return exposed_faces, dark_faces

