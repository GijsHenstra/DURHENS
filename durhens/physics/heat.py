#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:16:47 2020

@author: gijshenstra
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt

from helpers import help_functions as hf

from tqdm import tqdm


def max_dt(dxdydz, D):
    dx = dxdydz[0]
    dy = dxdydz[0]
    if len(dxdydz) == 2:
        dt = dx**2 * dy**2 / (2 * D * (dx**2 + dy**2))
    if len(dxdydz) == 3:
        dz = dxdydz[2]
        dt = dx**2 * dy**2 * dz**2 / (2 * D * (dx**2 + dy**2 + dz**2))
    if not isinstance(dt, (int, float)):
        dt = min(dt)
    return dt


def heat_diffusion_lst(T0_lst, idcs_ngbrs_ma, nr_ngbrs, gridres, D):
    """Propagate with forward-difference in time, central-difference in space.
    """
    u_ngbrs_sum = np.ma.array(T0_lst[idcs_ngbrs_ma], mask=idcs_ngbrs_ma.mask).sum(1).data

    # delta temperature per timestep
    dTdt = D * 1 / (gridres**3) * (u_ngbrs_sum - nr_ngbrs * T0_lst)

    return dTdt


def idcs2vals(idcs_lst, val_lst, func=np.sum):
    """Map indices to corresponding values."""
    vals = np.empty(np.shape(val_lst))
    vals[:] = np.nan


    for i, idcs in enumerate(idcs_lst):
        vals[i] = func(val_lst[idcs])

    return vals


def find_closest_val(arr, find_vals, return_idx=True):
    idcs = []
    found_vals = []
    for find_val in find_vals:
        idx = (np.abs(arr - find_val)).argmin()
        idcs += [idx]
        found_vals += [arr[idx]]

    if return_idx:
        return found_vals, idcs
    if not return_idx:
        return found_vals


def heat_diffusion2D(u0, u):
    """Propagate with forward-difference in time, central-difference in space"""
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
        (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx2
        + (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2)

    u0 = u.copy()
    return u0, u

# %% 

if __name__ == '__main__':

    D = 4. * 10**-6
    Tcool, Thot = 300, 320

    time_plot = [0, 100000, 500000, 1000000, 2000000, 3000000, 4000000]

    two_dim = True
    three_dim = True


    # plate size, m
    w = h = 100.
    # intervals in x-, y- directions, mm
    dx = dy = 1
    # Thermal diffusivity of steel, mm2.s-1

    nx, ny = int(w / dx), int(h / dy)

    # nr of timesteps
    nsteps = 101
    
    # %%
    
    
    t_plot = find_closest_val(time, time_plot)[1]
    
    if two_dim:
        dx2, dy2 = dx * dx, dy * dy
        dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
        dt = 40000
        print('dt for 2D', dt)
        
        time = np.zeros(nsteps)
        for t in range(nsteps - 1):
            time[t] = t * dt

        u0 = Tcool * np.ones((nx, ny))
        u0[25:75, 25:75] = Thot
        u = u0.copy()

        # Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
        # r, cx, cy = 2, 5, 5
        # r2 = r**2
        # for i in range(nx):
        #     for j in range(ny):
        #         p2 = (i*dx-cx)**2 + (j*dy-cy)**2
        #         if p2 < r2:
        #             u0[i,j] = Thot

        

        fig = plt.figure()
        for t in range(nsteps - 1):
            u0, u = heat_diffusion2D(u0, u)
            if t in t_plot:
                fig = plt.imshow(u.copy(), cmap=plt.get_cmap(
                    'hot'), vmin=Tcool, vmax=Thot)
                plt.title(str(time[t]) + '   2D')
                plt.colorbar()
                plt.show()

    # %%

    if three_dim:
        dz = dx
        dt = max_dt((dx, dy, dz), D)
        dt = 40000
        print('dt for 3D', dt)
        
        time = np.zeros(nsteps)
        for t in range(nsteps - 1):
            time[t] = t * dt

        elev = 0 * np.ones((nx, ny))

        cl_test = hf.elevationmap_to_coo(elev, street_height=0, res=1)

        cl_int = cl_test.astype(int)

        idcs_3D = hf.coo_to_idcs3D(coo_int=cl_int, shape=(elev.shape + (int(elev.max()+1),)))
        coords_all_ngbrs = hf.coords_all_adjacent_elements(
            shape=idcs_3D.shape, coo_int=cl_int)
        
        idcs_ngbrs = hf.coos_to_idcs(coords_all_ngbrs, idcs_3D) # !!! stond nog np.array om coords_all_nghbr

        T_lst = np.zeros((nsteps, len(cl_test)))

        T_lst[0, :] = Tcool
        T_lst[0, idcs_3D[25:75, 25:75, 0]] = Thot

        idcs_ngbrs_mtx = np.array(
            list(
                itertools.zip_longest(
                    *np.array(idcs_ngbrs),
                    fillvalue=None))).T
        
        idcs_ngbrs_mask = (idcs_ngbrs_mtx == None)
        
        idcs_ngbrs_mtx[idcs_ngbrs_mtx == None] = 0
        idcs_ngbrs_mtx = idcs_ngbrs_mtx.astype(int)
        idcs_ngbrs_ma = np.ma.array(idcs_ngbrs_mtx, mask=idcs_ngbrs_mask)

        nr_ngbrs = (~idcs_ngbrs_ma.mask).sum(1)
        
        t_plot = find_closest_val(time, time_plot)[1]
        plot_all = False
        for t in range(nsteps - 1):
            time[t] = t * dt
            dTdt = heat_diffusion_lst(T_lst[t], idcs_ngbrs_ma=idcs_ngbrs_ma, nr_ngbrs=nr_ngbrs, gridres=dx, D=D)
            T_lst[t + 1] = T_lst[t] + dTdt*dt
            
            img = T_lst[t][idcs_3D[:, :, 0]]
            if t in t_plot or plot_all:
                plt.imshow(img, cmap='hot', vmin=Tcool, vmax=Thot)
                plt.colorbar()
                plt.title(str(time[t]) + '   3D')
                plt.show()
