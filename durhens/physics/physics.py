#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:50:58 2020

@author: gijshenstra
"""

import datetime

import numpy as np
import math
import pandas as pd

from helpers import ray_tracing as rt
import matplotlib.pyplot as plt
from helpers import help_functions as hf

# source: https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubDIV/Zonnestraling_in_Nederland.pdf

# %% Radiation

def shortwave(t, envir_from, envir_to, q_sw, albedo, physics, sun_pos_t, sunlit_els, shadow_method, radiations_t, spread_shadow=False, envir_vf=None, force_recalc=False, q_sw_out_els_refl_prev_t=None, reflections=True, error_reflections=0.00005, max_reflections=25):
    
    nr_sw_refl = 0
    stop_reflections = False
    
    if physics["sw_in_dir"]:
        if len(sunlit_els[t]) == 0 or force_recalc:
    
            # no sunlit elements when sun is down
            if sun_pos_t['zen_rad'] >= math.pi/2:
                
                sunlit_els[t] = []

            # calculate sunlit elements
            if (shadow_method == 'rt_up' or shadow_method == 'rt'):
                sunlit_els[t] = rt.sunlit_elements_rt(
                    envir_from=envir_from,
                    envir_to=envir_to,
                    sun_pos_t=sun_pos_t,
                    shadow_method=shadow_method,
                    bool_include_shadowside=True)
    
        if len(sunlit_els[t]) > 0:
            
            # make a list of sunlit elements that change the state from lit to
            # unlit or unlit to lit, so to spread the illumination over the 
            # simulations steps
            if spread_shadow:
                sunlit_els_remain = list(set(sunlit_els[t - 1]).intersection(sunlit_els[t]))
                sunlit_els_change = list(set(sunlit_els[t - 1]).symmetric_difference(sunlit_els[t]))
              
            else:
                sunlit_els_remain = sunlit_els[t]  
                sunlit_els_change = []
            
            q_sw['in_dir'][t] = solar_irradiance(
                dir_sol_irr=radiations_t['Dir_perp'], 
                nr_els=len(envir_to["coo"]), 
                sunlit_els_remain=sunlit_els_remain, 
                sunlit_els_change=sunlit_els_change,
                faces_idcs=envir_to["faces_idcs"], 
                sun_pos_t=sun_pos_t,
                beer_lambert_attenuation=False)
            
        else:
            q_sw['in_dir'][t] = q_sw['in_dir'][t] * 0
            
    else:
        pass
        
    # Diffuse component (SW)
    if physics["sw_in_diff"] and (sun_pos_t['zen_rad'] < math.pi / 2):
        q_sw['in_diff'][t] = radiations_t["Diff_hor"] * envir_vf["dense_svf"]
    else:
        q_sw['in_diff'][t] = 0
    
    q_sw['in_from_sky'][t] = q_sw['in_dir'][t] + q_sw['in_diff'][t]
       
    
    # iterate the part where the reflected sw will be calculated 
    # stop at a maximum number of reflections or until sum of the reflected 
    # radiation changes only slightly.
    while not stop_reflections and not nr_sw_refl > max_reflections:
    
        # Reflected component (SW)
        if physics["sw_out_refl"]:
            if q_sw['out_els_refl'][t-1] is None:
                raise ValueError('Specifying q_sw_out_els_t is required for calculation.')
            if envir_vf is None:
                raise ValueError('Specifying envir_vf is required for calculation.')
            
            try:
                coo_arr = envir_vf['coords']
                vf_arr = envir_vf['data']
                upper_triangular = False
            except KeyError:
                coo_arr = envir_vf['coords_triu']
                vf_arr = envir_vf['data_triu']
                upper_triangular = True
                
            # flat surface (no coordinates of objects)
            if len(coo_arr) == 0:
                q_sw['in_from_els'][t] = 0
                q_sw_out_els_refl_prev_t = q_sw['in_from_els'][t]
                
            else:
                # if no sw radiation is defined, load the reflected radiation from previous timestep
                # after the first iteration a sw will be defined and that radiation will then be used.
                if q_sw_out_els_refl_prev_t is None:
                    q_sw_out_els_refl_prev_t = q_sw['out_els_refl'][t-1]

                q_sw['in_from_els'][t] = hf.coo_multiplication(
                    coo_arr=coo_arr,
                    vf_arr=vf_arr,
                    data=q_sw_out_els_refl_prev_t,
                    upper_triangular=upper_triangular,
                    nr_els=int(len(q_sw['in_from_els'][t])))
                
        else:
            q_sw['in_from_els'][t] = 0
            
        q_sw['in'][t] = q_sw['in_from_sky'][t] + q_sw['in_from_els'][t]
        
        q_sw['abs'][t] = q_sw['in'][t] * (1 - albedo)[:, np.newaxis]
        q_sw['out_els_refl'][t] = q_sw['in'][t] * (albedo)[:, np.newaxis]
        
        if q_sw_out_els_refl_prev_t.sum() == 0:
            stop_reflections = True
        
        nr_sw_refl += 1
        
        difference_refl = abs((q_sw['out_els_refl'][t].sum()-q_sw_out_els_refl_prev_t.sum()) / q_sw_out_els_refl_prev_t.sum())
        
        # stop iteration when sum of the outgoing reflected sw is smaller than 1.
        if q_sw['out_els_refl'][t].sum() < 1:
            stop_reflections = True
        
        if difference_refl < error_reflections or not reflections:
            stop_reflections = True            
    
        q_sw_out_els_refl_prev_t = q_sw['out_els_refl'][t].copy()
    
    return q_sw, sunlit_els, nr_sw_refl


def solar_irradiance(
        dir_sol_irr, 
        nr_els,
        sunlit_els_remain,
        sunlit_els_change,
        faces_idcs,
        sun_pos_t,
        beer_lambert_attenuation=False):

    q_sw_in_dir = np.zeros((nr_els, 6))
    
    exposed_faces, _ = rt.exposed_faces_sun(zen_deg=sun_pos_t['zen_deg'], azi_deg=sun_pos_t['azi_deg'])
    
    for exposed_face in exposed_faces:
        if exposed_face == 0 or exposed_face == 1:
            q_sw_face = dir_sol_irr * math.sin(sun_pos_t['zen_rad']) * abs(math.sin(sun_pos_t['azi_rad']))
            
        if exposed_face == 3:
              q_sw_face = dir_sol_irr * math.cos(sun_pos_t['zen_rad'])
            
        if exposed_face == 4 or exposed_face == 5:
            q_sw_face = dir_sol_irr * math.sin(sun_pos_t['zen_rad']) * abs(math.cos(sun_pos_t['azi_rad']))
            
            
        sunlit_els_remain_face = np.intersect1d(sunlit_els_remain, faces_idcs[exposed_face])
        sunlit_els_change_face = np.intersect1d(sunlit_els_change, faces_idcs[exposed_face])
    
        if len(sunlit_els_remain) > 0:
            q_sw_in_dir[sunlit_els_remain_face, exposed_face] += q_sw_face
        if len(sunlit_els_change) > 0:
            q_sw_in_dir[sunlit_els_change_face, exposed_face] += 1/2 * q_sw_face

    return q_sw_in_dir


def dewpoint_approximation(T_kelvin, rel_hum):
    """https://en.wikipedia.org/wiki/Dew_point#Calculating_the_dew_point"""
    def gamma(T, rel_hum):
            return (A * T / (B + T)) + np.log(rel_hum / 100.0)
    
    
    T_degC = T_kelvin - 273.15
    if not (T_degC > -45).all() or not (T_degC < 60).all():
        raise ValueError(
            "input temperature out of range (allowed: -45 degC < T < 60 degC)")
    T_dp = (B * gamma(T_degC, rel_hum)) / (A - gamma(T_degC, rel_hum))
    # if not (T_dp > 0).all() and (T_dp < 50).all():
    #     raise ValueError("calculated temperature out of range (allowed: 0 degC < T_dp < 50 degC)")
    return T_dp

def emissivity_sky(T_dp_degC):
    """Sky emissivity according to the Martin and Berdahl (1984).
    """
    
    return 0.711 + (0.0056 * T_dp_degC) + 0.000073 * T_dp_degC**2


# @profile
def glob_rdtn_no_atmosph(zen_rad, azi_rad, date, lat, sol_const):
    """Calculate the global radiation at the top of the atmosphere.

    Depending on the geographic location and the day of the year.
    """
    d = date.dayofyear - 1

    eta = 2 * math.pi / 365

    theta = zen_rad
    psi = azi_rad
    phi = np.deg2rad(lat)  # geographical width observer

    sin_delta = (np.cos(theta) * np.sin(phi)
                 - np.sin(theta) * np.cos(phi) * np.cos(psi))
    delta = np.arcsin(sin_delta)
    cos_delta = np.cos(delta)

    cos_omega = (np.cos(theta) - sin_delta * np.sin(phi)) / \
        (cos_delta * np.cos(phi))

    R_over_R_sqrd = 1.000110 + 0.034221 * np.cos(eta * d) + 0.000719 * np.cos(
        2 * eta * d) + 0.001280 * np.sin(eta * d) + 0.000077 * np.sin(2 * eta * d)

    E_e = sol_const * R_over_R_sqrd * \
        (sin_delta * np.sin(phi) + cos_delta * np.cos(phi) * cos_omega)

    G_no_atm = E_e

    # no solar irradiance if sun below horizon
    G_no_atm = np.where(zen_rad < math.pi / 2, G_no_atm, 0)
    G_no_atm = np.where(G_no_atm > 0, G_no_atm, 0)

    return G_no_atm


def glob_rdtn_cloudless(G_no_atm, zen_rad, T_L=4.0):
    """Calculate the cloudless global radiation.

    It that depends on the solar radiation impinging the atmosphere and the
    (Beer-Lambert) attenuation within the atmosphere.
    """
    # G_cloudless = G_no_atm * beer_lambert_attntn(zen_rad)
    # sun_elev = math.pi/2-zen_rad
    G_cloudless =  G_no_atm * 0.84*np.exp(-0.027*T_L/np.cos(zen_rad)) # (Kasten en Czeplak, 1980; Kasten, 1983)?
    return G_cloudless


def glob_rdtn(G_cloudless, cloudiness):
    """Calculate the global radiation.

    Calculate the global radiation that depends on the cloudiness (given in
    oktas 0-8).
    
    (Kasten en Czeplak, 1980; Kasten, 1983)
    """
    G = G_cloudless * (1 - 0.72 * (cloudiness / 8)**(3.2)) 

    return G


def beer_lambert_attntn(zen_rad):
    """Calculate the attenuation by the atmosphere."""
    tau_2000 = 0.23  # aerosol optical thickness in 2000
    year_now = datetime.date.today().year
    tau_now = tau_2000 - 0.032 * \
        (year_now - 2000) / 10  # Boers & Siebesma, 2017

    try:
        bl_attntn_factor = np.exp(-tau_now / np.cos(zen_rad))
    except OverflowError:
        bl_attntn_factor = 0

    return bl_attntn_factor

# @profile
def diff_radiation(G, G_no_atm, zen_rad, method="deJong"):
    """Calculate the diffuse part of incoming solar radiation."""
    

    D = np.zeros(np.shape(G))

    # De Jong (1980) heeft op basis van dagsommen voor De Bilt afgeleid
    if method == "deJong":
        R = 0.847 - 1.61 * np.cos(zen_rad) + 1.04 * (np.cos(zen_rad))**2
        
        D = np.where(G / G_no_atm <= 0.22, G, D)
        D = np.where((0.22 < G / G_no_atm) & (G / G_no_atm <= 0.35),
                     G * (1 - 6.4 * (G / G_no_atm - 0.22)**2), D)
        D = np.where((0.35 < G / G_no_atm) & (G / G_no_atm <=
                                              (1.47 - R) / 1.66), G * (1.47 - 1.66 * (G / G_no_atm)), D)
        D = np.where(G / G_no_atm > (1.47 - R) / 1.66, G * R, D)

    # Orgill en Hollands (1977) hebben op basis van gemeten uurlijkse
    # radianties in Toronto de volgende vergelijking gevonden:
    if method == "orgill_hollands":
        D = np.where(G / G_no_atm < 0.35, G * (1.0 - 0.249 * G / G_no_atm), D)
        D = np.where((0.35 < G / G_no_atm) & (G / G_no_atm <= 0.75),
                     G * (1.557 - 1.84 * G / G_no_atm), D)
        D = np.where(G / G_no_atm > 0.75, 0.177, D)

    return D



def longwave(t, q_lw, envir, envir_vf, 
             # emissivity_sky, T_sky, 
             lw_out_sky, materials, physics, T, q_lw_out_els_prev_t=None, reflections=True, error_reflections=0.00005, max_reflections=25):
    
    stop_reflections = False
    nr_lw_refl = 0
    
    
    
    q_lw_out_refl_prev_t = q_lw['out_refl'][t-1]
    
    q_lw['out_sky'][t] = lw_out_sky
    # q_lw['out_sky'][t] = radiative_pwr(emissivity_sky, T_sky) # emissivity_sky is 1 when using T_sky_eff_t
    
    if physics['lw_in_sky']:
        q_lw['in_from_sky'][t] = envir_vf['dense_svf'] * q_lw['out_sky'][t]
   
    if physics['lw_out_emm']:
        q_lw['out_rad'][t] = (
            radiative_pwr(materials['emissivity'], T[t])[:, np.newaxis]
            * envir['faces_lst'])
    
        # down-facing elements do not radiate
        q_lw['out_rad'][t][..., 2] = 0
        
    if q_lw_out_els_prev_t is None:
        q_lw_out_els_prev_t = q_lw['out_els'][t-1]
        
        # fixed so that there's not a leak of lw for t==0
        if not q_lw_out_els_prev_t.any():
            q_lw_out_els_prev_t = q_lw['out_rad'][t]
        
    while not stop_reflections and not nr_lw_refl > max_reflections:
        
        try:
            coo_arr = envir_vf['coords']
            vf_arr = envir_vf['data']
            upper_triangular = False
            raise Exception('Needs to be checked.') #!!!
        except KeyError:
            coo_arr = envir_vf['coords_triu']
            vf_arr = envir_vf['data_triu']
            upper_triangular = True
        
        if len(coo_arr) == 0:
            q_lw['in_from_els'][t] = 0
            q_lw_out_els_prev_t = q_lw['in_from_els'][t]
        
        else:
            if physics['lw_out_refl']:
                q_lw['in_from_els'][t] = hf.coo_multiplication(
                    coo_arr=coo_arr, 
                    vf_arr=vf_arr,
                    data=q_lw_out_els_prev_t, 
                    nr_els=len(q_lw['in_from_els'][t]),
                    upper_triangular=upper_triangular)
        
        q_lw['in'][t] = q_lw['in_from_els'][t] + q_lw['in_from_sky'][t]
        q_lw['abs'][t] = q_lw['in'][t] * (1 - materials['albedo_lw'])[:, np.newaxis]
        
        if physics['lw_out_refl']:
            q_lw['out_refl'][t] = q_lw['in'][t] * (materials['albedo_lw'])[:, np.newaxis]
        
        q_lw['out_els'][t] = q_lw['out_rad'][t] + q_lw['out_refl'][t]
        
        nr_lw_refl += 1
        
        difference_refl = abs((q_lw['out_refl'][t].sum()-q_lw_out_refl_prev_t.sum()) / q_lw_out_refl_prev_t.sum())
        
        if q_lw['out_refl'][t].sum() == 0:
            reflections = False
        
        if difference_refl < error_reflections or not reflections:
            stop_reflections = True
            
        q_lw_out_els_prev_t = q_lw['out_els'][t].copy()
        q_lw_out_refl_prev_t = q_lw['out_refl'][t].copy()

    q_lw['net'][t] = q_lw['abs'][t] - q_lw['out_rad'][t]

    return q_lw, nr_lw_refl


def radiative_pwr(emm, T):
    SB_const = 5.67 * pow(10, -8)

    return emm * SB_const * np.power(T, 4)


# %% other 

# @profile
def conduction_inwards(U_i, T_interior, T):
    """Calculates heat flux by conduction from facade/street to interior of buildings/ground. T in deg C."""
    q_conduction = (U_i * (T_interior - T))
    return q_conduction

# @profile
def convection(wind_speed, T_amb, T, exposed_faces, windy_faces, height=None):
    """Valid between 2 to 20 m/s.
    
    Reference
    ---------
    
    """
    # heat_trans_coeff = 12.12 - 1.16 * wind_speed + 11.6 * wind_speed**(1 / 2) # pre 27/9/23: https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html
    # heat_trans_coeff = 10 # after 27/9/23: somewhat of an average between https://thermal.mayahtt.com/?access=yes#hup-isot and https://thermal.mayahtt.com/?access=yes#vp-isot
    heat_trans_coeff = 4 + 4 * wind_speed  # after 26/1/24

    if isinstance(height, np.ndarray):
        print('heat_trans_coeff per h:', heat_trans_coeff)
        heat_trans_coeff = heat_trans_coeff[height]
        
        
        print('shape heat_trans_coeff', heat_trans_coeff.shape)

    q_convection = (heat_trans_coeff * np.subtract(T_amb, T))[:, np.newaxis] * exposed_faces * windy_faces
        
    return q_convection


def surface_equilbrium_temp(
        q_sw_abs,
        q_lw_abs,
        heat_transf_coeff,
        albedos,
        emissivities,
        sb_const,
        T_sky_eff,
        T_air):
    import itertools
    T_surf_eq = np.zeros(np.shape(q_sw_abs))


    for el, _ in np.ndenumerate(q_sw_abs):
        T_surf_eq[el] = np.roots(np.array([sb_const *
                                           emissivities[el[0]], 0, 0, heat_transf_coeff, -
                                           (q_lw_abs[el] +
                                            q_sw_abs[el] +
                                               heat_transf_coeff *
                                               T_air)])).real.max()
    return T_surf_eq



# unused
def urban_wind_speed(
        u10,
        height,
        front_dens_building,
        front_dens_tree,
        front_dens_total,
        area_map,
        H):
    """Calculate the wind speed at different heights."""

    # constants, 1974 Psychrometry and Psychrometric Charts, as presented by
    # Paroscientific
    A = 17.62
    B = 243.12  # degC

    u_60 = 1.308 * u10

    MacDonald_table = pd.DataFrame(data=[[0.066, 2, 0.04, -0.35, 0.56],
                                         [0.26, 2.5, 0.071, -0.35, 0.5],
                                         [0.32, 2.7, 0.084, -0.34, 0.48],
                                         [0.42, 1.5, 0.08, -0.56, 0.66],
                                         [0.57, 1.2, 0.77, -0.85, 0.92]],
                                   index=[0.08, 0.135, 0.18, 0.265, 1],
                                   columns=['d/H', 'z_w/H', 'z_0/H', 'A/H', 'B'],
                                   )

    idx = np.where(0.1 < MacDonald_table.index)[0][0]

    d = MacDonald_table.iloc[idx]['d/H'] * H  # zero plane displacement
    z_w = MacDonald_table.iloc[idx]['z_w/H'] * H  # Top of the roughness layer
    z_0 = MacDonald_table.iloc[idx]['z_0/H'] * H  # (surface) roughness length

    u_star = 0.4 * u_60 / (np.log((60 - d) / (z_0)))

    # Macdonald 2000
    if 0.6 * front_dens_building + 0.3 * front_dens_tree > 25 / area_map:
        # Parameter for interpolation wind profile
        A = MacDonald_table.iloc[idx]['A/H'] * H
        # Parameter for interpolation wind profile
        B = MacDonald_table.iloc[idx]['B']

        a = 9.6 * front_dens_total  # attenuation coefficient

        # the velocity profile does not satisfy the no-slip
        # condition at the ground, although when a > 5 there is very little flow in the lower
        # canopy so that the mean velocity at z = 0 is effectively zero

        u_zw = u_60 * ((np.log((z_w - d) / (z_0))) /
                       (np.log((60 - d) / (z_0))))
        # mean velocity measured at the top of the obstacles (at z = H )
        u_H = -u_star / B * np.log((A + B * z_w) / (A + B * H)) + u_zw
        # horizontally spatially averaged velocity
        u_height = u_H * np.exp(a * (height / H - 1))

    # Tennekes 1973
    else:
        k = 0.4  # von Karman constant
        u_height = u_star / k * np.log((height - d) / z_0)

    return u_height





    
    
# %% temperature
    
def dTdt(vol_heat_cap, area, thickness, q_net):
    volume = area * thickness
    dTdt = (area * q_net) / (volume * vol_heat_cap)
    return dTdt