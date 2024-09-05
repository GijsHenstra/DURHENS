#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:10:37 2020

@author: gijshenstra
"""

import numpy as np
import math
import matplotlib
import time
import os

from helpers import help_functions as hf

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import proj3d

# draw a vector

from matplotlib.ticker import MaxNLocator


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# %% Extras


def draw_sun(ax, t, loc, sun_zen_rad, sun_azi_rad, data_shape, el_size,
             sun_path_3D=False, sun_path_projection=True, sun_height=True):
    proj_sol_ellipse = 1.5 * data_shape[0] / 2

    sun_pos_rel = np.array([proj_sol_ellipse * np.sin(sun_azi_rad), 
                            -proj_sol_ellipse * np.cos(sun_azi_rad), 
                            proj_sol_ellipse / np.tan(sun_zen_rad)])
    
    sun_center = np.array([loc[0] +
                           data_shape[0] /
                           2 *
                           el_size, loc[1] +
                           data_shape[1] /
                           2 *
                           el_size, 0])[:, np.newaxis]
    sun_pos = sun_center + sun_pos_rel

    if np.rad2deg(sun_zen_rad[t]) < 90:
        ax.scatter(sun_pos[0][t], sun_pos[1][t], sun_pos[2][t],
                   s=200, marker='o', facecolors='none', c='#ffb90f')
    if np.rad2deg(sun_zen_rad[t]) > 90:
        ax.scatter(sun_pos[0][t], sun_pos[1][t], sun_pos[2][t],
                   s=200, marker='o', facecolors='none', c='#3f3d78')
    ax.scatter(sun_pos[0][t], sun_pos[1][t], 0, s=200,
               marker='o', facecolors='none', c='grey', alpha=0.3)

    if sun_path_3D:
        ax.plot(sun_pos[0], sun_pos[1],)
    if sun_path_projection:
        ax.plot([sun_pos[0][t], sun_pos[0][t]], [sun_pos[1][t], sun_pos[1][t]], [
                0, sun_pos[2][t]], color='grey', linestyle='dashed', alpha=0.3)
    if sun_height:
        ax.plot(sun_pos[0], sun_pos[1], color='grey',
                linestyle='dashed', alpha=0.3)

    max_height_sun = sun_pos[2].max()

    return max_height_sun


def create_cc(color_name, num_colors):
    cmap = plt.cm.get_cmap(color_name, num_colors)
#    cmap_sky = plt.cm.get_cmap('Oranges', num_colors)

    cc = []
#    cc_sky = []
    alpha = .1

    for i in range(cmap.N):
        # will return rgba, we take only first 3 so we get rgb
        rgb = cmap(i)[:3] + (alpha,)
#            rgb.append(0.5)
        cc.append(matplotlib.colors.rgb2hex(rgb))

    return cc


def draw_compass(ax, rot_north, compass_arrow_length):
    compass_x = compass_arrow_length * math.sin(math.radians(180 - rot_north))
    compass_y = compass_arrow_length * math.cos(math.radians(180 + rot_north))

    arrow_color = 'r'
    arrow_alpha = 0.3

    compass_arrow = Arrow3D([0, compass_x], [0, compass_y], [0, 0],
                            mutation_scale=10, lw=0.5, arrowstyle='simple',
                            color=arrow_color, alpha=arrow_alpha)

    ax.add_artist(compass_arrow)
    ax.text(compass_x, compass_y, 0, "North", color=arrow_color,
            alpha=arrow_alpha, horizontalalignment='center')


def draw_colorbar(minI, maxI, color_name, shrink=0.7, labelrotation=90):
    from matplotlib import cm

    norm = matplotlib.colors.Normalize(vmin=minI, vmax=maxI)
    m = cm.ScalarMappable(cmap=plt.cm.get_cmap(color_name), norm=norm)
    m.set_array([])
    cbar = plt.colorbar(m, shrink=shrink)
    cbar.ax.tick_params(labelrotation=labelrotation)


def sun_direction_arrow():
    # l_arrow = x_trans
    # l_proj = l_arrow * math.sin(zen_deg)
    # dx = l_proj*math.sin(azi_deg)
    # dy = l_proj*math.cos(azi_deg)
    # dz = l_arrow*math.cos(zen_deg)

    # color_arrow = 'k' if zen_deg<math.pi/2 else 'g'

    # arrow_xyz = Arrow3D([dx,0], [dy,0], [dz,0], mutation_scale=20,
    #         lw=2, arrowstyle="wedge", color=color_arrow)

    # line_xy = Arrow3D([dx,0], [dy,0], [0,0], mutation_scale=20,
    #         lw=2, arrowstyle="-", color="0.7",linestyle="dotted")

    # line_z_xy = Arrow3D([dx,dx], [dy,dy], [dz,0], mutation_scale=20,
    #         lw=2, arrowstyle="-", color="0.7",linestyle="dotted")

    # ax.add_artist(line_xy)
    # ax.add_artist(line_z_xy)
    # ax.add_artist(arrow_xyz)
    return None


# %% 3D voxels plot


# @profile
def voxels_plot(height3d, 
                plot_data, 
                cl,
                shade_2d=None,
                std_color='whitesmoke',
                t=None,
                rot_north=None,
                disp_compass=False,
                sun_zen_rad=None,
                sun_azi_rad=None,
                el_size=1,
                view_dir=[60, 110],
                loc=[0, 0],
                saveimg=False,
                save_ext='.pdf',
                save_path='',
                dpi=300,
                colorbar=True,
                disp_sun=True,
                figtitle='',
                color_name='Greys_r',
                face_visibility=1,
                grid_visibility=1,
                shade_faces=True,
                show=True,
                highlight_els={},
                labels=False,
                ticks=False,
                ax=None,
                aspect='equal',
                overwrite_existing=False,
                ):
    
    if os.path.isfile(str(save_path) + save_ext) and not overwrite_existing:
        print(f'{str(save_path) + save_ext} already exists. Skipped voxels-plot.')
        return
    
    plt.clf()

    fig = plt.figure(1, figsize=(30, 20), dpi=300)
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    facecolors = np.zeros((np.shape(height3d) + (4,)))
    facecolors[..., -1] = face_visibility
    facecolors[height3d] = mpl.colors.to_rgba(std_color)

    # ---- determine coloring of the voxels

    for item in plot_data:
        try:
            colordata_lst = plot_data[item]['data']
        except KeyError:
            # intensity depends on height
            colordata_lst = cl[:, 2]
        colordata3d_float = hf.array2d_to_array3d(colordata_lst, coo=cl.astype(int), shape=height3d.shape)
        color = plot_data[item]['color']
        if 'colorbar_lims' in plot_data[item].keys():
            colorbar_lims = plot_data[item]['colorbar_lims']
            colordata3d_float = np.clip(
                colordata3d_float, colorbar_lims[0], colorbar_lims[1])
        else:
            colorbar_lims = (colordata_lst.min(), colordata_lst.max())

        try:
            mask3d = plot_data[item]['mask3d']
        except KeyError:
            mask3d = height3d


        try:
            if all([isinstance(c, (float, int)) for c in color]):
                if len(color) == 3:
                    color += [1,]
                elif len(color) == 4:
                    pass
                else:
                    raise ValueError(f'Color is not appropriate rgb or rgba: {color}')
                    
                facecolors[mask3d] = color
                
                colorbar = False
            else:
                raise TypeError
        except TypeError:
                try:
                    if isinstance(color, np.ndarray):
                        raise ValueError
                    cmap = plt.get_cmap(color)  # "Greens"
                except ValueError:  # "["color1","color2"] or "green"
                    if isinstance(color, list):
                        cmap_colors = [mpl.colors.to_rgb(
                            c) for c in color]
                        # cmap_name = '2'.join(color)
                        cmap_name = 'cmap_' + item
                        cmap = mpl.colors.LinearSegmentedColormap.from_list(
                            cmap_name, cmap_colors, N=100)
                    else:
                        facecolors = np.where(
                            np.repeat(mask3d[..., np.newaxis], 4, axis=3),
                            mpl.colors.to_rgba(color),
                            facecolors)
                        continue

                if 'cmap' in locals():
                    # color_norm = ((colordata - colorbar_lims[0])
                    #               / (colorbar_lims[1] - colorbar_lims[0]))
                    norm = mpl.colors.Normalize(
                        vmin=colorbar_lims[0], vmax=colorbar_lims[1])
                    colordata3d_rgba = cmap(norm(colordata3d_float))
                    if colorbar:
                        sc = mpl.cm.ScalarMappable(
                            cmap=cmap,
                            norm=mpl.colors.Normalize(vmin=colorbar_lims[0],
                                                      vmax=colorbar_lims[1]))
                        sc.set_array([])
                    del cmap
        
                minI = np.min(colordata_lst)
                maxI = np.max(colordata_lst)
    
                facecolors[mask3d] = colordata3d_rgba[mask3d]

    if isinstance(shade_2d, np.ndarray):
        facecolors[..., :3] *= hf.array2d_to_array3d(shade_2d, cl, shape=height3d.shape)[..., np.newaxis]

    # ---- inspect elements by highlighting

    cl_int = cl.astype(int)
    for color, els in highlight_els.items():
        # if isinstance(color, np.ndarray):
        #     facecolors[cl_int[[els], 0], cl_int[[els], 1], cl_int[[els], 2], :3] = color[:, :3]
            
        # else:
        facecolors[cl_int[[els], 0], cl_int[[els], 1], cl_int[[els], 2], :3] = mpl.colors.to_rgb(color)
    

    # ---- shading and solar path drawing

    if (sun_zen_rad is None) and (sun_azi_rad is None):
        sun_zen_rad_t = math.radians(10)
        sun_azi_rad_t = math.radians(10)
    else: 
        try:
            sun_zen_rad_t = sun_zen_rad[t]
            sun_azi_rad_t = sun_azi_rad[t]
        except: 
            sun_zen_rad_t = sun_zen_rad
            sun_azi_rad_t = sun_azi_rad
            pass
        
    # determing position of the sun for shadow effect
    sun_elev = 90 - np.rad2deg(sun_zen_rad_t) #!!! 
    sun_azi = 180 - np.rad2deg(sun_azi_rad_t)
    ls = LightSource(sun_azi, sun_elev)

    if disp_sun:
        max_height_sun = draw_sun(
            ax, t, loc, sun_zen_rad, sun_azi_rad,
            data_shape=np.shape(height3d), el_size=1)
        
    ax.voxels(height3d, 
              facecolors=facecolors,
              edgecolor=[0, 0, 0, grid_visibility],
              lightsource=ls, 
              shade=shade_faces)

    if disp_compass:
        draw_compass(
            ax, rot_north, compass_arrow_length=np.shape(height3d)[0] / 2)
    if colorbar:
        try:
            plt.colorbar(sc)
        except Exception as e:
            print(e)

    # interaction with the plot (slow!)
    # cid = fig.canvas.mpl_connect('button_release_event', on_click)
    # fig.canvas.mpl_connect('close_event', handle_close)

    ax.view_init(elev=90 - view_dir[0], azim=view_dir[1] - 90)

    xy_lims = np.max(np.shape(height3d))

    ax.set_xlim3d(0, xy_lims)
    ax.set_ylim3d(xy_lims, 0)
    ax.set_zlim3d(0, xy_lims * (3 / 4))

    try:
        ax.set_aspect(aspect)
    except NotImplementedError:
        pass

    ax.zaxis.line.set_lw(0.)
    ax.set_zticks([])
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

   

    

    ax.grid(b=None, which='major', axis='both')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        
    if labels:
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
#        ax.set_zlabel('Z axis')
    
    fig.tight_layout()

    if saveimg:
        plt.savefig(str(save_path) + save_ext, dpi=dpi, bbox_inches='tight')
    if show:
        plt.pause(0.0001)

    if len(figtitle) > 0:
        plt.title(figtitle)

    if not show:
        plt.close(fig)
    


def hex2rgba(colors_hex, alpha):
    rgb = tuple(int(colors_hex.lstrip('#')[i:i + 2], 16)
                / 255.0 for i in (0, 2, 4))
    return rgb + (alpha,)


# def data3d_to_rgba(data, color_name, intensity_lims,
#                    num_colors=255, alpha_default=1):
#     # color initializing
#     cc_hex_set = create_cc(color_name, num_colors)
#     cc_rgb = list(map(lambda cc: list(
# int(cc.lstrip('#')[i:i + 2], 16) / num_colors for i in (0, 2, 4)),
# cc_hex_set))

#     min_data = intensity_lims[0]
#     max_data = intensity_lims[1]

#     maxI = max_data
#     minI = min_data

#     norm_data = np.round((data - minI) / (maxI - minI)
#                          * (num_colors - 1)).astype(int)

#     rgba = np.zeros(data.shape, dtype=(float, 4))

#     for ix in range(data.shape[0]):
#         for iy in range(data.shape[1]):
#             for iz in range(data.shape[2]):
#                 if 0 <= norm_data[ix, iy, iz] <= num_colors:
#                     rgba[ix, iy, iz] = cc_rgb[norm_data[ix, iy, iz]] + \
#                         [alpha_default]

#     return rgba, minI, maxI


# %% Surface plot

def plot_surf3d_simple(
        height,
        t=None,
        colordata=None,
        sun_zen_rad=None,
        sun_azi_rad=None,
        loc=[0, 0],
        intensity_lims=None,
        el_size=1,
        type='surface',
        save_path='',
        disp_sun=False,
        disp_compass=True,
        disp_colorbar=True,
        color_name='Greys_r',
        rot_north=0,
        shade=True,
        view_dir=[60, 110],
        title='',
        outp_fig=1,
        ve=1,
        antialiased=False,
        disp_z_axis=False):

    # Set up plot
    fig = plt.figure(outp_fig)
    ax = fig.add_subplot(111, projection='3d')

    z = height

    nrows, ncols = z.shape
    x = np.linspace(loc[0], loc[0] + height.shape[0] * el_size, ncols)
    y = np.linspace(loc[1] + height.shape[1] * el_size, loc[1], nrows)
    x, y = np.meshgrid(x, y)

    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    if colordata is None:
        colordata = height

    if intensity_lims is None:
        minI = np.min(colordata)
        maxI = np.max(colordata)

    else:
        minI = intensity_lims[0]
        maxI = intensity_lims[1]

    if disp_colorbar:
        draw_colorbar(minI, maxI, color_name)
    if disp_compass:
        draw_compass(
            ax, rot_north, compass_arrow_length=np.shape(height)[0] / 2)

    if disp_sun:
        draw_sun(ax, t, loc, sun_zen_rad, sun_azi_rad,
                 data_shape=np.shape(height), el_size=el_size)

        sun_elev = 90 - math.degrees(sun_zen_rad[t])
        sun_azi = math.degrees(sun_azi_rad[t])
        ls = LightSource(sun_azi, sun_elev)

    if not disp_sun:
        ls = LightSource(45, 45)

    facecolors = ls.shade(colordata, plt.cm.get_cmap(
        color_name), blend_mode='overlay')
    # facecolors= ls.shade(colordata, plt.cm.get_cmap(color_name), blend_mode='hsv')

    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        linewidth=0,
        facecolors=facecolors,
        shade=shade,
        antialiased=antialiased)

    # for line in ax.xaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax.yaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax.zaxis.get_ticklines():
    #     line.set_visible(False)

    if not disp_z_axis:
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_title(title)
    ax.view_init(elev=90 - view_dir[0], azim=view_dir[1] - 90)

    ax.set_xlim3d(loc[0], loc[0] + (np.max(x) - np.min(x)))
    ax.set_ylim3d(loc[1], loc[1] + (np.max(x) - np.min(x)))
    ax.set_zlim3d(0, (np.max(x) - np.min(x)))

    if len(save_path) > 0:
        plt.savefig(str(save_path + '.pdf'), dpi=300, bbox_inches='tight')

    plt.pause(0.00001)

    plt.show()
