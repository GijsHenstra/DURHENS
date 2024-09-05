#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:35:15 2020

@author: gijshenstra
"""
import matplotlib
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

from helpers import help_functions as hf

import numpy as np
import pandas as pd

import cv2
from helpers import online_data
# import get_data

from helpers import crs
import pdb
import scipy



# from matplotlib import cm


def dashboard(
        data,
        time_now,
        xdata=None,
        shape=(4, 1),
        axs=None,
        data_title=None,
        title=False,
        plot=False,
        alpha=0.7,
        hide_spines=[]):
    """Generate plot to display multiple data values over time."""

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
            # print('sub_data', sub_data)
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

                # print('values', values)
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
        # axs[i].set_xticklabels(ax.get_xticklabels())

        if title:
            axs[i].set_title(sub_plot_data)
        axs[i].margins(0.01, 0.15)
        for spine in hide_spines:
            axs[i].spines[spine].set_visible(False)

        # axs[i].yaxis.tick_right()
        # axs[i].grid(axis='y')
        # axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[i].legend(prop={'size': 8})

        try:
            lims = data[sub_plot_data]["lims"]
            axs[i].set_ylim(lims)
        except Exception:
            pass

        # else:
            # axs[i].xticks(np.arange(0, 1, 1 / len(min_data)))
        axs[i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        axs[i].xaxis.set_major_locator(mpl.dates.HourLocator(interval=6))
        if i != len(data) - 1:
            # axs[i].get_xaxis().set_visible(False)
            axs[i].set_xticklabels([])
            axs[i].xaxis.set_ticks_position('none')
        axs[i].axvline(time_now, color='red', alpha=0.4)
        if (i + 1) % shape[0] == 0:
            ymin = 0
        else:
            ymin = -1

    # fig.autofmt_xdate()

    if plot:
        # fig.tight_layout()
        plt.show(block=False)

    return axs


def density_scatter(x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = scipy.interpolate.interpn((0.5*(x_e[1:] + x_e[:-1]), 
                           0.5*(y_e[1:]+y_e[:-1]) ), 
                          data, 
                          np.vstack([x,y]).T, 
                          method = "splinef2d", 
                          bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = mpl.colors.Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


def gdf(data_gdf, show_plot=True, save_dir=None, gridsize=None,
        address=None, column=[None], bbox=None, crs_out=None,
        crs_init=None, figsize=(10, 10), nr_colors=8,
        cmap=plt.cm.viridis, return_ticks=False, ticks=None):
    """Plot GeoDataframe."""
    
    
    
    for col in column:
        fig, ax = plt.subplots(1, figsize=figsize)
            
        try:
            figtitle = ''
            
            

            if col is not None:
                if data_gdf[col].dtype == np.float64 or data_gdf[col].dtype == np.int64:
                    scheme = 'fisher_jenks'
                else:
                    scheme = None
                # filter nonzero data
                _data_gdf = data_gdf[data_gdf[col].notna()]
                figtitle = str(col).title()
                
                if len(_data_gdf) == 0:
                    raise ValueError(f'"{col}" has no printable data. Plot skipped.')
            else:
                _data_gdf = data_gdf
                scheme = None

            figtitle += '\n'

            

            # figure initialisation
            

            if gridsize is not None:
                figtitle += str(f'Area of {gridsize} x {gridsize} m')

            if isinstance(address, str):
                figtitle += str(' ' + address)
            ax.set_title(figtitle)

            if crs_out is not None:
                if data_gdf.crs is not crs_out:
                    _data_gdf = data_gdf.set_crs(crs_init).to_crs(crs_out)
                    bbox = crs.transform(
                        bbox, crs_in=crs_init, crs_out=crs_out)

                    crs_init = crs_out # to be deleted?

            if col is None:
                _data_gdf.plot(ax=ax, linewidth=1, edgecolor='black')

            else:
                _data_gdf.plot(ax=ax, column=col, scheme=scheme,
                              k=nr_colors, cmap=cmap, linewidth=1,
                              edgecolor='black', legend=True)

            ax.set_facecolor("lightgray")
            # plt.axis('equal')

            if bbox is not None:
                ax.set_xlim(bbox[0], bbox[2])
                ax.set_ylim(bbox[1], bbox[3])

            # write the labels out in full
            ax.ticklabel_format(useOffset=False)

            # plt.set_aspect('equal')
            ax.set_aspect(1. / ax.get_data_ratio())


            if save_dir is not None:
                plt.savefig(save_dir / (str(col) + '.pdf'),
                            bbox_inches='tight')

        except ValueError as e:
            print(e)
            plt.clf()
            pass

        if show_plot:
            plt.show(block=False)
        else:
            plt.clf()

    if return_ticks:
        x_locs, _ = plt.xticks()
        y_locs, _ = plt.yticks()
        return (x_locs, y_locs)

    else:
        return

def plot_properties():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    linestyles = ['-', '--', '-.', ':']
    linewidths = [1, 2, 3, 4, 5]

    return colors, linestyles, linewidths


def surf_2d(ds, color_df, xlims, ylims, title='', plot_3d=False,
            plot_2d=True, save_path='', cmap='Reds', grid=False):

    z = color_df

    # x, y = np.meshgrid(range(xlims[0],xlims[1]), range(ylims[0],ylims[1]))
    x, y = np.meshgrid(np.arange(np.shape(ds)[0]), np.arange(np.shape(ds)[1]))

    if plot_3d:
        start = time.time()
        # show height map in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, extent=(0, 1, 0, 1))
        ax.set_xlim(BBox[0], BBox[1])
        ax.set_ylim(BBox[2], BBox[3])
        plt.title('z as 3d height map')
        ax.set_zlim3d(0, gridsize)
        plt.show()
        print("3dplot took:", time.time() - start)

    if plot_2d:
        plt.clf()
        start = time.time()

        # show height map in 2d
        # fig, ax = plt.subplots()

        fig = mpl_interaction.figure_pz(1)
        # pan_zoom = mpl_interaction.PanAndZoom(fig)
        # print(pan_zoom)
        # plt.xlim(xmin,xmax)
        # plt.ylim(ymax, ymin)

        # ax.xaxis.tick_top()

        ax = plt.gca()

        plt.title('z as 2d heat map' + title)
        p = plt.imshow(z,
                       extent=[xlims[0], xlims[1], ylims[1], ylims[0]],
                       aspect=float(np.diff(xlims) / np.diff(ylims)),
                       cmap=plt.get_cmap(cmap))
        plt.clim(color_df.min(), np.min([50, color_df.max()]))
        plt.colorbar(p)

        if grid:
            # loc = plticker.MultipleLocator(base=test) # this locator puts ticks at regular intervals
            # ax.xaxis.set_major_locator(loc)

            y_tick = np.diff(plt.yticks()[0])[0]
            ax.xaxis.set_major_locator(plt.MultipleLocator(y_tick * 2))

            # print('y tick interval',y_tick)

            ax.grid(which='major', color='w', linestyle='-', linewidth=0.5)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=0.3)

        if len(save_path) > 0:
            plt.savefig(str(save_path + '.png'), dpi=1000)

        plt.show()
        print("2dplot took:", round(time.time() - start), 2)


def plot_df(
        df1,
        time_array,
        albedo_set,
        hw_set,
        max_x,
        df2=[],
        plot=True,
        save=False,
        xlabel='Time [hours]',
        ylabel1='',
        ylabel2='',
        title=''):

    plotax2 = len(df2) > 0

    colors, linestyles, linewidths = plot_properties()

    fig, ax1 = plt.subplots(figsize=(6, 6))

    if plotax2:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.set_aspect('equal', 'box')

    for hw_iter in range(len(hw_set)):
        for albedo_iter in range(len(albedo_set)):

            #            ax1.plot(time_array_s/3600,T_all_ave_t[:,albedo_iter,hw_iter], label='average all sides, albedo = %.1f, H/W =  %.1f' % (albedo_set[albedo_iter],hw_set[hw_iter]),linewidth=linewidths[hw_iter],color=colors[0],linestyle=linestyles[0])
            #            ax1.plot(time_array_s/3600,T_ave_t[:,2,albedo_iter,hw_iter], label='average street, albedo = %.1f, H/W =  %.1f' % (albedo_set[albedo_iter],hw_set[hw_iter]),linewidth=linewidths[hw_iter],color=colors[albedo_iter],linestyle=linestyles[0])
            ax1.plot(time_array,
                     df1[:,
                         albedo_iter,
                         hw_iter],
                     label='average street, albedo = %.1f, H/W =  %.1f' % (albedo_set[albedo_iter],
                                                                           hw_set[hw_iter]),
                     linewidth=linewidths[hw_iter],
                     color=colors[albedo_iter],
                     linestyle=linestyles[0])
            if plotax2:
                ax2.plot_date(time_array,
                              df2[:,
                                  albedo_iter,
                                  hw_iter],
                              label='average street, albedo = %.1f, H/W =  %.1f' % (albedo_set[albedo_iter],
                                                                                    hw_set[hw_iter]),
                              linewidth=linewidths[hw_iter],
                              color=colors[albedo_iter],
                              linestyle=linestyles[0])

    fig.legend()

    ax1.set_xlabel(xlabel)

    hfmt = matplotlib.dates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(hfmt)
    ax1.set_autoscale_on(True)

    ax1.set_ylabel(ylabel1, color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])

    if plotax2:
        ax2.set_ylabel(ylabel2, color=colors[1])
        ax2.tick_params(axis='y', labelcolor=colors[1])

    # fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.title(title + ' for H/W = ' + str(round(np.min(hw_set), 1)) + ' to ' +
              str(round(np.max(hw_set), 1)) + ', for x and y = ' + str(max_x))

    if plot:
        plt.show()

    if save:
        plt.savefig(
            'Temp over time, x and y = ' +
            str(max_x) +
            ' for street-wall albedo = ' +
            str(albedo_set) +
            '.pdf',
            dpi=300,
            bbox_inches='tight')
    if not save:
        plt.close(fig)


def overview_plot(df, plot_vars, plot_order=[
                  'q_sw', 'q_lw', 'T', ''], plot_locs="ground", figtitle=''):

    plot_vars_sorted = [[] for _ in plot_order]

    for i, var_type in enumerate(plot_order):
        if var_type == '':
            vars_temp = plot_vars
            continue

        vars_temp = []
        for var in plot_vars:
            print(var + ' starts with ' + var_type +
                  ' :' + str(var.startswith(var_type)))
            if var.startswith(var_type):
                vars_temp.append(var)
                plot_vars.remove(var)

        plot_vars_sorted[i] = vars_temp

    cols = 2
    fig, axs = plt.subplots(2, cols)

    fig.autofmt_xdate()
    for _, ax in np.ndenumerate(axs):
        ax.fmt_xdf = mpl.dates.DateFormatter('%m-%d')
        ax.set_xlim(df.index.min(), df.index.max())

    fig.suptitle(figtitle)

    axs[0, 0].set_title('short wave flux')
    for i, var_type in enumerate(plot_vars_sorted):
        for var in var_type:
            axs[i / cols, i %
                cols].plot(df.index, (df["ground"][var]), label=var, block=False)

    #         axs[0,0]
    # axs[0,0].plot(df.index, (df["ground"]["q_sw_in_diff"]),label="in diff")
    # axs[0,0].plot(df.index, (df["ground"]["q_sw_in_from_els"]),label="in from els")
    # axs[0,0].plot(df.index, (df["ground"]["q_sw_abs"]),label="abs")

    # axs[0,1].set_title('long wave flux')
    # axs[0,1].plot(df.index, (df["ground"]["q_lw_in_from_els"]),label="in from els")
    # axs[0,1].plot(df.index, (df["ground"]["q_lw_in_from_sky"]),label="in form sky")
    # axs[0,1].plot(df.index, (df["ground"]["q_lw_out_rad"]),label="out rad")
    # axs[0,1].plot(df.index, (df["ground"]["q_lw_out_refl"]),label="out refl")
    # # axs[0,1].plot(df.index, (df["ground"]["q_lw_out_els"]))

    # axs[1,0].set_title('temperature')
    # axs[1,0].plot(df.index, (df["ground"]["T"]))

    # axs[1, 1].plot(df.index, (df["ground"]["q_convection"]))
    # axs[1, 1].plot(df.index, (df["ground"]["q_conduction"]))

    axs[0, 0].set_ylim(0, 850)
    axs[0, 1].set_ylim(0, 850)
    axs[1, 0].set_ylim(250, 360)
    # axs[0,0].set_xlabel('date')
    # axs[0,1].set_xlabel('date')
    # axs[0,0].set_ylabel('flux')
    # axs[0,1].set_ylabel('flux')
    # axs[0,0].grid(True)

    plt.legend()
    fig.tight_layout()
    plt.show()


def overview_plot_multi(
        df,
        plot_vars,
        plot_order=['q_sw', 'q_lw', 'T', ''],
        plot_locs=["ground"],
        showfig=True,
        figtitle='',
        savefig=False,
        save_path='./monthplot.png'):

    plot_vars_copy = plot_vars.copy()
    # print('plot_vars_copy',plot_vars_copy)

    plot_vars_sorted = [[] for _ in plot_order]

    for i, var_type in enumerate(plot_order):
        if var_type == '':
            vars_temp = plot_vars_copy
            continue

        vars_temp = []
        for var in plot_vars_copy:
            # print(var+' starts with '+var_type+' :'+str(var.startswith(var_type)))
            if var.startswith(var_type):
                vars_temp.append(var)
                # print('appended and then removed'+var)
                # plot_vars_copy.remove(var)

        plot_vars_sorted[i] = vars_temp

    # print('plot_vars_sorted',plot_vars_sorted)

    ncols = len(plot_order)
    nrows = len(plot_locs)
    fig, axs = plt.subplots(nrows, ncols, figsize=(11.7, 8.27))

    fig.autofmt_xdate()
    for _, ax in np.ndenumerate(axs):
        ax.fmt_xdf = mpl.dates.DateFormatter('%d')
        ax.set_xlim(df.index.min(), df.index.max())

    fig.suptitle(figtitle)

    # axs[0,0].set_title('short wave flux')
    for row, loc in enumerate(plot_locs):

        for col, var_type in enumerate(plot_vars_sorted):
            for var in var_type:
                axs[row, col].plot(df[str(loc)][str(var)],
                                   label=var, block=False)
                axs[row, col].legend()
                axs[row, 0].set_ylim(0, 1000)
                axs[row, 1].set_ylim(0, 1450)
                axs[row, 2].set_ylim(250, 350)

    # axs[0,0].set_ylim(0, 850)
    # axs[0,1].set_ylim(0, 850)
    # axs[1,0].set_ylim(250, 360)
    # axs[0,0].set_xlabel('date')
    # axs[0,1].set_xlabel('date')
    # axs[0,0].set_ylabel('flux')
    # axs[0,1].set_ylabel('flux')
    # axs[0,0].grid(True)

    plt.legend()
    fig.tight_layout()

    if savefig:
        plt.savefig(save_path)

    if showfig:
        plt.show()


def data_over_img(
        img,
        data,
        data_mask=True,
        # data_transparancy=0,
        cmap='coolwarm',
        colorbar_lims=None,
        data_alpha=1):
    """Plot data over an rgb image."""
    img = img.copy()
    
    if (img > 1).any():
        img = img / 255
        
    if img.shape[2] != 4:
        img = np.dstack((img, np.ones(img.shape[:2])))
    # # make img more gray
    # img = (img + mpl.colors.to_rgb('dimgray')) / 2

    cmap = mpl.cm.get_cmap(cmap)

    if colorbar_lims is None:
        if (data < 0).any():
            cmax = np.abs([np.nanmin(data), np.nanmax(data)]).max()
            colorbar_lims = tuple((-cmax, cmax))
        else:
            colorbar_lims = tuple((np.nanmin(data), np.nanmax(data)))

    masked_data = np.ma.masked_invalid(data)
    mask = ~masked_data.mask & data_mask
    
    data_rgb, colorbar = hf.arr_to_rgb(masked_data, 
                                       cmap=cmap, 
                                       colorbar_lims=colorbar_lims)
    
    # pdb.set_trace()
    if isinstance(data_alpha, np.ndarray):
        data_rgba = np.dstack((data_rgb, data_alpha))
        
    elif isinstance(data_alpha, (int, float)):
        data_rgba = np.dstack((data_rgb, data_alpha * np.ones(img.shape[:2]) * mask))
        
    else:
        raise NotImplementedError('data_alpha must be array or float value')
        
    # plt.imshow(data_rgba)
    # plt.show(block=False)
    
    
    # result_rgb = img[..., :3] * img[..., 3][..., np.newaxis] + data_rgba[..., :3] * data_rgba[..., 3][..., np.newaxis]
    result_rgba = img[..., :3] * (1-data_rgba[..., 3][..., np.newaxis]) + data_rgba[..., :3] * data_rgba[..., 3][..., np.newaxis]
    # result_rgba = np.clip(result_rgb, 0, 1)
    return result_rgba, colorbar


def cross_section(data, slice_xyz, idcs_3d, remove_neg_idcs=True, origin='lower', title=None, show=True, colorbar=True, **kwargs):
    """Plots the cross section of a 3-dimensional data set on a 2-dimensional plane.
    
    Parameters
    ----------
    data : ndarray
        1D data set.
    slice_xyz : tuple of slices
        3D slice to take a cross section of `data`.
    idcs_3d : ndarray
        3D indices.
    remove_neg_idcs : bool, optional
        If True, removes the values at the location where the indices in 
        `idcs_3d` are negative from the values to be plotted.
    cmap : str, optional
        Color map used in plotting. The default is 'viridis'.
    title : str, optional
        Title for the plot.
    show : bool, optional
        If True, shows the plot. The default is True.
    colorbar : bool, optional
        If True, adds a color bar to the plot. The default is True.
    **kwargs : Optional
        Additional keyword arguments to be passed to `plt.imshow`.
        
    Raises
    ------
    Exception
        If the resulting array with indices on the plane is not 2-dimensional.
    """
    
    if data.ndim != 1:
        raise Exception(f'Input data set is not a 1D data set but has ndim {data.ndim}')
    
    idcs_on_plane = idcs_3d[slice_xyz].T
    if idcs_on_plane.ndim != 2:
        raise Exception('Resulting array with indices on plane is not 2D.')
    values_on_plane = np.take(data, idcs_on_plane)  
    if remove_neg_idcs:
        values_on_plane[idcs_on_plane<0] = np.nan
    
    plt.imshow(values_on_plane, origin=origin, **kwargs)
    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar()
    if show:
        plt.show()
        
    return values_on_plane


def show_img(
        img,
        plot=False,
        colorbar=None,
        contours=None,
        contours_rgb=(0, 0, 0, 0.7),
        save_path=None,
        figtitle=None,
        show_axis=True,
        ax=None):

    if plot or save_path is not None:

        # plt.close()
        # plt.clf()

        if ax is None:
            fig, ax = plt.subplots()
            
        if contours is not None:
            img = cv2.drawContours(img.copy(), contours, -1, contours_rgb, 1)
            

            
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox = dict(boxstyle ="round", fc ="1", ec="0", ) 
            arrowprops = dict( 
                arrowstyle = "->", 
                facecolor='black',
                connectionstyle = "angle, angleA = 0,  angleB = 90, rad = 10") 
            
            ax.annotate('UHI Measurement area', (x + (w / 2), y + h), (img.shape[0] / 2, img.shape[1] - 5), bbox = bbox, arrowprops = arrowprops)
            # plt.imshow(img)
            # plt.show()
            
            # im = plt.imshow(img)
            # plt.show()
        

        if figtitle is not None:
            ax.set_title(figtitle)

        if isinstance(colorbar, mpl.cm.ScalarMappable):
            plt.colorbar(colorbar, ax=ax, extend='both',
                         fraction=0.0415, pad=0.04)

        if not show_axis:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        im = ax.imshow(img)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if plot:
            plt.show(block=False)
        if not plot:
            plt.close()

def masks2img(
        masks,
        map_colors,
        height2d=None,
        plot=False,
        save_dir=None):

    img_size = list(masks.values())[0].shape + (3,)
    img = np.zeros(img_size)

    for surf_type in masks:
        if not (masks[surf_type]).any():
            continue
        # create boolean mask for where to input rgb data
        mask_repeat = (np.repeat(masks[surf_type][:, :, np.newaxis], 3, 2)
                       .astype(bool))
        try:
            rgb = mpl.colors.to_rgb(map_colors[surf_type])

        except ValueError:
            colors = [mpl.colors.to_rgb(color)
                      for color in map_colors[surf_type]]
            cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap',
                                                                colors,
                                                                N=100)

            norm = mpl.colors.Normalize(vmin=height2d[masks[surf_type]].min(),
                                        vmax=height2d[masks[surf_type]].max())

            rgba = cmap(norm(height2d))
            rgb = rgba[..., :3]

        img = np.where(mask_repeat, rgb, img)

    plt.tight_layout()

    if plot:
        plt.imshow(img)
        plt.show(block=False)

    if save_dir is not None:
        hf.save_image(img, save_dir)

    plt.close()
    plt.clf()

    return img


def overview_bandwiths_multi(df, knmi_data, plot_vars, date_slice,
                             plot_order=['q_sw', 'q_lw', 'T', ''],
                             plot_locs=["ground"],
                             figtitle='',
                             utci_log=None,
                             showfig=True,
                             savefig=False,
                             save_path='./monthplot.png'):

    plot_vars_copy = plot_vars.copy()
    # print('plot_vars_copy',plot_vars_copy)

    plot_vars_sorted = [[] for _ in plot_order]

    for i, var_type in enumerate(plot_order):
        if var_type == '':
            vars_temp = plot_vars_copy
            continue

        vars_temp = []
        for var in plot_vars_copy:
            # print(var+' starts with '+var_type+' :'+str(var.startswith(var_type)))
            if var.startswith(var_type):
                vars_temp.append(var)
                # print('appended and then removed'+var)
                # plot_vars_copy.remove(var)

        plot_vars_sorted[i] = vars_temp

    # print('plot_vars_sorted',plot_vars_sorted)

    ncols = len(plot_order)
    nrows = len(plot_locs)
    fig, axs = plt.subplots(nrows, ncols, figsize=(11.7, 8.27))

    fig.autofmt_xdate()
    for _, ax in np.ndenumerate(axs):
        ax.fmt_xdf = mpl.dates.DateFormatter('%d')
        ax.set_xlim(df.index.min(), df.index.max())

    fig.suptitle(figtitle)
    color = plt.rcParams["axes.prop_cycle"].by_key()['color']

    # axs[0,0].set_title('short wave flux')
    for row, loc in enumerate(plot_locs):

        for col, var_type in enumerate(plot_vars_sorted):
            for i, var in enumerate(var_type):
                # print('i:',var)

                # print('str(loc)][str(var)',str(loc),str(var))
                if not var.startswith('knmi_'):
                    data = df[date_slice][str(loc)][str(var)].astype(float)
                else:
                    data = knmi_data[date_slice][str(var[5:])].astype(float)
                # print(data)
                # dates = np.unique(data.index.date)
                if var.startswith('q'):
                    sum_data = data.groupby(pd.Grouper(
                        freq='D')).agg(lambda x: x.sum())
                    filter = sum_data > 0
                    dates = sum_data.index.date
                    axs[row,
                        col].plot(dates[filter],
                                  sum_data[filter],
                                  label=varname2latex(var),
                                  block=False)
                    axs[row, col].set_ylabel(
                        loc.title() + "\n" + "Average [MJ/d/m2]")

                # if var.startswith('knmi'):
                #     min_data = data[data.groupby(pd.Grouper(freq='D')).agg(lambda x : x.idxmin()).dropna()]
                #     max_data = data[data.groupby(pd.Grouper(freq='D')).agg(lambda x : x.idxmax()).dropna()]
                #     min_data -= 273.15
                #     max_data -= 273.15

                #     if var.endswith('T'):
                #         axs[row,col].set_ylabel(loc.title()+"\n"+"T [K]")

                if var == 'knmi_T' and row == 0:
                    dates = data.index
                    axs[row,
                        col].plot(dates,
                                  data,
                                  color=color[i],
                                  alpha=0.4,
                                  label=varname2latex(var),
                                  block=False)
                    axs[row, col].set_ylabel("Temp [C]")

                if var.endswith('FH') and row == 1:
                    dates = data.index
                    axs[row,
                        col].plot(dates,
                                  data,
                                  color=color[i],
                                  alpha=0.4,
                                  label=varname2latex(var),
                                  block=False)
                    axs[row, col].set_ylabel("Wind speed [m/s]")

                if var == 'knmi_Q' and row == 2:
                    dates = data.index
                    axs[row,
                        col].plot(dates,
                                  data,
                                  color=color[i],
                                  alpha=0.4,
                                  label=varname2latex(var),
                                  block=False)
                    axs[row, col].set_ylabel("Global solar radiation [W/m2]")

                if var == 'T':
                    min_data = data[data.groupby(pd.Grouper(freq='D')).agg(
                        lambda x: x.idxmin()).dropna()]
                    max_data = data[data.groupby(pd.Grouper(freq='D')).agg(
                        lambda x: x.idxmax()).dropna()]
                    print('min_data', min_data)
                    print('max_data', max_data)
                    min_data -= 273.15
                    max_data -= 273.15
                    dates = min_data.index.date
                    axs[row, col].fill_between(
                        dates, min_data, max_data, alpha=0.2, color=color[i])
                    axs[row,
                        col].plot(dates,
                                  min_data,
                                  color=color[i],
                                  alpha=0.4,
                                  label=varname2latex(var),
                                  block=False)
                    axs[row, col].plot(
                        dates, max_data, color=color[i], alpha=0.4, block=False)

                    # min_data = data[data.groupby(pd.Grouper(freq='D')).agg(lambda x : x.idxmin()).dropna()]
                    # max_data = data[data.groupby(pd.Grouper(freq='D')).agg(lambda x : x.idxmax()).dropna()]

                    # axs[row,col].plot(data, label='T$_{\mathrm{surf}}$')
                    # axs[row,col].plot(T_amb, label = "T$_{\mathrm{amb}}$")

                    axs[row, col].set_ylabel(loc.title() + "\n" + "T [K]")
                axs[row, col].legend()
                # axs[row,0].set_ylim(0, 1000)
                # axs[row,1].set_ylim(0, 1450)
                # axs[row,2].set_ylim(250, 350)

        axs[row, 0].set_ylim(0, 7000)
        axs[row, 1].set_ylim(0, 26000)

    for i, var in enumerate(utci_log.columns):
        data = utci_log[str(var)].astype(float)

        min_data = data[data.groupby(pd.Grouper(freq='D')).agg(
            lambda x: x.idxmin()).dropna()]
        max_data = data[data.groupby(pd.Grouper(freq='D')).agg(
            lambda x: x.idxmax()).dropna()]
        dates = min_data.index.date
        axs[i, 3].fill_between(dates, min_data, max_data,
                               alpha=0.2, color=color[i + 1])
        axs[i, 3].plot(dates, min_data, color=color[i + 1], alpha=0.4,
                       label=varname2latex('UTCI_' + var), block=False)
        axs[i, 3].plot(dates, max_data, color=color[i + 1],
                       alpha=0.4, block=False)
        axs[i, 3].legend()

    # axs[1,0].set_ylim(250, 360)
    # axs[0,0].set_xlabel('date')
    # axs[0,1].set_xlabel('date')
    # axs[0,0].set_ylabel('flux')
    # axs[0,1].set_ylabel('flux')
    # axs[0,0].grid(True)

    plt.legend()
    fig.tight_layout()

    if savefig:
        plt.savefig(save_path)

    if showfig:
        plt.show()


def varname2latex(var):
    var_split = var.split('_')
    if 'from' in var_split:
        var_split.remove('from')
    return var_split[0] + '$_{' + ','.join(var_split[1:]) + '}$'


def data2rgba(
        data,
        cmap=plt.get_cmap('viridis'),
        colorbar_lims=None,
        mask=None,
        std_color='whitesmoke'):

    # Start by setting the facecolors to a standard value
    img = np.tile(mpl.colors.to_rgba(std_color), (np.shape(data) + (1,)))

    # color = plot_data[item]['color']
    if mask is None:
        mask = np.ones(np.shape(data), dtype=bool)

    if colorbar_lims is not None:
        data = np.clip(
            data,
            a_min=colorbar_lims[0],
            a_max=colorbar_lims[1])
    else:
        colorbar_lims = (data.min(), data.max())

    mask3d = np.repeat(mask[..., np.newaxis], 4, axis=2)

    try:
        # small workaround to hide a warning.
        # with warnings.catch_warnings():
        #     warnings.simplefilter(action='ignore', category=FutureWarning)
        cmap = plt.get_cmap(cmap)  # "Greens"
    except ValueError:  # "["color1","color2"] or "green"
        if isinstance(cmap, list):
            colors = [mpl.colors.to_rgb(color) for color in cmap]
            # cmap_name = '2'.join(cmap)
            cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap',
                                                                colors,
                                                                N=100)
        else:

            img = np.where(mask3d, list(
                mpl.colors.to_rgba_array(cmap)), img)

    # return None

    if 'cmap' in locals():
        # color_norm = (data-colorbar_lims[0])/(colorbar_lims[1]-colorbar_lims[0])
        norm = mpl.colors.Normalize(vmin=colorbar_lims[0],
                                    vmax=colorbar_lims[1])
        img = np.where(mask3d, cmap(norm(data)), img)

        del cmap

    return img, colorbar_lims
