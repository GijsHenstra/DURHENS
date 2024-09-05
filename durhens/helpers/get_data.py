#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:48:04 2021

@author: gijshenstra
"""

import pdb

import rasterio
import os
import glob
import numpy as np
import rasterstats as rs
import geopandas as gpd
import cv2
import scipy
import math
import shapely
import requests
import itertools
import datetime
import time


import knmi

from helpers import download
from helpers import fill_missing
from helpers import boundingbox as bb
from helpers import offline_data
from helpers import help_functions as hf

import pandas as pd
import warnings
import rasterio.mask
import matplotlib.pyplot as plt
import osmnx as ox

from plot import plot2d

from collections import Counter
from geopy.geocoders import Nominatim
from io import BytesIO
from PIL import Image
from affine import Affine
from shapely import affinity
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import show
from owslib.wcs import WebCoverageService
from shapely.geometry import Point
from requests import Request

from owslib.wms import WebMapService

from helpers import download
from helpers import crs

warnings.simplefilter("ignore", UserWarning)

# %% Supportive functions

import threading

def wait_for_user_input_or_timeout():
    input_thread.result = input("Press Enter to continue, or wait for 5 seconds: ")

def _gdf_to_array(gdf, load_res, bbox, arr_shape):
    """
    Convert a GeoDataframe to a Numpy array.

    Parameters
    ----------
    gdf : GeoDataFrame
        The input GeoDataFrame to be converted.
    load_res : float
        The resolution for the output array.
    bbox : tuple
        A tuple of (min_x, min_y, max_x, max_y) specifying the bounding box of the output array.
    arr_shape : tuple
        A tuple of (nrows, ncols) specifying the shape of the output array.

    Returns
    -------
    arr : np.ndarray
        The output rasterized Numpy array.
    """
    gdf = gdf.dropna(subset=['geometry'])
    transform = Affine(load_res, 0.0, bbox[0], 0.0, -load_res, bbox[3])

    arr = rasterio.features.rasterize(
        [(x.geometry, 1) for i, x in gdf.iterrows()], out_shape=arr_shape,
        transform=transform, fill=0, all_touched=True, dtype=rasterio.uint8)
    return arr


def _arr_to_rgb(arr, cm=plt.cm.viridis):
    """
    Convert a Numpy array to a RGB image.

    Parameters
    ----------
    arr : numpy array
        The input Numpy array to be converted.
    cm : colormap, optional
        The colormap used to map the values of the array to RGB values. Default is `plt.cm.viridis`.

    Returns
    -------
    mapped_data : numpy array
        The output RGB image represented as a Numpy array.
    """

    normalized_data = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    mapped_data = cm(normalized_data, bytes=True)[..., :3]

    return mapped_data


def dominant_angle(arr):
    """
    Compute the dominant angle of an array using the Hough lines method.
    
    Parameters
    ----------
    arr : numpy array
        The input Numpy array.
    
    Returns
    -------
    dominant_angle : float
        The dominant angle in degrees.
    """
    arr_rgb = _arr_to_rgb(arr)
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_BGR2GRAY)

    canimg = cv2.Canny(gray, 50, 200)
    rho_step = .1
    rho = rho_step
    lines = None
    
    # while True:
    #     rho += rho_step
    #     add_lines = cv2.HoughLines(canimg, rho, np.pi / (180 / rho_step), 100)
        
        
    #     if add_lines is not None: 
    #         if lines is None:
    #             lines = add_lines
    #         # print('add_lines',add_lines)
    #         lines = np.append(lines, add_lines, axis=0)
        
    #     if rho > 90:
    #         break
    
    while lines is None:
        rho += rho_step
        lines = cv2.HoughLines(canimg, rho, np.pi / (180 / rho_step), 100)
        if rho > 360:
            break

    if lines is None:
        dominant_angle = float(input('Manually enter angle to rotate map: '))
    else:
        lines_lst = list(lines[:, :, 1].flatten())

        # lines_counter = Counter(lines_lst)
        # dominant_angle_rad = lines_counter.most_common(3)[2][0]


        # Sort lines on most-occuring rotation angle
        dominant_angle_rad = sorted(lines_lst,
                                    key=lines_lst.count,
                                    reverse=True)[0]
        dominant_angle = np.rad2deg(dominant_angle_rad) % 90

        if dominant_angle > 45:
            dominant_angle = dominant_angle - 90

        print('\n To align the map with horizontal and vertical axis, rotate it over '
              + str(round(dominant_angle, 3)) + ' degrees \n')

    return round(dominant_angle, 2)


def rotate_crop(arr_zoomout, rotation, gridsize, res):
    """
    Rotate an array over an angle and then crop it to the desired size.
    
    Parameters
    ----------
    arr_zoomout : numpy array or list of numpy arrays
        The input array or list of arrays to be rotated and cropped.
    rotation : float
        The angle in degrees to rotate the array.
    gridsize : float
        The size of the output array in physical units.
    res : float
        The resolution of the array in physical units per pixel.
    
    Returns
    -------
    rst : numpy array or tuple of numpy arrays
        The rotated and cropped array or tuple of arrays.
   """

    if not rotation:
        rotation = 0

    if not isinstance(arr_zoomout, list):
        arr_zoomout = [arr_zoomout]

    rst = []
    for data in arr_zoomout:

        if isinstance(data, np.ndarray):

            if rotation != 0:
                data_rot = scipy.ndimage.rotate(data, rotation, axes=(-2, -1),
                                                reshape=False)
            else:
                data_rot = data

            data_crop = hf.crop_center(
                data_rot, gridsize / res, gridsize / res)
            if data.dtype == bool:
                data_crop = np.round(data_crop)

            rst.append(data_crop.astype(data.dtype))

        if isinstance(data, dict):
            dct = {}
            for item in data:
                data_rot = scipy.ndimage.rotate(data[item].astype(float),
                                                rotation,
                                                reshape=False)
                data_crop = hf.crop_center(data_rot,
                                           gridsize / res,
                                           gridsize / res)
                dct.update({str(item): np.round(data_crop)
                            .astype(data_crop.dtype)})
            rst.append(dct)

    if len(arr_zoomout) == 1:
        return rst[0]

    else:
        return tuple(rst)


def crop_DatasetReader(src, shapes, folder, gridsize=None, rotation=None,
                       crop_file="crop.tif", plot=False, bbox=None, fill=True,
                       nodata=True):
    """
    Crop a raster dataset based on a given set of shapes or bounding box.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        The source dataset to be cropped.
    shapes : Union[gpd.geoseries.GeoSeries, Tuple[float, float, float, float], shapely.geometry.polygon.Polygon]
        The shapes or bounding box used to crop the dataset.
    folder : str
        The folder in which to save the cropped dataset.
    gridsize : Optional[int], optional
        The size of the grid to use for rotation, by default None
    rotation : Optional[float], optional
        The angle of rotation, by default None
    crop_file : str, optional
        The name of the file to save the cropped dataset as, by default "crop.tif"
    plot : bool, optional
        Whether to display the cropped dataset, by default False
    bbox : Optional[Tuple[float, float, float, float]], optional
        The bounding box to use for cropping, by default None
    fill : bool, optional
        Whether to fill missing data, by default True
    nodata : bool, optional
        Whether to use nodata, by default True

    Returns
    -------
    src_crop : rasterio.io.DatasetReader
        The cropped dataset.
    """

    nodataval = 0

    if isinstance(shapes, gpd.geoseries.GeoSeries):
        pass

    elif isinstance(shapes, (tuple, list)):
        bbox = shapes
        if not isinstance(
            shapes[0],
            (shapely.geometry.multipolygon.MultiPolygon,
             shapely.geometry.polygon.Polygon)):
            bbox = shapes
            shapes = [shapely.geometry.box(*shapes)]

    elif isinstance(shapes, (shapely.geometry.multipolygon.MultiPolygon,
                             shapely.geometry.polygon.Polygon)):
        shapes = [shapes]

    else:
        raise ValueError(f'Invalid boundingbox/shapes inserted: {shapes}')
    if rotation:
        shapes = [affinity.rotate(shape, -rotation, 'center')
                  for shape in shapes]

    out_image, out_trans = rasterio.mask.mask(src, shapes, crop=True)

    out_meta = src.meta

    if bbox:
        out_trans = Affine(out_trans.a, out_trans.b, bbox[0],
                           out_trans.d, out_trans.e, bbox[3])

    if rotation:
        if not gridsize:
            ValueError('Wrong value for gridsize')

        if nodata is False:
            pass
        else:
            nodata_val = out_image.min()
            nanmask = (out_image == nodata_val)[0]
            out_image[:, nanmask] = nodataval


        if fill:
            out_image = fill_missing.array(out_image, sigma=3.0)

        if plot:
            plt.imshow(np.moveaxis(out_image, 0, -1))
            plt.show()

        out_image = np.stack([rotate_crop(out_image[i],
                                          rotation,
                                          gridsize,
                                          src.res[0]) for i in range(len(out_image))],
                             axis=0)

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_trans,
                     "nodata": nodataval})

    with rasterio.open(f'{folder}/{crop_file}', "w", **out_meta) as file:
        file.write(out_image)

    src_crop = rasterio.open(f'{folder}/{crop_file}', driver="GTiff")
    if plot:
        rasterio.plot.show(src_crop)

    return src_crop


# %% Get weather data

def knmi_weather(closest_stn, t_step, start_date, end_date,  simulate_averages, timezone, average_years, save_path):
    
    if simulate_averages:
        knmi_tmp = pd.read_csv('data/uurgeg_344_2001-2020_raw.txt',
                               skipinitialspace=True)
        times = pd.to_datetime((knmi_tmp["YYYYMMDD"].astype(
            str) + '-' + (knmi_tmp["HH"] - 1).astype(str)).astype(str), format='%Y%m%d-%H')
        knmi_tmp['Date'] = times
        knmi_tmp = knmi_tmp.set_index('Date')
        del knmi_tmp['HH']
        del knmi_tmp['YYYYMMDD']

    if not simulate_averages:
        
        start_date_load = start_date - datetime.timedelta(days=1)
        end_date_load = start_date + datetime.timedelta(days=1, minutes=-1)

        while True:
            try:
                knmi_tmp = knmi.get_hour_data_dataframe(
                    stations=[closest_stn], start=start_date_load.strftime('%Y%m%d%H'), end=end_date_load.strftime('%Y%m%d%H'))
                break
            except KeyError:
                print('Downloaded KNMI weather dataframe is empty. Retrying after 5 seconds...')
                
                time.sleep(5)
                    
            
        knmi_tmp.index.names = ['Date']
        knmi_tmp.index = pd.to_datetime(knmi_tmp.index)

    # dates = np.unique(np.array(
    #     knmi_tmp[start_date:end_date].index.strftime('%Y-%m-%d')).astype(str))[:-1]

    # correct time from UTC
    knmi_tmp.index = knmi_tmp.index.tz_localize('UTC')
    knmi_tmp.index = knmi_tmp.index.tz_convert(timezone)
    knmi_tmp.index = knmi_tmp.index.tz_localize(None)
    
    

    knmi_tmp = knmi_tmp.apply(pd.to_numeric,
                              errors='coerce')  # strings to floats

    knmi_tmp.drop(knmi_tmp.tail(1).index,
                  inplace=True)  # drop last n rows

    cols_to_divide = [
        'FH', 'FF', 'FX', 'T10N', 'T', 'TD', 'SQ', 'DR', 'RH', 'P']
    knmi_tmp[cols_to_divide] = knmi_tmp[cols_to_divide].astype(
        float) / 10

    knmi_tmp["Q"] *= 100 * 100 / 3600  # irradiance from J/h/cm to W/m2

    knmi_tmp = knmi_tmp[knmi_tmp.index.notnull()]

    # drop duplicate times at switch to summertime/wintertime
    knmi_tmp = knmi_tmp[~knmi_tmp.index.duplicated(keep='first')]

    knmi_tmp = (knmi_tmp.asfreq(freq=str(t_step) + 'S')
                .interpolate(method='linear'))
    
    

    hourly_averages = ['DD', 'FH', 'FF', 'SQ', 'Q', 'DR', 'RH', 'WW']
    steps_per_30min = int((60 * 30) / t_step)
    knmi_tmp[hourly_averages] = knmi_tmp[hourly_averages].shift(
        steps_per_30min)

    # throw away the unused leading and trailing time rows
    knmi_tmp = knmi_tmp.loc[start_date.strftime('%Y-%m-%d %H:%M'): (end_date + datetime.timedelta(days=1) - datetime.timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M')]

    if simulate_averages:
        knmi_tmp = knmi_tmp.loc[slice(str(average_years),
                                      str(average_years))]

        # define how many values are in the top 2 percentage
        top_perc = 2
        n = int(((knmi_tmp.index.month == 1) *
                (knmi_tmp.index.hour == 1)).sum() * top_perc / 100)

        for i, (func_type, day) in enumerate({'min': 5,
                                              'mean': 15,
                                              'max': 25}.items()):
            if func_type == 'min':
                knmi_tmp_grouped = knmi_tmp.groupby([
                    knmi_tmp.index.month,
                    knmi_tmp.index.hour]).apply(pd.DataFrame.nsmallest,
                                                n=n,
                                                columns='T')
                knmi_tmp_grouped.index.rename(
                    ['Month', 'Hour', 'Date'], inplace=True)
                knmi_tmp_grouped = knmi_tmp_grouped.groupby(
                    ['Month', 'Hour']).mean()
            if func_type == 'mean':
                knmi_tmp_grouped = knmi_tmp.groupby([
                    knmi_tmp.index.month,
                    knmi_tmp.index.hour]).mean()

            if func_type == 'max':
                knmi_tmp_grouped = knmi_tmp.groupby([
                    knmi_tmp.index.month,
                    knmi_tmp.index.hour]).apply(pd.DataFrame.nlargest,
                                                n=n,
                                                columns='T')
                knmi_tmp_grouped.index.rename(
                    ['Month', 'Hour', 'Date'], inplace=True)
                knmi_tmp_grouped = knmi_tmp_grouped.groupby(
                    ['Month', 'Hour']).mean()

            date_index = pd.DataFrame(
                {
                    'year': average_years[0],
                    'month': knmi_tmp_grouped.index.get_level_values(0),
                    'day': day,
                    'hour': knmi_tmp_grouped.index.get_level_values(1)})

            knmi_tmp_grouped = knmi_tmp_grouped.set_index(
                pd.to_datetime(date_index).rename('Date'))

            knmi_tmp_grouped = pd.concat(
                [
                    knmi_tmp_grouped.set_index(
                        knmi_tmp_grouped.index -
                        pd.DateOffset(
                            days=1)),
                    knmi_tmp_grouped,
                    knmi_tmp_grouped.set_index(
                        knmi_tmp_grouped.index +
                        pd.DateOffset(
                            days=1))])

            # smooth by rolling window
            knmi_tmp_grouped = knmi_tmp_grouped.rolling(
                7, center=True, win_type='triang').mean()

            knmi_tmp_grouped = knmi_tmp_grouped[knmi_tmp_grouped.index.day == day]

            if i == 0:
                knmi_data = knmi_tmp_grouped
            else:
                knmi_data = pd.concat([knmi_data, knmi_tmp_grouped])

        # add the 14th, 16th, 18th day of the month by copying the day before
        # knmi_data = pd.concat([knmi_data,
        # knmi_data.set_index(knmi_data.index + pd.DateOffset(CONFIG['DAYS']=1))])

        knmi_data = knmi_data.sort_index()
        
    else:
        knmi_data = knmi_tmp

    
    
    return knmi_data

# %% Get datasets for map

def address(address,
            crs_in="EPSG:4326", crs_out="EPSG:28992", xy_dtype=int):
    """
    Get the coordinates of an address in a specified coordinate system (CRS).

    Parameters
    ----------
    address : str
        The address to look up. Can be a street name or a string of two comma-
        separated coordinates. 
        (e.g. "TU Delft Library" or "52.00270834407012, 4.375309401127023")
    crs_in : str, optional
        The input coordinates system, by default "EPSG:4326"
    crs_out : str, optional
        The output coordinates system, by default "EPSG:28992"
    xy_dtype : type, optional
        The data type of the output coordinates, by default int

    Returns
    -------
    Tuple[int, int, str, Location]
    x, y : int
        Coordinates of the address in the specified coordinate system.
    address : str
        String of the address.
    location : Location
        Location object.
    """

    geolocator = Nominatim(user_agent="specify_random_user_agent")
    # print('Looking for %s' % address)
    location = geolocator.geocode(address, addressdetails=True)
    while location is None:
        raise ValueError(
            f'No location found for "{address}", enter valid location')

    # if "address" is a set of coordinates
    if not address.lower().islower():
        latitude = float(address.split(',')[0])
        longitude = float(address.split(',')[1])

    # if "address" is a streetname
    else:
        latitude = location.latitude
        longitude = location.longitude

    x, y = map(xy_dtype, crs.transform([longitude, latitude],
                                       crs_in=crs_in, crs_out=crs_out))

    return x, y, address, location


def wijkenbuurten(bbox, layer='cbs_buurten_2020', plot_col=None):
    """
    Retrieve neighborhood data from nationaalgeoregister within a bounding box.

    Parameters
    ----------
    bbox : List[float]
        The bounding box coordinates in the form [minx,miny,maxx,maxy]
    layer : str, optional
        The layer to retrieve data from, by default 'cbs_buurten_2020'
    plot_col : Optional[str], optional
        Column to use for plotting, by default None

    Returns
    -------
    gdf : gpd.GeoDataFrame
        The retrieved data as a GeoDataFrame.
    """

    gridsize = (bbox[2] - bbox[0])
    border_thickness = int(math.ceil(gridsize * (math.sqrt(2) - 1)))
    bbox_zoomout = [bbox[0] - border_thickness,
                    bbox[1] - border_thickness,
                    bbox[2] + border_thickness,
                    bbox[3] + border_thickness]

    url = 'https://geodata.nationaalgeoregister.nl/wijkenbuurten2020/wfs'

    # string of bbox for the request url
    bbox_str = ','.join(map(str, bbox_zoomout))

    # Maximum lines returned is 1000, so if more that 1000 are asked the
    # request is sent multiple times.
    count = 1000

    start_index = 0
    gdf = gpd.GeoDataFrame()  # start with empty gdf
    while True:

        params = dict(service='WFS',
                      version="2.0.0",
                      request='GetFeature',
                      typeName=layer,
                      bbox=bbox_str,
                      # outputFormat='application/gml+xml; version=3.2',
                      srsname='urn:ogc:def:crs:EPSG::28992',
                      outputFormat='json',
                      )
        # Parse the URL with parameters
        q = Request('GET', url, params=params).prepare().url

        # Read data from URL
        gdf_snippet = gpd.read_file(q)

        gdf = gdf.append(gdf_snippet)

        if len(gdf_snippet) < count - 1:
            break
        start_index += count

    if plot_col:
        gdf.plot(column=plot_col, legend=True)

    return gdf


def bag(bbox, load_res, save_dir='./',
        save_tif=True, rotation=None, arr_shape=None,
        force_download=False):
    """
    Load BAG data by loading 'pand' and 'verblijfsobject' data from BAG.

    The bag function is used to load data from the Basisregistratie Adressen en
    Gebouwen (BAG) dataset. The data is retrieved from BAG's Web Feature 
    Service (WFS) and includes information on buildings ('pand') and dwelling 
    units ('verblijfsobject'). The function takes in a bounding box (bbox), a 
    resolution to use for loading data (load_res), and several optional 
    parameters such as a directory to save the data (save_dir), whether to save 
    the data as a TIFF file (save_tif), a rotation angle to apply to the data 
    (rotation), the shape of the data array (arr_shape), and a flag to force 
    the data to be downloaded even if it already exists (force_download). The 
    function returns a GeoDataFrame containing the BAG data and the rotation 
    angle used.

    Parameters
    ----------
    bbox : list
        List of 4 values representing the bounding box of the area to be loaded.
    load_res : float
        Resolution of the BAG data to be loaded.
    save_dir : str, default './'
        Directory where the BAG data should be saved.
    save_tif : bool, default True
        If True, the BAG data will be saved as a TIF file.
    rotation : float, default None
        Rotation angle for the BAG data in degrees.
    arr_shape : tuple, default None
        Shape of the output array.
    force_download : bool, default False
        If True, the BAG data will be downloaded even if it has already been saved.
    
    Returns
    -------
    bag_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the BAG data.
    rotation : float
        Rotation angle used for the BAG data.

    """
    import geopandas as gpd

    save_path = save_dir / 'bag_gdf'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = save_path / 'orig'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path_shp = save_path / 'data.shp'

    try:
        if force_download:
            raise Exception
        bag_gdf = gpd.read_file(save_path_shp).set_index('index')

        print('Succesfully read BAG: pand & verblijfsobject')
    except Exception:
        url = 'https://service.pdok.nl/lv/bag/wfs/v2_0'

        # --- load data from bag:pand ---
        drop_cols = ['gml_id', 'gid', 'status', 'oppervlakte_min',
                     'oppervlakte_max', 'rdf_seealso']
        pand_gdf = download.wfs(
            bbox,
            url,
            typeName='pand',
            zoomout=True,
            index='identificatie',
            drop_cols=drop_cols,
            sortby='bag:identificatie')

        # --- load data from bag:verblijfsobject ---
        drop_cols = ['gml_id', 'gid', 'oppervlakte', 'status', 'rdf_seealso',
                     'identificatie', 'pandstatus', 'geometry',
                     'gebruiksdoel']
        verblijfsobj_gdf = download.wfs(
            bbox,
            url,
            typeName='verblijfsobject',
            zoomout=True,
            index='pandidentificatie',
            drop_cols=drop_cols,
            sortby='verblijfsobject:identificatie')

        # --- join datasets ---

        bag_gdf = pand_gdf.join(verblijfsobj_gdf, how='outer',
                                lsuffix='', rsuffix='_verblijfsobject')
        bag_gdf = bag_gdf.reindex(sorted(bag_gdf.columns), axis=1)

        # restore coordinate system information
        # bag_gdf.crs = gpd.read_file(q).crs

        if bag_gdf.empty:
            print('Empty BAG returned')
            rotation = 0
            return bag_gdf, rotation  # 01/02/22 "return None"
        bag_gdf.plot()
        plt.show(block=False)

        bag_gdf.reset_index().to_file(save_path_shp)

    if rotation is not None:
        if isinstance(rotation, bool):
            if not isinstance(rotation, (str, float)):
                if any([v is None for v in [arr_shape]]):
                    raise NameError('Arr_shape missing')

                bag_arr = _gdf_to_array(bag_gdf, load_res=load_res,
                                     bbox=bbox, arr_shape=arr_shape)

                rotation = dominant_angle(bag_arr)

        if isinstance(rotation, (str, float)):
            origin = tuple(list(np.reshape(bbox, (2, 2)).mean(0)))

            # perform rotation
            bag_gdf['geometry'] = bag_gdf['geometry'].rotate(rotation, origin)

    bag_gdf[bag_gdf.postcode.notna()]

    return bag_gdf, rotation


def ahn_sheets(
        bbox,
        url,
        identifier,
        folder,
        save_name,
        fn_prefix,
        fn_extension='.tif',
        download_loc='data/maps/AHN/',
        zoomout=True,
        fill=False,
        fill_max_sigma=25):
    """
    Parameters
    ----------


    Returns
    -------"""

    bbox_orig = bbox

    gridsize = bbox[2] - bbox[0]

    if zoomout:
        bbox = bb.zoomout(bbox)

    try:
        ahn_dsr = rasterio.open(f'{folder}/{save_name}.tif')
        if not bb.equals(ahn_dsr, bbox):
            raise ValueError('Bounds not equal: ', ahn_dsr.bounds, bbox)
    except BaseException:
        mapindex_table = pd.read_csv('data/bladindex_EPSG28992.txt',
                                     index_col='bladnr')

        filenames = bb.to_mapindex(bbox, mapindex_table=mapindex_table)

        datasetreaders = []
        for filename in filenames:

            try:
                if fill:
                    fn = f'{fn_prefix}{filename}_filled{fn_extension}'
                if not fill:
                    fn = f'{fn_prefix}{filename}{fn_extension}'

                print(f'Trying rasterio.open({download_loc}/{fn})')

                ahn_dsr = rasterio.open(f'{download_loc}/{fn}')
                print('Loaded ahn_dsr')
            except BaseException:
                print('Failed to load ahn_dsr')
                fn = f'{fn_prefix}{filename}{fn_extension}'

                ahn_dsr = download.zip_url(
                    fn=fn, base_url=f'{url}', download_loc='data/maps/AHN/')
                if fill:
                    ahn_dsr = fill_missing.geotiff(
                        ahn_dsr, max_sigma=fill_max_sigma)
                    # os.remove(f'{download_loc}/{fn}')

            datasetreaders.append(ahn_dsr)

        if len(datasetreaders) > 1:
            ahn_arr, out_trans = rasterio.merge.merge(
                datasetreaders, bounds=bbox, method='last')

        else:
            ahn_arr, out_trans = rasterio.mask.mask(
                datasetreaders[0], shapes=[
                    shapely.geometry.box(
                        *bbox)], crop=True)

        res = ahn_dsr.res[0]
        out_meta = datasetreaders[0].meta
        out_meta.update({"driver": "GTiff",
                         "height": (bbox[3] - bbox[1]) / res,
                         "width": (bbox[2] - bbox[0]) / res,
                         "transform": out_trans,
                         "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                         }
                        )
        with rasterio.open(f'{folder}/{save_name}.tif', "w", **out_meta) as file:
            file.write(ahn_arr)

        ahn_dsr = rasterio.open(f'{folder}/{save_name}.tif')

    # if rotation:
    #     if fill:
    #         crop_file = f'{save_name}_filled_crop.tif'
    #     if not fill:
    #         crop_file = f'{save_name}_crop.tif'
    #     ahn_dsr = crop_DatasetReader(ahn_dsr, shapes=bbox_orig, gridsize=gridsize,
    #                                  folder=folder,  rotation=rotation,
    #                                  crop_file=crop_file, bbox=bbox_orig, fill=fill,
    #                                  nodata=True)

    return ahn_dsr


def elevation(
        bbox,
        save_dir,
        fn='chm.tif',
        meta=None,
        zoomout=False,
        ahn_version=3,
        force_download=False,
        method='WCS',
        **kwargs):
    """
    Download elevation data from the Algemene Hoogtebestand Nederland (AHN) datasets.
    
    This function allows the user to download and retrieve elevation data from 
    either the AHN2 or AHN3 datasets. The data can be downloaded via WCS or by 
    scraping the sheets provided by the AHN. 
    
    It returns the digital surface model (DSM) and digital terrain model (DTM) 
    in the form of GeoTIFF datasets and their associated metadata. The function 
    can also perform optional tasks such as zooming out the data and filling in 
    missing values.

    Parameters
    ----------
    bbox : list of 4 int
        bounding box of the area of interest, in the format 
        [xmin, ymin, xmax, ymax]
    save_dir : str
        directory to save the downloaded data
    fn : str, optional
        name of the file to save the data to, defaults to 'chm.tif'
    meta : dict, optional
        metadata to include in the saved tiff file
    zoomout : bool, optional
        whether to zoom out the data to fit the requested bbox, defaults to False
    ahn_version : int, optional
        version of the AHN dataset to download, options are 2 or 3, defaults to 3
    force_download : bool, optional
        whether to download the data even if it already exists in the save_dir, 
        defaults to False
    method : str, optional
        method to download the data, options are 'WCS' or 'sheets', defaults 
        to 'WCS'
    **kwargs : additional keyword arguments
        additional keyword arguments to pass to the download functions

    Returns
    -------
    numpy array
        the elevation data, with missing values filled in
    numpy array
        a boolean mask of the missing values
    """

    fp = f'{save_dir}/ahn{ahn_version}_{fn}'

    if method == 'WCS':
        if ahn_version == 2:
            ahn_url = kwargs.get('ahn_url','https://geodata.nationaalgeoregister.nl/ahn2/wcs?SERVICE=WCS') 
            identifier_dsm = kwargs.get('identifier_dsm', 'ahn2_05m_ruw')
            identifier_dtm = kwargs.get('identifier_dtm', 'ahn2_05m_non')
            download_format = 'GEOTIFF_FLOAT32'
            raise ValueError('AHN2 is taken offline for WCS distribution. See: https://geoforum.nl/t/datasets-ahn1-en-ahn-2-bij-pdok-uit-productie/6624')

        if ahn_version == 3:
            ahn_url = kwargs.get('ahn_url', 'https://geodata.nationaalgeoregister.nl/ahn3/wcs?SERVICE=WCS')
            identifier_dsm = kwargs.get('identifier_dsm', 'ahn3_05m_dsm')
            identifier_dtm = kwargs.get('identifier_dtm', 'ahn3_05m_dtm')
            download_format = 'GEOTIFF_FLOAT32'
                
        if ahn_version == 4:
            ahn_url = kwargs.get('ahn_url', 'https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS')
            identifier_dsm = kwargs.get('identifier_dsm', 'dsm_05m')
            identifier_dtm = kwargs.get('identifier_dtm', 'dtm_05m')
            download_format = 'image/tiff'
                
        dsm_dsr = download.wcs(
            url=ahn_url,
            identifier=identifier_dsm,
            bbox=bbox,
            save_dir=save_dir,
            zoomout=zoomout,
            download_format=download_format,
            force_download=force_download)[0]

        dtm_dsr = download.wcs(
            url=ahn_url,
            identifier=identifier_dtm,
            bbox=bbox,
            save_dir=save_dir,
            zoomout=zoomout,
            download_format=download_format,
            force_download=force_download)[0]

    elif method == 'sheets':
        if ahn_version == 2:
            if 'ahn_url' in kwargs:
                ahn_url = kwargs['ahn_url']
            else:
                ahn_url = 'https://ns_hwh.fundaments.nl/hwh-ahn/AHN2/'

            if 'identifier_dsm' in kwargs:
                identifier_dsm = kwargs['identifier_dsm']
            else:
                identifier_dsm = 'ahn2_05m_dsm'

            if 'identifier_dtm' in kwargs:
                identifier_dtm = kwargs['identifier_dtm']
            else:
                identifier_dtm = 'ahn2_05m_dtm'

            dsm_dsr = ahn_sheets(
                bbox,
                url=f'{ahn_url}DSM_50cm',
                identifier=identifier_dsm,
                zoomout=zoomout,
                folder=save_dir,
                save_name=identifier_dsm,
                fn_prefix='r',
                fill=False)
            dtm_dsr = ahn_sheets(
                bbox,
                url=f'{ahn_url}DTM_50cm',
                identifier=identifier_dsm,
                zoomout=zoomout,
                folder=save_dir,
                save_name=identifier_dtm,
                fn_prefix='i',
                fill=False)

    else:
        print('enter valid "method"')

    dsm_fill = fill_missing.geotiff(
        dsm_dsr, save_path_filled=f'{save_dir}/{identifier_dsm}_filled.tif')
    dtm_fill = fill_missing.geotiff(
        dtm_dsr, save_path_filled=f'{save_dir}/{identifier_dtm}_filled.tif')

    try:
        chm_arr = rasterio.open(fp)

    except BaseException:

        chm_arr = dsm_fill.read(1) - dtm_fill.read(1)

        if not meta:
            meta = dsm_dsr.meta

        with rasterio.open(fp, 'w', **meta) as dst:
            dst.write(chm_arr[np.newaxis, ...])

        chm_arr = rasterio.open(fp)

    return chm_arr, dsm_dsr, dtm_dsr, dsm_fill


def roads(
        save_dir,
        bbox,
        crs_init,
        gridsize,
        subdir='roads_gdf',
        rotation=None,
        plot=False,
        force_download=False,
        service='wms',
        load_res=0.5,
        load_res_raw=0.25):
    """
    Download road data from RDW (Dienst Wegverkeer) database or from local data.
    
    This function loads road data from the RBW database or from local data, 
    and processes it to be used in further analysis. The function takes in 
    various parameters such as the save directory, bounding box, grid size, 
    rotation, and more. It uses either the WFS or WMS service to download data, 
    and uses the geopandas library to handle the data. The processed data is 
    then returned in the form of a GeoDataFrame.
    
    Parameters
    ----------
    save_dir : str
        Directory where the data should be saved, if downloaded.
    bbox : list
        Bounding box of the area of interest in the format [xmin, ymin, xmax, ymax].
    crs_init : int
        EPSG code of the CRS of the bbox.
    gridsize : int
        Size of the grid in pixels.
    subdir : str
        Subdirectory within save_dir, where the data should be saved, if downloaded.
    rotation : Union[float,int]
        Angle of rotation, in degrees, of the data to align it with North.
    plot : bool
        Flag to indicate if data should be plotted after loading.
    force_download : bool
        Bool to indicate if data should be downloaded and saved, even if it already exists in the `save_dir`.
    service : str
        Service to use for loading the data, either 'wms' or 'wfs'.
    load_res : float
        Resolution of the data when loading it.
    load_res_raw : float
        Resolution of the data when loading it before cropping.
    
    Returns
    -------
    gdf: GeoDataFrame
        if service is 'wfs'
        
    or 
    
    img: numpy array
        if service is 'wms'
    mask: numpy array
        if service is 'wms'
        
    """

    save_path = f'{save_dir}/{subdir}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path_shp = f'{save_path}/data.shp'

    if service == 'wfs':
        try:
            if force_download:
                raise Exception('Forced to download new road data')

            gdf = gpd.read_file(save_path_shp).set_index('gml_id')
            bbox_found = gdf.geometry.total_bounds
            if not bb.fits_in('1in2', bbox, bbox_found):
                raise TypeError(
                    f'Loaded file has wrong bbox, {bbox} does not fit in {bbox_found}')
            print('Succesfully loaded roads from local data')

        except (Exception, TypeError) as e:
            print(e)
            # worked until 01-09-2022
            url = 'https://geodata.nationaalgeoregister.nl/inspire/tn-ro/wfs?'
            # still works after 01-09-2022
            url = 'https://geodata.nationaalgeoregister.nl/nwbwegen/wfs'

            gdf = download.wfs(bbox, url, typeName='nwbwegen:wegvakken',
                               srsname='urn:ogc:def:crs:EPSG::28992'
                               )

            gdf.to_file(save_path_shp)

        if isinstance(rotation, (float, int)):
            gdf['geometry'] = gdf['geometry'].rotate(
                rotation, origin=tuple(list(np.reshape(bbox, (2, 2)).mean(0))),
                use_radians=False)

        return gdf

    if service == 'wms':

        if rotation:
            bbox_load = bb.zoomout(bbox)

        else:
            bbox_load = bbox

        # if provider == 'rws':
        #     # gives linestrings
        #     url = 'https://service.pdok.nl/rws/nwbwegen/wms/v1_0?request=GetCapabilities&service=WMS'

        #     dsr = download.wms(url=url,
        #                        bbox=bbox_load,
        #                        layers=['GeodataWegvakken'],
        #                        save_dir=save_path,
        #                        # srs='urn:ogc:def:crs:EPSG::28992'
        #                        srs='EPSG:28992',
        #                        load_res=load_res_raw,
        #                        )[0]
        provider = 'kadaster' 
        if provider == 'kadaster':
            url = 'https://service.pdok.nl/kadaster/tn/wms/v1_0?request=GetCapabilities&service=WMS'
            dsr = download.wms(url=url,
                               bbox=bbox_load,
                               layers=['TN.RoadTransportNetwork.RoadArea'],
                               save_dir=save_path,
                               # srs='urn:ogc:def:crs:EPSG::28992'
                               srs='EPSG:28992',
                               load_res=load_res_raw,
                               )[0]

        save_name = 'test'
        crop_file = f'{save_name}_crop.tif'

        img_raw = np.transpose(dsr.read(), [1, 2, 0])
        mask_raw = ~(img_raw == [255, 255, 255]).all(2)

        mask_rot = rotate_crop(
            mask_raw.astype(float),
            rotation,
            gridsize,
            res=load_res_raw) > 0.1

        mask = hf.reduce_res_mean(int(load_res / load_res_raw), mask_rot)[0]

        dsr = crop_DatasetReader(dsr, shapes=bbox, gridsize=gridsize,
                                 folder=save_dir, rotation=rotation,
                                 crop_file=crop_file, bbox=bbox, fill=True,
                                 nodata=False)

        img = np.transpose(dsr.read(), [1, 2, 0])

        return img, mask


def water(osm_gdf, bbox, arr_shape, plot=False): 
    """
    Extract water features from OSM geodataframe and convert to numpy array.
    
    Parameters
    ----------
    osm_gdf : GeoDataFrame
        OSM geodataframe containing natural and landuse features
    bbox : tuple of ints
        Bounding box in the format [min_x, min_y, max_x, max_y]
    arr_shape : tuple
        Shape of the output numpy array (rows, cols)
    plot : bool
        Whether or not to plot the water features in the geodataframe. 
        Default is False
        
    Returns
    -------
    water_gdf : GeoDataFrame
        GeoDataFrame containing the water features
    water_arr : np.ndarray
        Numpy array containing the water features. Each cell is 1 if it contains water and 0 otherwise.
    """
    
    # osm_gdf = OpenStreetMaps(bbox, tags=dict(natural=True, landuse=True), rotation=rotation, zoomout=zoomout)

    # water geodataframe
    water_gdf = osm_gdf[((osm_gdf['natural'] == 'water').values +
                         (osm_gdf['landuse'] == 'aquaculture').values +
                         (osm_gdf['landuse'] == 'basin').values +
                         (osm_gdf['landuse'] == 'salt_pond').values +
                         (osm_gdf['landuse'] == 'farmyard').values +
                         (osm_gdf['landuse'] == 'greenfield').values +
                         (osm_gdf['landuse'] == 'flowerbed').values +
                         (osm_gdf['landuse'] == 'vineyard').values)]

    if plot:
        plot2d.gdf(
            water_gdf, bbox=bbox,
            # crs_out=crs_out, crs_init=crs_init,
            address=address, column=['natural', 'landuse'])

    water_arr = _gdf_to_array(
        water_gdf,
        load_res=0.5,
        bbox=bbox,
        arr_shape=arr_shape)
    # water_arr = rotate_crop(water_arr, gridsize=gridsize, res=load_res, rotation=rotation)

    return water_gdf, water_arr


def grass(osm_gdf, bbox, arr_shape, plot=False):
    """
    Extract grass features from OSM geodataframe and convert to numpy array.
    
    Parameters
    ----------
    osm_gdf : GeoDataFrame
        OSM geodataframe containing natural and landuse features
    bbox : tuple of ints
        Bounding box in the format [min_x, min_y, max_x, max_y]
    arr_shape : tuple
        Shape of the output numpy array (rows, cols)
    plot : bool
        Whether or not to plot the grass features in the geodataframe. 
        Default is False
        
    Returns
    -------
    water_gdf : GeoDataFrame
        GeoDataFrame containing the grass features
    water_arr : np.ndarray
        Numpy array containing the grass features. Each cell is 1 if it contains water and 0 otherwise.
    """
    
    grass_gdf = osm_gdf[((osm_gdf['landuse'] == 'grass').values +
                        (osm_gdf['natural'] == 'grassland').values +
                        (osm_gdf['natural'] == 'scrub').values +
                        (osm_gdf['natural'] == 'heath').values +
                         (osm_gdf['landuse'] == 'forest').values +
                         (osm_gdf['landuse'] == 'orchard').values +
                         (osm_gdf['landuse'] == 'meadow').values +
                         (osm_gdf['landuse'] == 'farmyard').values +
                         (osm_gdf['landuse'] == 'farmland').values +
                         (osm_gdf['landuse'] == 'greenfield').values +
                         (osm_gdf['landuse'] == 'flowerbed').values +
                         (osm_gdf['landuse'] == 'vineyard').values +
                         (osm_gdf['landuse'] == 'village_green').values)]

    print('grass_gdf', grass_gdf)

    if plot:
        plot2d.gdf(
            grass_gdf, bbox=bbox,
            # crs_out=crs_out, crs_init=crs_init,
            address=address)

    grass_arr = _gdf_to_array(
        grass_gdf,
        load_res=0.5,
        bbox=bbox,
        arr_shape=arr_shape)
    # grass_arr = rotate_crop(water_arr, gridsize=gridsize, res=load_res, rotation=rotation)

    return grass_gdf, grass_arr


def OpenStreetMaps(bbox, rotation=None, delete_points=True, zoomout=False,
                   tags=dict(natural=True), plot=False, **kwargs):
    """
    Download OpenStreetMaps features as GeoDataFrame within a bounding box.
    
    The function makes use of the osmnx library to download the data, and the 
    resulting GeoDataFrame can be plotted and transformed as required. The 
    function returns the resulting GeoDataFrame.
    
    Parameters
    ----------
    bbox : tuple or list of ints
        The bounding box as a list of 4 coordinates [xmin, ymin, xmax, ymax].
    rotation : float, optional
        Angle of rotation to rotate data ,in degrees. Default is None.
    delete_points : bool, optional
        Whether to delete Point features from the GeoDataFrame. Default is True.
    zoomout : bool, optional
        Whether to zoom out the bounding box. Default is False.
    tags : dict, optional
        The tags of the features to be downloaded. Default is dict(natural=True).
    plot : bool, optional
        Whether to plot the data. Default is False.
    **kwargs : 
        Additional keyword arguments passed to the `gdf.plot()` function.
        
    Returns
    -------
    gdf : GeoDataFrame
        The features from OpenStreetMaps as a GeoDataFrame
    """

    if zoomout:
        bbox = bb.zoomout(bbox)

    # transform coordinate system of bounding box    
    bbox_4326 = crs.transform(bbox)
    west, south, east, north = bbox_4326

    tag_keys = list(tags.keys())

    gdf = ox.geometries.geometries_from_bbox(north=north, south=south,
                                             east=east, west=west, tags=tags)

    if delete_points:
        gdf = gdf[gdf.geometry.type != 'Point']

    # make column with tags
    gdf['tag'] = np.nan
    for tag in tag_keys:
        try:
            gdf.loc[~gdf[tag].isnull(), 'tag'] = tag
        except BaseException:
            pass

    gdf = gdf.to_crs(epsg=28992)

    # rotate around origin
    origin = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    if rotation:
        gdf['geometry'] = gdf['geometry'].rotate(
            rotation,
            origin=origin,
            use_radians=False)

    fig, ax = plt.subplots()
    if plot and not gdf.empty:
        gdf.plot(column='tag', ax=ax, legend=True, **kwargs)
        plt.show()

    ax.set_title(' and '.join(tag_keys))

    return gdf


# def top10nl(bbox, save_dir, zoomout=False):
#     """
#     Download Top10NL satellite image data as GeoTIFF within a bounding box.
    
#     Parameters
#     ----------
#     bbox: tuple of ints
#         The bounding box of the area to download the Top10NL data for. The 
#         order is (xmin, ymin, xmax, ymax)
#     save_dir: str
#         The directory where the downloaded GeoTIFF will be saved.
#     zoomout : bool
#         Whether or not to zoom out the bounding box slightly to include more 
#         data so that the image will not be too small after rotation and 
#         cropping. Default is False.
    
#     Returns
#     -------
#     list of datasetreader
#         The list of datasets containing the downloaded Top10NL data.
#     """
    
#     if zoomout:
#         bbox = bb.zoomout(bbox)

#     layers = ['terreinvlak', 'gebouwvlak', 'wegdeelvlak']

#     save_name = 'top10nl'
#     url = 'https://geodata.nationaalgeoregister.nl/top10nlv2/wms?SERVICE=WMS '

#     img = download.wms(
#         url,
#         bbox,
#         layers,
#         save_dir,
#         save_name=save_name,
#         srs='urn:ogc:def:crs:EPSG::28992')

#     return img


# not in use?
# def load_SVF(bbox, folder, rotation=None, svf_path='data/SVF/',
#              crop_file='svf.tif', always_reload=False, zoomout=False):

#     if zoomout:
#         bbox = bb.zoomout(bbox)
#     bbox = tuple(bbox)

#     try:
#         if always_reload:
#             raise Exception('Forced reload')
#         dsr = rasterio.open(f'{folder}/{crop_file}', driver="GTiff")

#         bbox_found = tuple(dsr.bounds)
#         if bbox.equals(bbox, bbox_found):
#             raise TypeError(f'Loaded file has wrong bbox, {bbox} does not fit in {bbox_found}')
#         else:
#             print(f'Found good file, {bbox} does fit in {bbox_found}')

#         return dsr

#     except (Exception, TypeError) as e:
#         print(e)
#         bladindex_table = pd.read_csv(
#             'data/bladindex_EPSG28992.txt', index_col='bladnr')

#         bladindex = bb.to_mapindex(bbox, bladindex_table)

#         download_loc = 'data/SVF/'
#         svf_files = svf(bladindex, download_loc=download_loc)

#         dsr = rasterio.open(svf_files[0], driver="GTiff")

#         print('len(svf_files):', len(svf_files))

#         if len(svf_files) > 1:
#             datasetreaders = [rasterio.open(svf_file) for svf_file in svf_files]
#             mosaic, out_trans = merge(datasetreaders, method='last')
#             out_meta = dsr.meta.copy()
#             out_meta.update({"driver": "GTiff",
#                              "height": mosaic.shape[1],
#                              "width": mosaic.shape[2],
#                              "transform": out_trans,
#                               "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
#                              }
#                             )
#             with rasterio.open(download_loc + 'merge_temp.tif', "w", **out_meta) as file:
#                 file.write(mosaic)

#             dsr = rasterio.open(download_loc + 'merge_temp.tif', driver='GTiff')
#             os.remove(download_loc + 'merge_temp.tif')


#         # crop the dataset if a bounding box is defined
#         dsr = crop_DatasetReader(dsr, bbox, rotation=rotation, crop_file=crop_file, folder=folder, plot=True,
#                                  nodata=True)


#         return dsr


def svf(
        bbox,
        folder,
        save_name,
        download_loc='data/maps/SVF/',
        rotation=None,
        fill=True,
        fill_max_sigma=25):
    """
    Download or load the Sky View Factor (SVF) data for the given bounding box. 
    
    The data is downloaded from KNMI and saved in the specified folder with the given save_name.
    The data is then cropped to the bbox, if rotation is specified the data is also rotated.
    
    Parameters
    ----------
    bbox : tuple of floats
        Bounding box of the data in the format (minx, miny, maxx, maxy)
    folder : str
        Folder where the data should be saved.
    save_name : str
        Name of the saved tif-file.
    download_loc : str
        Location to download the data
    rotation : Optional[float], optional
        Rotation angle in degrees, by default None
    fill : bool, optional
        If True, fill the no-data pixels with interpolation, by default True
    fill_max_sigma : int, optional
        Maximum standard deviation for the fill process, by default 25
        
    Returns
    -------
    svf_dsr : rasterio.io.DatasetReader
        Sky view factor DatasetReader
    """
    
    bbox_orig = bbox

    gridsize = bbox[2] - bbox[0]

    if rotation:
        bbox = bb.zoomout(bbox)

    try:
        svf_dsr = rasterio.open(f'{folder}/{save_name}.tif')
        print('Found SVF-file')
        if not bb.equals(svf_dsr, bbox):
            raise ValueError('not equal: ', svf_dsr.bounds, bbox)
    except BaseException:
        mapindex_table = pd.read_csv('data/bladindex_EPSG28992.txt',
                                     index_col='bladnr')

        filenames = bb.to_mapindex(bbox, mapindex_table=mapindex_table)

        datasetreaders = []
        for filename in filenames:
            fn_prefix = 'SVF_r'
            extension = '.tif'

            try:
                if fill:
                    fn = f'{fn_prefix}{filename}_filled{extension}'
                if not fill:
                    fn = f'{fn_prefix}{filename}{extension}'

                svf_dsr = rasterio.open(f'{download_loc}/{fn}')
                print('Loaded svf_dsr from', f'{download_loc}/{fn}')
            except BaseException:
                fn = f'{fn_prefix}{filename}{extension}'
                svf_dsr = download.knmi(
                    fn,
                    dataset_name="SVF_NL",
                    download_loc=download_loc,
                    fill=fill,
                    fill_max_sigma=fill_max_sigma)
            datasetreaders.append(svf_dsr)

        if len(datasetreaders) > 1:
            svf_arr, out_trans = rasterio.merge.merge(
                datasetreaders, bounds=bbox, method='last')

        else:
            svf_arr, out_trans = rasterio.mask.mask(
                datasetreaders[0], shapes=[
                    shapely.geometry.box(
                        *bbox)], crop=True)

        res = svf_dsr.res[0]
        out_meta = datasetreaders[0].meta
        out_meta.update({"driver": "GTiff",
                         "height": (bbox[3] - bbox[1]) / res,
                         "width": (bbox[2] - bbox[0]) / res,
                         "transform": out_trans,
                         "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                         }
                        )
        with rasterio.open(f'{folder}/{save_name}.tif', "w", **out_meta) as file:
            file.write(svf_arr)

        svf_dsr = rasterio.open(f'{folder}/{save_name}.tif')

    if rotation:
        if fill:
            crop_file = f'{save_name}_filled_crop.tif'
        if not fill:
            crop_file = f'{save_name}_crop.tif'
        svf_dsr = crop_DatasetReader(
            svf_dsr,
            shapes=bbox_orig,
            gridsize=gridsize,
            folder=folder,
            rotation=rotation,
            crop_file=crop_file,
            bbox=bbox_orig,
            fill=fill,
            nodata=True)

    return svf_dsr


def satellite_image(
        bbox,
        save_dir,
        load_res=1,
        rotation=None,
        layers=['Actueel_ortho25'],
        force_download=False,
        **kwargs):
    """
    Download and optionally rotate a satellite image from the Netherlands PDOK WMS service within a bounding box.
    
    Parameters
    ----------
    bbox : tuple of ints
        Bounding box in the form [xmin, ymin, xmax, ymax]
    save_dir : str
        Directory to save the image.
    load_res : float, optional
        Resolution at which to load the image, by default 1
    rotation : float, optional
        Angle in degrees to rotate the image, by default None
    layers : list, optional
        List of layers to download from the WMS service, by default ['Actueel_ortho25']
    force_download : bool, optional
        Flag to indicate whether to redownload the image if it already exists, by default False
    kwargs : dict, optional
        Additional parameters to pass to the download.wms function
    
    Returns
    -------
    img : np.ndarray
        The satellite image as a numpy array
    """
    
    
    gridsize = bbox[2] - bbox[0]
    if rotation:
        bbox = bb.zoomout(bbox)

    url = 'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=wms'
    dsr = download.wms(
        url=url,
        bbox=bbox,
        load_res=load_res,
        format='image/jpeg',
        layers=layers,
        save_dir=save_dir,
        srs='EPSG:28992',
        force_download=force_download,
        **kwargs)[0]

    if rotation:
        if 'fn' in kwargs:
            fn = kwargs['fn']
        else:
            fn = 'sat_img'

        crop_file = f'{fn}_crop.tif'

        dsr = crop_DatasetReader(dsr, shapes=bbox, gridsize=gridsize,
                                 folder=save_dir, rotation=rotation,
                                 crop_file=crop_file, bbox=bbox, fill=True,
                                 nodata=False)

    # transpose to make the image plottable with plt.imshow
    img = np.transpose(dsr.read(), [1, 2, 0])

    return img

# %% Other data

def anthropogenic_buildings(bag_gdf, air_temp, simulate_averages, nr_buildings_els, area):
    # count the number of households by looking at which buildings have the
    # "gebruiksdoel" as "woonfunctie"

    col_gebruiksdoel = [
        col for col in bag_gdf if col.startswith('gebruiksd')][0]
    nr_households = bag_gdf[col_gebruiksdoel].str.contains('woon').sum()

    total_joule_gas = (
        35e6 * bag_gdf["Gemiddelde_aardgaslevering_woningen_gecorrigeerd"].sum())
    total_joule_electr = (
        3.6e6 * bag_gdf["Gemiddelde_elektriciteitslevering_woningen"].sum())

    if simulate_averages:
        pass

    if not simulate_averages:
        outside_temp_average = air_temp.rolling('10d').mean()
        temp_timestep = (np.diff(outside_temp_average.index).min()
                         / np.timedelta64(1, 's'))

        degree_day = np.clip(
            18 - outside_temp_average, a_min=0, a_max=None)
        gas_power_total = (total_joule_gas
                           * (degree_day / degree_day.sum())
                           / temp_timestep)

    buildings_total_surface = nr_buildings_els * area

    gas_power_m2 = gas_power_total / buildings_total_surface

    electr_power_m2 = total_joule_electr / (60 * 60 * 24 * 365) / buildings_total_surface

    anthropogenic_power_m2 = (gas_power_m2 + electr_power_m2)
    
    return


# %% Get and combine data datasets

def elevation_and_masks(bbox, save_dir, load_res, gridsize,
                        rotation=False,
                        address=None,
                        crs_out=None,
                        force_download=False,
                        ahn_version=2,
                        ahn_method='sheets',
                        plot_bag=False,
                        plot_bag_elevation=False,
                        plot_energy=False,
                        plot_roads=False,
                        plot_dsm=False,
                        plot_dtm=False,
                        plot_chm=False):
    
    """
    This function downloads elevation data and generates masks for all surface 
    type (e.g. roads, buildings, water).
    
    Parameters
    ----------
    bbox : tuple of ints
        Bounding box of the area of interest in the format (x_min, y_min, x_max, y_max)
    save_dir : Path
        Directory where the data should be saved.
    load_res : float
        Resolution of the data.
    gridsize : float
        Size of the grid in meter
    rotation : bool, optional
        Rotation angle, by default False
    address : str, optional
        Address to locate the area, by default None
    crs_out : str, optional
        Output CRS, by default None
    force_download : bool, optional
        If True, re-download data, by default False
    ahn_version : int, optional
        Version of the elevation data set, by default 2
    ahn_method : str, optional
        Method to download elevation data, by default 'sheets'
    plot_bag : bool, optional
        if True, plot building data, by default False
    plot_bag_elevation : Optional[bool], optional
        if True, plot building data with elevation, by default False
    plot_energy : Optional[bool], optional
        if True, plot solar panel data, by default False
    plot_roads : Optional[bool], optional
        if True, plot road data, by default False
    plot_dsm : Optional[bool], optional
        if True, plot digital surface model, by default False
    plot_dtm : Optional[bool], optional
        if True, plot digital terrain model, by default False
    plot_chm : Optional[bool], optional
        if True, plot canopy height model, by default False
    """
    
    arr_shape = (int((bbox[2] - bbox[0]) / load_res),
                 int((bbox[3] - bbox[1]) / load_res))

    data_path = save_dir / 'elvtn_data.npz'

    try:
        if force_download:
            raise Exception
        data = np.load(str(data_path), allow_pickle=True)
        # chm_arr = data["chm_arr"]
        dsm_arr = data["dsm_arr"]
        masks = data["masks"].item()
        rotation = float(data["rotation"])
        bag_gdf = data["bag_gdf"]
        # roads_gdf = data["roads_gdf"]

        print('Succesfully loaded Canopy Height Model and masks')
        return dsm_arr, masks, rotation, bag_gdf
    except (FileNotFoundError, KeyError, Exception) as e:
        # print(e)
        # --- Buildings ---

        bag_gdf, rotation = bag(
            bbox=bbox,
            load_res=load_res,
            save_dir=save_dir,
            rotation=rotation,
            arr_shape=arr_shape,
            force_download=force_download)

        print(f'\nCalculated rotation: {rotation} degrees \n')

        crs_init = bag_gdf.crs

        if plot_bag or plot_chm:
            plot2d.gdf(
                bag_gdf,
                save_dir=save_dir,
                bbox=bbox,
                crs_out=crs_out,
                crs_init=crs_init,
                cmap=plt.cm.viridis,
                address=address)

        if bag_gdf.empty is not True:
            buildings_mask = _gdf_to_array(bag_gdf,
                                        load_res=load_res,
                                        bbox=bbox,
                                        arr_shape=arr_shape).astype(bool)

            # Increase buildings mask size a little bit
            buildings_mask_convolve = scipy.signal.convolve2d(
                buildings_mask, np.ones((3, 3)) / 9, mode='same')
            buildings_mask = buildings_mask_convolve > 0.3

        else:
            buildings_mask = np.zeros(arr_shape, dtype=bool)

        # %%% Elevation
        chm_dsr, dsm_dsr, _, dsm_fill = elevation(
            bbox=bbox, save_dir=save_dir, zoomout=True, ahn_version=ahn_version, method=ahn_method)
        blur = False
        chm_zoomout = chm_dsr.read(1)
        if blur:
            chm_zoomout = cv2.medianBlur(np.float32(chm_zoomout), ksize=3)

        dsm_zoomout = dsm_fill.read(1)
        if blur:
            dsm_zoomout = cv2.medianBlur(np.float32(dsm_zoomout), ksize=3)

        # ---- DSM
        dsm_arr = rotate_crop(dsm_zoomout, rotation, gridsize, res=load_res)
        plt.imshow(dsm_arr)
        plt.title('dsm_arr')
        plt.colorbar()
        plt.show()
        hist = plt.hist(
            hf.roundto(
                dsm_arr.flatten(), 0.5), bins=int(
                math.ceil(
                    dsm_arr.max() - dsm_arr.min()) / 0.5))
        plt.clf()
        ground_level = hist[1][np.where(hist[0] / dsm_arr.size > 0.05)[0][0]]

        dsm_arr = np.clip(dsm_arr, a_min=ground_level, a_max=None)

        dsm_arr -= dsm_arr.min()

        offline_data.save_array_as_img(
            dsm_arr, folder=save_dir, filename='dsm_unblurred.png')
        plt.imshow(dsm_arr)
        plt.title('dsm_arr')
        plt.colorbar()
        plt.show()
        input('Change dsm_unblurred_scale*.png if needed, then press enter...')

        dsm_arr_old = dsm_arr.copy()
        dsm_arr = offline_data.load_img_as_arr(folder=save_dir, file='dsm.png')
        
        if not np.all(np.equal(dsm_arr_old, dsm_arr)):
            plt.imshow(dsm_arr)
            plt.title('dsm_arr after adjustments')
            plt.colorbar()
            plt.show()


        # ---- CHM
        # CHM is used to properly detect environmental objects, corrected by variation in ground elevation
        plt.imshow(chm_zoomout)
        plt.title('chm_zoomout')
        plt.show()

        chm_arr = rotate_crop(chm_zoomout, rotation, gridsize, res=load_res)
        set_ground_level = True
        plt.imshow(chm_arr)
        plt.title('chm_arr before set_ground_level')
        plt.show()
        if set_ground_level:
            hist = plt.hist(
                hf.roundto(
                    chm_arr.flatten(), 0.5), bins=int(
                    math.ceil(
                        chm_arr.max() - chm_arr.min()) / 0.5))
            plt.clf()
            ground_level = hist[1][np.where(
                hist[0] / chm_arr.size > 0.05)[0][0]]
            chm_arr = np.clip(chm_arr, a_min=ground_level, a_max=None)

        chm_arr -= chm_arr.min()

        offline_data.save_array_as_img(
            chm_arr, folder=save_dir, filename='chm_unblurred.png')
        plt.imshow(chm_arr)
        plt.title('chm_arr')
        plt.colorbar()
        plt.show()
        input('Change chm_unblurred_scale*.png if needed, then press enter...')

        chm_arr = offline_data.load_img_as_arr(folder=save_dir, file='chm.png')
        chm_arr -= chm_arr.min()

        # --- Detect balconies (outside buildings that exceed their size outside the kadaster area)
        possibly_balconies = (
            scipy.ndimage.binary_dilation(buildings_mask, iterations=2)
            ^ buildings_mask)

        balconies_mask = possibly_balconies * (dsm_arr > 3)
        buildings_mask += balconies_mask

        if plot_bag_elevation and bag_gdf.empty is not True:
            print(
                'Plotting buildings heights from "Basisregistratie Adressen en Gebouwen"')
            if 'Elevation mean' not in bag_gdf.columns:
                dsm_buildings = rs.zonal_stats(
                    bag_gdf.dropna(subset=['geometry']).reset_index(),
                    dsm_arr,
                    prefix='Elevation ',
                    stats=['mean'],
                    affine=Affine(load_res,
                                  0.0,
                                  bbox[0],
                                  0.0,
                                  -load_res,
                                  bbox[3]),
                    nodata=0,
                    all_touched=True,
                    geojson_out=True)
            # adds heightdata to GDF
            bag_gdf = gpd.GeoDataFrame.from_features(dsm_buildings)
            try:
                bag_gdf = bag_gdf.set_index('index')
                plot2d.gdf(
                    bag_gdf, save_dir=save_dir, column=['Elevation mean'],
                    bbox=bbox, crs_out=crs_out, crs_init=crs_init,
                    cmap=plt.cm.viridis, address=address)
            except KeyError:
                pass

        # %%% Energy consumption
        energy_data = (pd.read_csv(
            './data/Publicatiefile_Energie_postcode6_2019.csv',
            delimiter=';', na_values='.',
            skipinitialspace=True)
            .set_index('Postcode6').astype(int, errors='ignore'))

        bag_gdf = bag_gdf[bag_gdf.postcode.notna()]

        try:
            energy_data_houses = energy_data.loc[bag_gdf.postcode].set_index(
                bag_gdf.index)

            bag_gdf = pd.concat(
                [bag_gdf, energy_data_houses], axis=1, sort=False)

            energy_cols = bag_gdf.columns.values[
                bag_gdf.columns.str.contains('gas|elektr', case=False)]

            bag_gdf.crs = crs_init

            if plot_energy:
                plot2d.gdf(bag_gdf, save_dir=save_dir, column=energy_cols,
                           bbox=bbox, crs_out=crs_out, crs_init=crs_init,
                           cmap=plt.cm.viridis, address=address)
        except Exception as e:
            # print(e)
            pass

        # %%% Roads
        try:
            service = 'wms'

            if service == 'wms':

                roads_img, roads_mask = roads(save_dir, bbox, gridsize=gridsize, crs_init=crs_init,
                                              rotation=rotation, force_download=force_download, service='wms')
                # roads_mask = ~(roads_img == [255,255,255]).all(2)

            if service == 'wfs':

                roads_gdf = roads(save_dir, bbox,
                                  crs_init=crs_init, rotation=rotation,
                                  force_download=force_download)

                if plot_roads:
                    plot2d.gdf(
                        roads_gdf,
                        bbox=bbox,
                        crs_out=crs_out,
                        crs_init=crs_init,
                        address=address)

                roads_mask = _gdf_to_array(roads_gdf,
                                        load_res=load_res,
                                        bbox=bbox,
                                        arr_shape=arr_shape).astype(bool)
        except ValueError:
            roads_mask = np.zeros(arr_shape, dtype=bool)
            roads_gdf = gpd.GeoDataFrame()

        roads_mask[buildings_mask] = False

        # %%% OpenStreetMaps data

        print('\n Loading OpenStreetMaps data... \n')
        osm_gdf = OpenStreetMaps(
            bbox,
            tags=dict(
                natural=True,
                landuse=True),
            rotation=rotation,
            zoomout=True)

        # %%% Water

        try:
            water_mask_osm = water(
                osm_gdf,
                bbox=bbox,
                arr_shape=arr_shape,
                plot=True)[1].astype(bool)
        except Exception as e:
            water_mask_osm = np.zeros_like(roads_mask)

        # surface map has "nodata" value on water areas
        dsm_nodatamask_zoomout = dsm_dsr.read(1, masked=True).mask
        if (dsm_nodatamask_zoomout == False).any() and isinstance(
                dsm_nodatamask_zoomout, np.bool_):
            water_mask = water_mask_osm
        else:
            dsm_nodatamask = rotate_crop(
                dsm_nodatamask_zoomout, rotation, gridsize, load_res)

            water_mask = dsm_nodatamask & water_mask_osm

        # %%% Grass
        try:
            # if True:
            raise Exception()
            grass_mask_osm = grass(
                osm_gdf,
                bbox=bbox,
                arr_shape=arr_shape,
                plot=True)[1].astype(bool)
        # else:
        except Exception as e:
            grass_mask_osm = np.zeros_like(roads_mask)


        # %%% Trees

        trees_mask = (np.invert(buildings_mask)
                      * (chm_arr > 0.3))

        # %%% Combine all

        water_mask[trees_mask] = False

        chm_arr[water_mask] = 0
        dsm_arr[water_mask] = 0

        grass_mask = (grass_mask_osm
                      * np.invert(water_mask)
                      * np.invert(buildings_mask)
                      * np.invert(roads_mask)
                      * np.invert(trees_mask))

        plt.imshow(grass_mask_osm)
        plt.title('grass_mask_osm')
        plt.show()

        plt.imshow(grass_mask)
        plt.title('grass_mask')
        plt.show()

        open_surface_mask = (np.invert(buildings_mask)
                             & np.invert(water_mask)
                             & np.invert(grass_mask)
                             & np.invert(roads_mask))

        masks = {"buildings": buildings_mask,
                 "trees": trees_mask,
                 "open surface": open_surface_mask,
                 "water": water_mask,
                 "grass": grass_mask,
                 "roads": roads_mask}

        np.savez(data_path, dsm_arr=dsm_arr, masks=masks, rotation=rotation,
                 bag_gdf=bag_gdf)

    if bag_gdf.empty is not True:
        bag_gdf.reset_index().to_file(f'{save_dir}/bag_gdf/data.shp')

    return dsm_arr, masks, rotation, bag_gdf
