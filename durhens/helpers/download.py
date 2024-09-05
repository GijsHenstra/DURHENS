#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:16:19 2021

@author: gijshenstra
"""

import rasterio
import os
import geopandas as gpd
import requests
import zipfile
import rasterio.mask
import sys


from affine import Affine
from owslib.wcs import WebCoverageService
from requests import Request

from owslib.wms import WebMapService

from helpers import boundingbox
from helpers import crs
from helpers import fill_missing



WFS_DICT = dict(service='WFS', 
                version="2.0.0", 
                request='GetFeature',
                # typeName=layer,
                outputFormat='application/gml+xml; version=3.2',
                srsname='urn:ogc:def:crs:EPSG::28992', 
                # bbox=bbox_str,
                # start_index = 0,
                # count=1000,
                # sortby='bag:identificatie'
                )


WCS_DICT = dict(crs='urn:ogc:def:crs:EPSG::28992',
                resx=0.5,
                resy=0.5)


def wfs(bbox, url,  typeName, count=1000, zoomout=False, index=None,
                  drop_cols=None, **kwargs):
    """
    Retrieves data from a Web Feature Service (WFS) service in a given bounding box.

    Parameters
    ----------
    bbox : tuple of ints
        The bounding box of the area for which data should be retrieved, in the format (xmin, ymin, xmax, ymax)
    url : str
        The URL of the WFS service
    typeName : str
        The name of the layer to be retrieved from the WFS service
    count : int, optional
        The number of features to be retrieved per request, 
        by default 1000 (which is themaximum)
    zoomout : bool, optional
        Whether to zoom out the bounding box to compensate for cropping while rotating the map.
    index : str, optional
        The column to use as the index of the returned GeoDataFrame, by default None
    drop_cols : list of strings, optional
        A list of columns to be dropped from the returned GeoDataFrame, by default None
    **kwargs : 
        Additional arguments to be passed to the WFS request

    Returns
    -------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the retrieved data
    """
    
    gdf = gpd.GeoDataFrame() # start with empty gdf
    
    if zoomout:
        bbox = boundingbox.zoomout(bbox)

    # transform bbox if another crs than the standard crs is used
    if 'srsname' in kwargs:
        srsname = kwargs['srsname']
        if srsname != WFS_DICT['srsname']:
            bbox = crs.transform(
                bbox, crs_in=WFS_DICT['srsname'], crs_out=srsname)
        
            bbox = boundingbox.reverse_order(bbox)
            bbox = tuple(bbox)

    bbox_str = ','.join(map(str, bbox))
    
    kwargs.update({'bbox': bbox_str,
                    'typeName': typeName})
    
    params = WFS_DICT.copy()
    params.update(kwargs)
    
    url_disp = {'/'.join(url.split('/')[2:5])}
    print(f"Retreiving data using WFS from {url_disp}")
    
    startindex = 0
    while True:
        params.update({'startindex': startindex})
        
        # Parse the URL with parameters
        q = Request('GET', url, params=params).prepare().url
        
        # Read data from URL
        gdf_snippet = gpd.read_file(q, )
        gdf = gdf.append(gdf_snippet)
        
        if len(gdf_snippet) < count - 1:
            break
        
        startindex += count
        
    if index:
        gdf = gdf.set_index(index)

    if drop_cols:
        drop_cols = [col for col in drop_cols if col in gdf.columns]
        gdf = gdf.drop(columns=drop_cols)

    # transform coordinate system
    if gdf.crs != WFS_DICT['srsname']:
        try:
            gdf = gdf.to_crs(WFS_DICT['srsname'])
        except (AttributeError, ValueError): 
            pass

    return gdf

def wms(url, bbox, layers, save_dir, size=None, load_res=1, 
        format='image/png', srs='EPSG:28992', fn_prefix=None, force_download=False, **kwargs):
    """
    Retrieve and save raster data from a WebMapService (WMS) as tif files.
    
    Parameters
    ----------
    url : str
        The url of the WMS service.
    bbox : tuple of ints
        A bounding box in the format of (xmin, ymin, xmax, ymax).
    layers : list or str
        A list of layers to be downloaded, or a single layer as a string.
    save_dir : str
        The directory to save the tif files.
    size : tuple, optional
        The size of the image to download, by default None
    load_res : int, optional
        The resolution of the image to download, by default 1
    format : str, optional
        The format of the image to download, by default 'image/png'
    srs : str, optional
        The spatial reference system of the image to download, by default 'EPSG:28992'
    fn_prefix : str, optional
        The prefix for the filename, by default None
    force_download : bool, optional
        If True, always download new files, by default False
    **kwargs : 
        Additional key word arguments to pass to the WMS request.
    
    Returns
    -------
    dsr_lst : list
        A list of datasetreaders representing the downloaded tif files.
    """
    
    wms = WebMapService(url, version='1.3.0')
    if fn_prefix:
        fn_prefix = wms.provider.contact.organization
    
    try:
        styles = [list(wms[layer].styles.keys())[0] for layer in layers]
    except:
        styles=None
    
    if size is None:
        width = int((bbox[2] - bbox[0]) / load_res)
        height = int((bbox[3] - bbox[1]) / load_res)
        
        size = (width, height)
    
    if layers is None:
        layers = wms.contents.keys()
        
    # initialize empty list of datasetreaders
    dsr_lst = []
        
    for layer in layers:
        if 'fn' in kwargs:
            fn = kwargs['fn']
            # save_path = f'{save_dir}/{fn}'.lstrip('/') pre 13 sept 23
            save_path = f'{save_dir}/{fn}'
        else:
            # save_path = f'{save_dir}/{fn_prefix}_{layer}'.lstrip('/') pre 13 sept 23
            save_path = f'{save_dir}/{fn_prefix}_{layer}'
        
        try:
            if force_download is True:
                raise Exception('Downloading new file...')
            dsr = rasterio.open(f'{save_path}.tif')
            print(f'loaded {save_path}.tif')
            if not boundingbox.equals(dsr, bbox):
                raise Exception('Bounding boxes are not equal')
            if not size == dsr.shape:
                raise Exception('Loaded resolution is different')
            
        except: 
            
            # try:
            response = wms.getmap(
                layers=[layer],
                styles=styles,
                bbox=bbox,
                size=size,
                transparent=False,
                format=format,
                srs=srs
               )
        
            img = response.read()
            
            with open(f'{save_path}.png', 'wb') as file:
                file.write(img)
                
            out_meta = {'driver': 'GTiff',
                        'dtype': 'uint8',
                        'nodata': None,
                        'width': width,
                        'height': height,
                        'count': 3,
                        'crs': 'EPSG:28992',
                        'transform': Affine(load_res, 0.0, bbox[0],
                                            0.0, -load_res, bbox[3])}
                
            with rasterio.open(f'{save_path}.tif', "w", **out_meta) as file:
                print(f'saved to {save_path}')
                file.write(rasterio.open(f'{save_path}.png').read())
                
            os.remove(f'{save_path}.png')
                
            dsr = rasterio.open(f'{save_path}.tif')
            
            print(f'saved {layer}')
        # except Exception as e:
        #     print(e)
        #     raise Exception(e)
            
        dsr_lst.append(dsr)
        
    return dsr_lst
    

def wcs(url, identifier, save_dir, bbox, download_format='image/tiff', zoomout=False, 
                  force_download=False, **kwargs):
    """
    Retrieve and download a raster image from a Web Coverage Service (WCS) 
    using the `rasterio` library and save it to a specified directory.
    
    Parameters
    ----------
    url : str
        The url of the WCS service
    identifier : str
        An identifier for the image to retrieve
    save_dir : str
        The directory to save the image to
    bbox : tuple
        A bounding box specifying the area of the image to retrieve
    zoomout : bool, optional
        If True, the bounding box will be zoomed out before retrieval, by default False
    force_download : bool, optional
        If True, the image will be downloaded even if it already exists locally, by default False
    **kwargs : 
        Additional parameters to be passed to the WCS service
    
    Returns
    -------
    dsr : 
        Datasetreader of geotiff raster image
    save_path : str
        the file path of the saved Geotiff
    
    Raises
    ------
    ValueError
        If the bounding box area does not match the area of the found image file locally.
    """
    
    # save_path = f'{save_dir}/{identifier}.tif'.lstrip('/') # pre 13 sept 23
    save_path = f'{save_dir}/{identifier}.tif'
    
    if zoomout:
        bbox = boundingbox.zoomout(bbox)
    
    try:
        if force_download:
            raise Exception
            
        dsr = rasterio.open(save_path, driver="GTiff")
        if boundingbox.to_area(bbox, res=WCS_DICT['resx']) != (dsr.height * dsr.width):
            raise ValueError('Found file has wrong area.')
        
        return dsr, save_path
    
    except:
        wcs = WebCoverageService(url, version='1.0.0')
        
        kwargs.update({'bbox': tuple(bbox),
                       'identifier': identifier})
        
        params = WCS_DICT.copy()
        params.update(kwargs)
        params.update({'format': download_format})
                
        url_disp = '/'.join(url.split('/')[2:5])
        print(f"Retreiving {identifier}-data using WCS.")
        
        # download data
        response = wcs.getCoverage(**params)
        
        # write data to file
        with open(save_path, 'wb') as file:
            print('save_path', save_path)
            file.write(response.read())
        
        # open DataSet reader
        try:
            dsr = rasterio.open(save_path, driver="GTiff")
        except rasterio.RasterioIOError:
            raise ValueError(response.read())
                
        return dsr, save_path
    
# def get_svf(filenames, download_loc = 'data/SVF/'):
    
#     # import logging
#     import sys
#     from datetime import datetime
#     from pathlib import Path
#     # import itertools
    
#     api_url = "https://api.dataplatform.knmi.nl/open-data"
#     api_version = "v1"

#     # Parameters
#     api_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImNjOWE2YjM3ZjVhODQwMDZiMWIzZGIzZDRjYzVjODFiIiwiaCI6Im11cm11cjEyOCJ9"
#     dataset_name = "SVF_NL"
#     fn_prefix = 'SVF_r'
#     filename_postfix = '.tif'
#     dataset_version = "3"
#     max_keys = "10"

#     # Use list files request to request first 10 files of the day.
#     timestamp = datetime.utcnow().date().strftime("%Y%m%d")
#     start_after_fn_prefix = f"KMDS__OPER_P___10M_OBS_L2_{timestamp}"
#     list_files_response = requests.get(
#         f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files",
#         headers={"Authorization": api_key},
#         params={
#             "maxKeys": max_keys, 
#             "startAfterFilename": start_after_fn_prefix},
#     )
#     list_files = list_files_response.json()

#     # logger.info(f"List files response:\n{list_files}")
#     # dataset_files = list_files.get("files")

#     # fn_prefix = common([file['filename'] for file in dataset_files])
# # 
#     # Retrieve first file in the list files response
#     # filename = dataset_files[0].get("filename")
#     filenames = [fn_prefix + fn + filename_postfix for fn in filenames]
    
#     for filename in filenames:
#         if os.path.isfile(download_loc + filename):
#             print(f'Found {filename}')
#             continue
#         else:
#             print(f'Could not find {filename}')
        
#         # logger.info(f"Retrieve file with name: {filename}")
#         endpoint = f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
#         get_file_response = requests.get(endpoint, headers={"Authorization": api_key})
#         if get_file_response.status_code != 200:
#             # logger.error("Unable to retrieve download url for file")
#             # logger.error(get_file_response.text)
#             sys.exit(1)
    
#         download_url = get_file_response.json().get("temporaryDownloadUrl")
#         dataset_file_response = requests.get(download_url)
#         if dataset_file_response.status_code != 200:
#             # logger.error("Unable to download file using download URL")
#             # logger.error(dataset_file_response.text)
#             sys.exit(1)
    
#         # Write dataset file to disk
#         p = Path(download_loc + filename)
#         p.write_bytes(dataset_file_response.content)
#         print(f"Successfully downloaded dataset file to {p}")

#     svf_files = [download_loc + svf_file for svf_file in filenames] 

#     return svf_files
    
# def zip_geotiff2(fn, base_url, download_loc):
#     url = base_url + fn
#     r = requests.get(url, stream=True)
    
#     save_path = download_loc + fn
#     with open(download_loc, 'wb') as fd:
#         for chunk in r.iter_content(chunk_size=chunk_size):
#             fd.write(chunk)

def zip_url(fn, base_url, download_loc, filetype='datasetreader'):
    """
    Downloads a zip file from a specified url, and extract it to a specified location.
    
    Parameters
    ----------
    fn : str
        File name
    base_url : str
        The base url where the zip file is located
    download_loc : str
        The location where the zip file should be downloaded and extracted
    filetype : str, optional
        File type of the downloaded file, by default 'datasetreader'
        
    Returns
    -------
    rst : rasterio Dataset Reader
        Rasterio dataset reader for the unzipped file

    Raises
    ------
    ValueError
        If the filetype is unknown.
    """
    
    #Defining the zip file URL
    url = f'{base_url}/{fn}.zip'
    
    print(f'Starting dowload from {url}')
    
    req = requests.get(url)
    tmp_zip = download_loc + '/temp.zip'
    
    with open(tmp_zip,'wb') as output_file:
        output_file.write(req.content)
    print('Downloading Completed')
    
    with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
        zip_ref.extractall(download_loc)
        
    os.remove(tmp_zip)
    
    if filetype == 'datasetreader':
        rst = rasterio.open(f'{download_loc}/{fn}')
    else:
        raise ValueError(f'Uknown file type: {filetype}')
    
    return rst
    

def knmi(fn, dataset_name, download_loc = '', fill=False, fill_max_sigma=25, key='private'):
    """
    Downloads a geotiff file from KNMI open data platform, and returns a Rasterio dataset reader for the file.
    
    Parameters
    ----------
    fn : str
        File name
    dataset_name : str
        The dataset name on the KNMI open data platform
    download_loc : str, optional
        The location where the file should be downloaded, by default ''
    fill : bool, optional
        Whether to fill missing data in the file, by default False
    fill_max_sigma : int, optional
        The maximum sigma value for fill, by default 25
    key : str, optional
        The key used to access the KNMI open data platform, by default 'private'
        
    Returns
    -------
    object
        A Rasterio dataset reader for the downloaded file
    """
    
    
    
    from pathlib import Path
    
    api_url = 'https://api.dataplatform.knmi.nl/open-data'
    api_version = 'v1'

    if key == 'public':
        api_key = 'eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImNjOWE2YjM3ZjVhODQwMDZiMWIzZGIzZDRjYzVjODFiIiwiaCI6Im11cm11cjEyOCJ9'
    if key == 'private':
        api_key = 'eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjAyMGI4ZGQwOWM3NDQ3Yzk5MTY4NmFhODRhYzkzMWVkIiwiaCI6Im11cm11cjEyOCJ9'

    dataset_version = '3'
    
    print('download_loc', download_loc)
    
    
    fp = f'{download_loc}/{fn}'.lstrip('/')
    print('fp', fp)
    
    idx = fp.index('.')
    print(f'fp_fill = {fp[:idx]} + _filled + {fp[idx:]}' )
    fp_fill = fp[:idx] + '_filled' + fp[idx:]
    
    print('fp_fill thus is', fp_fill)
    
    if fill:
        try:
            dsr = rasterio.open(fp_fill)
            return dsr
        except:
            print('no succes dsr = rasterio.open(fp_fill)')
            pass
    
    try:
        dsr = rasterio.open(fp)
        print(f'Found {fn}')
        
    except:
        print(f'Could not find {fn}')
        endpoint = f'{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files/{fn}/url'
        get_file_response = requests.get(endpoint, headers={'Authorization': api_key})
        if get_file_response.status_code != 200:
            raise Exception(f'Fail for {endpoint}: status_code is not 200, but {get_file_response.status_code}, reason: "{get_file_response.reason}"')
            sys.exit(1)
    
        download_url = get_file_response.json().get('temporaryDownloadUrl')
        dataset_file_response = requests.get(download_url)
        if dataset_file_response.status_code != 200:
            raise Exception(f'Fail for {download_url}: status_code, but {dataset_file_response.status_code}')

        # Write dataset file to disk
        p = Path(fp)
        p.write_bytes(dataset_file_response.content)
        print(f'Successfully downloaded dataset file to {p}')

        dsr = rasterio.open(fp)
    
    if fill:
        dsr = fill_missing.geotiff(dsr, max_sigma=fill_max_sigma)
        os.remove(fp)
        
    return dsr
    