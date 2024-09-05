#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rasterio
# import os
# import glob
import numpy as np
# import rasterstats as rs
# import geopandas as gpd
# import cv2
# import scipy
import math
# import shapely
# import requests
import itertools

# import pandas as pd
# import warnings
import rasterio.mask
# import matplotlib.pyplot as plt
# import osmnx as ox

# from io import BytesIO
# from PIL import Image
# from affine import Affine
# from shapely import affinity
# from rasterio.mask import mask
# from rasterio.merge import merge
# from rasterio.plot import show
# from owslib.wcs import WebCoverageService
# from shapely.geometry import Point
# from requests import Request

# from owslib.wms import WebMapService

# import geodata_functions as gf

def reverse_order(bbox):
    return tuple(np.array(bbox).reshape(2, 2)[:, ::-1].flatten())

def zoomout(bbox):
    """Increase bounding box (bbox) size so that area remains intact under rotation and cropping."""
    gridsize = (bbox[2] - bbox[0])
    border_thickness = int(math.ceil(gridsize * (math.sqrt(2) - 1)))
    bbox_zoomout = [bbox[0] - border_thickness,
                    bbox[1] - border_thickness,
                    bbox[2] + border_thickness,
                    bbox[3] + border_thickness]
    
    return bbox_zoomout

def to_area(bbox, res):
    width = (bbox[2] - bbox[0]) / res
    height = (bbox[3] - bbox[1]) / res
    return height * width

def equals(*args):
    bbox_lst = []
    for item in args:
        if isinstance(item, rasterio.io.DatasetReader):
            bbox_lst.append(tuple(item.bounds))
            
        elif isinstance(item, (list, tuple)) and len(item) == 4 :
            bbox_lst.append(tuple(item))
            
        else:
            ValueError(f'Invalid input: {item}')
        
    # print('bbox', bbox)
        
    return all(bbox == bbox_lst[0] for bbox in bbox_lst)


def fits_in(method, *args):
    
    print(args)
    bbox_lst = []
    for item in args:
        if isinstance(item, rasterio.io.DatasetReader):
            bbox_lst.append(tuple(item.bounds))
            
        elif isinstance(item, (list, tuple, np.ndarray)) and len(item) == 4 :
            bbox_lst.append(tuple(item))
            
        else:
            ValueError(f'Invalid input: {item}')
        
        
    print('bbox_lst', bbox_lst)
        
    if method=='equal':
        return all(bbox == bbox_lst[0] for bbox in bbox_lst)
    
    if method=='1in2':
        return all([bbox_lst[0][0] > bbox_lst[1][0], 
                    bbox_lst[0][1] > bbox_lst[1][1], 
                    bbox_lst[0][2] < bbox_lst[1][2], 
                    bbox_lst[0][3] < bbox_lst[1][3]])


def to_cardinalpoints(bbox, crs_in='EPSG:28992', crs_out="EPSG:4326"):
    """Transform """
    if isinstance(bbox, rasterio.coords.BoundingBox):
        xy_southwest = [bbox.left, bbox.bottom]
        xy_northeast = [bbox.right, bbox.top]
        
    elif isinstance(bbox, (tuple, list)):
        xy_southwest = bbox[:2]
        xy_northeast = bbox[2:]
    
    west, south = crs.transform(xy_southwest,
                                           crs_in=crs_in,
                                           crs_out=crs_out)
        
    east, north = crs.transform(xy_northeast,
                                           crs_in=crs_in,
                                           crs_out=crs_out)
    
    return north, south, east, west

def to_mapindex(bbox, mapindex_table, mapindex_crs='EPSG:28992', bbox_crs='EPSG:28992'):
    """Find which maps the bbox are in"""
    x_points = [bbox[0], bbox[2]]
    y_points = [bbox[1], bbox[3]]
    points = list(itertools.product(x_points, y_points))
    
    row_select = np.zeros(len(mapindex_table), dtype=bool)
    
    for i, point in enumerate(points):
        point_in_row = ((mapindex_table.minx < point[0]) & 
                        (mapindex_table.maxx > point[0]) &
                        (mapindex_table.miny < point[1]) & 
                        (mapindex_table.maxy > point[1]))
        row_select = row_select | point_in_row
        
    mapindex = mapindex_table[row_select].index
    
    if mapindex.empty:
        raise ValueError(f'No card found for specified bbox {bbox}')
    
    return mapindex.values.astype(str)

