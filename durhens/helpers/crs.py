#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:46:20 2021

@author: gijshenstra
"""

from shapely.geometry import Point
import geopandas as gpd


def transform(xy_in, crs_in="EPSG:28992", crs_out="EPSG:4326"):
    """Tranform a flat x-y tuple to another coordinate system."""

    points = [Point(xy_in[x:x + 2]) for x in range(0, len(xy_in), 2)]

    bbox_in = gpd.GeoDataFrame({'geometry': points}, crs=crs_in)
    bbox_out = bbox_in.to_crs(crs_out)

    xy_out = [coord for xy in bbox_out.geometry for coord in xy.coords[0]]
    return xy_out