#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:10:08 2023

@author: gijshenstra
"""

import math

def direct_normal():
    """https://klimapedia.nl/wp-content/uploads/2013/06/W-9_zonnestraling_en_zonstralingsgegevens.pdf"""
    
    
    q_sw_dir_norm = a-b*np.exp(-c * np.cos(zen_rad))
    
def diffuse_horizontal():
    """"https://klimapedia.nl/wp-content/uploads/2013/06/W-9_zonnestraling_en_zonstralingsgegevens.pdf"""
    q_sw_diff_hor = 1/3 * (q_sw_extraterrestrial - q_sw_dir_norm) 
    
def direct_extraterrestrial(day_of_year):
    """https://klimapedia.nl/wp-content/uploads/2013/06/W-9_zonnestraling_en_zonstralingsgegevens.pdf"""
    q_sw_extraterrestrial = 1355 * (1 - 0.033*math.sin((day_of_year-93)/(365) * 360))