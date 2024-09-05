#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:42:01 2020

@author: gijshenstra
"""

import numpy as np


def element_across_street2(cl, el, face, faces_list, F_matrix, dz=1):
    # find set of opposing faces
    if face == 0:
        face_opp = 1
        opp_els = np.where((cl[:, 0] < cl[el, 0]) &
                           (cl[:, 1] == cl[el, 1]) &
                           (cl[:, 2] == cl[el, 2]) &
                           (faces_list[:, face_opp] == 1))[0]
        try:
            el_closest = opp_els[cl[opp_els][:, 0].argsort()][-1]
        except BaseException:
            return None

    if face == 1:
        face_opp = 0
        opp_els = np.where((cl[:, 0] > cl[el, 0]) &
                           (cl[:, 1] == cl[el, 1]) &
                           (cl[:, 2] == cl[el, 2]) &
                           (faces_list[:, face_opp] == 1))[0]
        try:
            el_closest = opp_els[cl[opp_els][:, 0].argsort()][0]
        except BaseException:
            return None

    if face == 4:
        face_opp = 5
        opp_els = np.where((cl[:, 1] < cl[el, 1]) &
                           (cl[:, 0] == cl[el, 0]) &
                           (cl[:, 2] == cl[el, 2]) &
                           (faces_list[:, face_opp] == 1))[0]
        try:
            el_closest = opp_els[cl[opp_els][:, 1].argsort()][-1]
        except BaseException:
            return None

    if face == 5:
        face_opp = 4
        opp_els = np.where((cl[:, 1] > cl[el, 1]) &
                           (cl[:, 0] == cl[el, 0]) &
                           (cl[:, 2] == cl[el, 2]) &
                           (faces_list[:, face_opp] == 1))[0]
        try:
            el_closest = opp_els[cl[opp_els][:, 1].argsort()][0]
        except BaseException:
            return None

    return el_closest


def hw_between_points(cl, el, el_closest, dz):
    # distance from mid to mid
    width = np.linalg.norm(cl[el] - cl[el_closest])
    
    # the distance between the sides is 1 smaller then from mid to mid
    width -= 1
    
    height = cl[el, 2] - dz/2

    return height / width


def calc_hw_map(cl, shape_ds, faces_list, save_dir=None, dz=1):

    hw_map = np.zeros((shape_ds))
    for check_faces in [[0, 4], [1, 5]]:
        for face in check_faces:
            els_in_face = np.where(
                (faces_list[:, 3] == 1) & (faces_list[:, face] == 1))[0]

            for el in els_in_face:
                el_closest = element_across_street2(
                    cl, el, face, faces_list, dz)

                hw = hw_between_points(cl, el, el_closest, dz)

                if el_closest is None:
                    continue

                x_start = int(
                    np.min([cl[el, 0] - 0.5, cl[el_closest, 0] - 0.5]))
                x_end = int(np.max([cl[el, 0] - 0.5, cl[el_closest, 0] - 0.5]))

                y_start = int(
                    np.min([cl[el, 1] - 0.5, cl[el_closest, 1] - 0.5]))
                y_end = int(np.max([cl[el, 1] - 0.5, cl[el_closest, 1] - 0.5]))
                
                if face==0:
                    x_start = x_start+1
                    x_end = x_end-1
                if face==1:
                    x_start = x_start+1
                    x_end = x_end-1
                if face==4:
                    y_start = y_start+1
                    y_end = y_end-1
                if face==5:
                    y_start = y_start+1
                    y_end = y_end-1
                # hw_map[x_start:x_end + 1, 
                #        y_start:y_end + 1] = np.maximum(hw, hw_map[x_start:x_end + 1, 
                #                                                   y_start:y_end + 1])
                
                # if x_start == x_end:
                #     x_end = x_end + 1
                    
                # if y_start == y_end:
                #     y_end = y_end + 1
                
                hw_map[x_start:x_end + 1, 
                       y_start:y_end + 1] = np.maximum(hw, hw_map[x_start:x_end + 1, 
                                                                  y_start:y_end + 1])                                                  
              
    hw_map = np.transpose(hw_map)
    if save_dir is not None:
        np.savez(save_dir / 'hw', hw_map=hw_map)
    return hw_map
