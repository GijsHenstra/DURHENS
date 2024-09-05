#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:50:40 2021

@author: gijshenstra
"""

import numpy as np
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
import glob

def save_array_as_img(arr, folder, filename='img.png', scale=None):
    scale = int(255 / arr.max())
    arr_scaled = scale * arr
    img = Image.fromarray(np.uint8(arr_scaled), 'L')
    name, extension = filename.split('.')
    extension = '.' + extension
    path = folder / (name + '_scale%i' % scale + extension)
    img.save(path)
    
    # folder = path.rstrip((path.split('/')[-1]))
    height_colorbar(scale, plot=True, savepath=folder / 'colorbar.png')

    # return path
    
def load_img_as_arr(folder, file='img.png', scale=None):
    filename = file.split('.')[0]
    files = glob.glob(str(folder / '*.png'))
    file = [file for file in files if filename in file][0]
    if scale is None:
        try: 
            scale = int(file.rstrip('.png').split('scale')[-1])
        except:
            print('Could not read "scale" from the filename, so set scale of 4.')
            scale = 4
    arr_scaled = np.array(Image.open(file).convert("L"))
    arr = arr_scaled / scale
    return arr

def height_colorbar(scale, plot=False, savepath='colorbar.png'):
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.80, 0.9, 0.2])
    ax1.set_title(f"Colormap for heightdata in {savepath}")
    
    bounds = np.arange(0, 255 / scale, 1 / scale)
    # ticks =  list(np.arange(0, 255 / scale, 10)) + [255 / scale]
    
    norm = mpl.colors.Normalize(vmin=0, vmax=255 / scale)
    # mpl.colorbar.ColorbarBase(ax1, norm=norm, ticks=ticks, cmap=plt.get_cmap('gray'), boundaries=bounds, orientation='horizontal')
    mpl.colorbar.ColorbarBase(ax1, norm=norm, cmap=plt.get_cmap('gray'), boundaries=bounds, orientation='horizontal')
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    if plot:
        plt.show()