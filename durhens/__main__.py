"""
Dutch URban Heated Environment Numerical Simulation

Please find information on the simulation setup in the report:
'Enhanced Radiation Modelling for Urban Surface Temperature Analysis - Finding how urban properties affect the surface energy balance and thereby influence surface temperatures in idealized and real-world geometries'
    
@author: Gijs Henstra , 2024
"""



import warnings
import sys
import numpy as np
import pandas as pd
import math
import cv2
import os
import datetime
import sparse
import time
import pickle
import scipy
import calendar
import rasterio
import sklearn
import itertools
import imageio
import glob
import yaml
import knmi

from pathlib import Path
from functools import reduce
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Custom modules
from helpers import height_width
from helpers import help_functions as hf
from helpers import view_factor_jit as vf_j

import physics
from plot import plot2d, plot3d, bar
from physics import solar_angles
from physics import utci
from helpers import named_colors
from helpers import get_data
from helpers import offline_data
from helpers import online_data

# plt.style.use('ggplot')

# Set minimal plotting resolution
plt.rcParams['figure.dpi'] = 300
plt.style.use('bmh')

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

data_names = {
    "UHI": "Urban Heat Island",
    "T_mr": "Mean Radiant Temperature",
    "q_sw_abs": "Absorbed shortwave irradiance",
    "q_sw": "Shortwave irradiance",
    "sensor_sw_in": "Incoming shortwave irradiance",
    "sensor_lw_in": "Incoming longwave irradiance",
    "sensor_lw_in_sky": "Incoming longwave irradiance from sky",
    "sensor_lw_in_els": "Incoming longwave irradiance from elements",
    "sensor_sw_in_sky": "Incoming shortwave irradiance from sky",
    "sensor_sw_in_els": "Incoming shortwave irradiance from elements"}

DEMOS_DIR = ('demos')

show_all_plots = True


def list_to_dataseries(df, var, data):
    
    if isinstance(data, (float, int)):
        df[var] = data
        
    elif isinstance(data, (list, np.ndarray)):
        for i, date in enumerate(np.unique(df.index.date)):
            try:
                df.loc[(df.index.date == date), var] = data[i]
            except:
                pass
            
    else:
        ValueError(f'Wrong type for data: {type(data)}')

# @profile
def main(input_0, **kwargs):
    
    FOLDER = DEMOS_DIR / Path(f'{input_0}')
    
    CONFIG_FILE = FOLDER / Path('config.yaml')

    if not CONFIG_FILE.exists():
        raise Exception(f'File {CONFIG_FILE} not exist.')

    with CONFIG_FILE.open('r') as fp:
        config_temp = yaml.safe_load(fp)
        
    print('kwargs', kwargs)
        
    # overwrite variables from config file with keyword arguments in command line
    for key, value in kwargs.items():
        # make same type as original value in config file
        try:
            try:
                # value = type(config_temp[key])(eval(value)) # 7 sept 23
                value = eval(value)
            except:
                value = type(config_temp[key])(value)
            
            if isinstance(value, dict):
                print(f'Overwriting {config_temp[key]} with {value}')
                config_temp[key].update(value)
            else:
                print(f'Overwriting {key}={config_temp[key]} with {key}={value}')
                config_temp.update({key: value})
        except KeyError:
            pass
        
    CONFIG = config_temp

    SIMULATE_AVERAGES = bool(CONFIG['AVERAGE_YEARS'])
    skip_steps = CONFIG.get('SKIP_STEPS', False)

    UHI_BOOL = CONFIG['CALC_UHI']

    while True:
        if FOLDER.exists():
            break

        user_input = input(
            'Specified FOLDER can not be found, enter "new" to create a new or press enter to retry...\n')
        if user_input == 'new':
            os.mkdir(FOLDER)

    GRIDSIZE = CONFIG['GRIDSIZE']  
    FORCE_RECALC_ENVIR = CONFIG['FORCE_RECALC_ENVIR'] 
    CHECK_TIMESTEP = CONFIG['CHECK_TIMESTEP'] 
    
    if CONFIG['DAYS'] is None:
        CONFIG['DAYS'] = 31  # 1 year: 366
        
    # Date and Time settings
    STEPS_PER_DAY = int(24 * 60 * 60 / CONFIG['T_STEP'])
    FREQ_STR = str(np.min([CONFIG['T_STEP'], 15 * 60])) + 's'
    start_dates = []
    end_dates = []
    dates_lst = []
    dates = pd.DatetimeIndex([])
    for start_date in CONFIG['START_DATES']:
        start_date = datetime.datetime(*start_date)
        start_dates.append(start_date)
        end_date = start_date + datetime.timedelta(days=CONFIG['DAYS']-1, hours=23, minutes=59)
        end_dates.append(end_date) 
        
        _dates = pd.date_range(start=start_date, end=end_date)
        dates_lst.append(_dates)
        dates = dates.append(_dates)
        # dates.append(date_obj)
        
    # dates = pd.DatetimeIndex(dates_lst)
        
    

    map_colors = hf.load_dict('constants/map_colors.txt')

    if not glob.glob('constants/map_colors_plot.*'):
        named_colors.plot_colortable(
            map_colors,
            save_path='constants/map_colors_plot',
            png=True)

    map_colors_gradient = map_colors.copy()
    map_colors_gradient['trees'] = [map_colors['grass'], [
        128 / 255, 173 / 255, 83 / 255], [128 / 255, 173 / 255, 83 / 255]]

    # %% Load topographic data

    # max distance between elements for which view factors must be calculated
    # in meters
    MAX_RAY_ELS = int(CONFIG['MAX_RAY_DIST'] / CONFIG['RES'])

    FOLDER_RES = FOLDER
    
    hw = CONFIG['HW']
        
    if CONFIG['ADDRESS'] == 'custom':
        
        FOLDER_RES = FOLDER_RES / f'hw{hw}'
        
        if not os.path.isdir(FOLDER_RES):
            os.mkdir(FOLDER_RES)
            
    FOLDER_RES = FOLDER_RES / ('res' + str(CONFIG['RES']))
    if not FOLDER_RES.exists():
        os.mkdir(FOLDER_RES)

    data_dir = './data'
    
    # %%% Load custom data
    if CONFIG['ADDRESS'] == 'custom':
        ADDRESS = CONFIG['ADDRESS']
        load_resx, load_resy = 1, 1

        # ---- Get the local digital surface map (DSM)
        while True:
            try:
                dsm = offline_data.load_img_as_arr(
                    folder=FOLDER, file='dsm.png')
                dsm *= hw
                break
            except (IndexError, FileNotFoundError):
                print('No file found "dsm_*.png". Please insert the file.')
                input('Press enter to retry...')

        # ---- Create masks from DSM
        masks = {}

        masks.update({"buildings": dsm >= 4})
        masks.update({"trees": (0 < dsm) & (dsm < 4)})
        masks.update({"grass": dsm == np.min(dsm)})
        masks.update({"concrete": np.zeros(dsm.shape, dtype=bool)})
        masks.update({"water": np.zeros(dsm.shape, dtype=bool)})
        masks.update({"roads": np.zeros(dsm.shape, dtype=bool)})
        masks.update({"open surface": np.zeros(dsm.shape, dtype=bool)})
        if not type(CONFIG['ROTATION']) in (str, int):
            if not CONFIG['ROTATION']:
                ROT = 0
        latitude, longitude = CONFIG['SOLAR_ANGLES'].get('default_latitude_longitude', [52, 4.375])
        if CONFIG['ADDRESS_STR'] is None:
            address_str = FOLDER.parts[-1]
        else:
            address_str = CONFIG['ADDRESS_STR']

        sat_img = None
        
        if not isinstance(CONFIG['ROTATION'], (int, float)):
            raise ValueError('When loading a custom map, specify the "ROTATION."')
        else:
            ROT = CONFIG['ROTATION']

    MASKS_MAP_PATH = f'{FOLDER}/map_discrete{"_no_trees" if not CONFIG["TREES"] else ""}.png'

    # %%% Load real world map
    if CONFIG['ADDRESS'] != 'custom':
        load_resx, load_resy = 0.5, 0.5
        resx = resy = resz = CONFIG['RES']

        # ---- Get location information
        while True:
            try:
                x, y, ADDRESS, address_location = get_data.address(
                    address=CONFIG['ADDRESS'], crs_out="EPSG:28992")
                address_str = online_data.location2address_str(
                    address_location)
                break
            except ValueError:
                input(
                    f'No address found for "{ADDRESS}". Change ADDRESS in {CONFIG_FILE} \n Then press enter...') or ADDRESS
                with CONFIG_FILE.open('r') as fp:
                    CONFIG = yaml.safe_load(fp)
                ADDRESS = CONFIG['ADDRESS']

        latitude = address_location.latitude
        longitude = address_location.longitude

        print('Address: ' + address_str,
              'at lat, long: ' + str(latitude) + ', ' + str(longitude))


        # ---- Get elevation and satellite maps and generate masks for surface types
        xmin, xmax = int(x - GRIDSIZE / 2), int(x + GRIDSIZE / 2)
        ymin, ymax = int(y - GRIDSIZE / 2), int(y + GRIDSIZE / 2)
        bbox = [xmin, ymin, xmax, ymax]
        
        # try to load map data
        try:
            path_elevation = (
                f'{FOLDER_RES}/map_data{"_no_trees" if not CONFIG["TREES"] else ""}')
            with open(f'{path_elevation}.pkl', 'rb') as f:
                dsm, masks, rotation_load, bag_gdf = pickle.load(f)
                if not type(CONFIG['ROTATION']) in (str, int):
                    if not CONFIG['ROTATION']:
                        ROT = 0 
                    else:
                        ROT = rotation_load

            sat_img_HD = np.moveaxis(
                rasterio.open(f'{FOLDER}/sat_img_HD_crop.tif').read(), 0, -1)
            sat_img = np.moveaxis(
                rasterio.open(f'{FOLDER}/sat_img_crop.tif').read(), 0, -1)

        # if map data can not be found or is invalid, (down)load new data 
        except (FileNotFoundError, ValueError, rasterio.RasterioIOError):
            dsm_unblurred, masks, ROT, bag_gdf = get_data.elevation_and_masks(
                bbox=bbox,
                save_dir=FOLDER,
                crs_out='EPSG:4326',
                load_res=load_resx,
                gridsize=GRIDSIZE,
                rotation=CONFIG['ROTATION'],
                address=address_str,
                ahn_version=CONFIG.get('AHN_VERSION', 2),
                # ahn_method='sheets',
                ahn_method='WCS' if CONFIG['AHN_VERSION'] == 4  else 'sheets',
                force_download=False,
                plot_bag=True,
                plot_bag_elevation=False,
                plot_dsm=True,
                plot_dtm=True,
                plot_energy=False,
                plot_roads=True)

            if CONFIG['BLUR_ELEVATION']:
                # perform a blurring operation in order to smooth out 
                # irregularities in the loaded elevation data
                dsm = cv2.medianBlur(np.float32(dsm_unblurred), ksize=CONFIG['BLUR_ELEVATION'])
            else:
                dsm = dsm_unblurred
                if CONFIG['RES'] == 0.5:
                    print('\n\n Elevation map is used unblurred, in combination with the fine resolution of 0.5m, this can lead a jagged landscape. \n\n\n') 

            if not CONFIG['TREES']:
                # set height to zero outside buildings mask
                dsm[~masks['buildings']] = 0
                masks['trees'] = np.zeros(masks['buildings'].shape).astype(bool)

            if CONFIG['TREES']:
                masks["roads"][masks["trees"]] = False
                masks["open surface"][masks["trees"]] = False

            # get satellite image on 0.25m resulution
            try:
                sat_img_HD = get_data.satellite_image(
                    bbox,
                    FOLDER,
                    load_res=0.25,
                    rotation=ROT,
                    fn='sat_img_HD',
                    layers=['2016_ortho25'])
                plt.imshow(sat_img_HD)
                plt.show(block=False)
            except BaseException:
                sat_img_HD = None

            # get satellite image on 0.5m resulution
            sat_img = get_data.satellite_image(
                bbox,
                save_dir=FOLDER,
                load_res=0.5,
                rotation=ROT,
                fn='sat_img',
                layers=['2016_ortho25'])

            plt.imshow(sat_img)
            plt.show(block=False)

            # define colors in the satellite img
            sat_palette = { 
                'concrete': [153, 155, 151], 
                'grass': [92, 113, 94]}
            masks_sat = hf.img_to_masks_sat(sat_img, sat_palette)

            masks.update({"grass": (masks['open surface'] * masks_sat['grass']) + masks['grass'],  
                          "concrete": (masks['open surface'] * masks_sat['concrete']) * ~masks['grass']})

            masks["open surface"] = np.zeros(np.shape(dsm))

            masks['grass'][(masks['roads'] +
                            masks['grass'] +
                            masks['water'] +
                            masks['buildings'] +
                            masks['trees']) == 0] = True

            # save elevation and masks data
            with open(f'{path_elevation}.pkl', 'wb') as f:
                pickle.dump([dsm, masks, ROT, bag_gdf], f)

        if sat_img_HD is not None:
            sat_scale = int(sat_img_HD.shape[0] / dsm.shape[0])

            interp_func = scipy.interpolate.interp2d(
                np.arange(
                    dsm.shape[0]), np.arange(
                    dsm.shape[1]), dsm, kind='cubic')
            dsm_HD = interp_func(x=np.arange(0, dsm.shape[0], 1 / sat_scale),
                                 y=np.arange(0, dsm.shape[1], 1 / sat_scale))

    plt.imshow(dsm, cmap='Greys_r', vmin=0, vmax=50)
    plt.title('Elevation map: Digital Surface Model')
    cbar = plt.colorbar()
    cbar.set_label('elevation [m]', rotation=270)
    plt.savefig(FOLDER_RES / 'dsm_colorbar.png')
    plt.show()

    # %%% Generate images from masks

    if os.path.isfile(MASKS_MAP_PATH):
        update_masks_map = False
    else:
        update_masks_map = True

    if not os.path.isfile(MASKS_MAP_PATH):
        map_img = plot2d.masks2img(
            masks,
            map_colors,
            plot=True,
            save_dir=MASKS_MAP_PATH,
        )
    if update_masks_map:
        input('Adjust ' + MASKS_MAP_PATH.split('/')[-1] + ' file as needed.\n'
              + 'Then, press enter...')

    map_img = cv2.cvtColor(cv2.imread(MASKS_MAP_PATH), cv2.COLOR_BGR2RGB)
    masks = hf.img_to_masks(map_img, map_colors,
                      load_dir=MASKS_MAP_PATH, showfig=False)

    build_frac, imperv_frac = hf.get_mask_stats(masks)
    print(f'Buildings fraction: {build_frac}')
    print(f'Impervious fraction: {imperv_frac}')

    reduce_res = int(CONFIG['RES'] / load_resx)

    map_img_gradient = plot2d.masks2img( # TODO rename
        masks=masks,
        map_colors=map_colors_gradient,
        height2d=dsm,
        plot=False,
        # save_dir=MASKS_MAP_PATH,
    )

    if CONFIG['TREES']:
        FOLDER_RES_TREES = FOLDER_RES / 'trees'
    if not CONFIG['TREES']:
        FOLDER_RES_TREES = FOLDER_RES / 'no_trees'
    if not os.path.isdir(FOLDER_RES_TREES):
        os.mkdir(FOLDER_RES_TREES)
        
    FOLDER_RES_TREES_RAY = FOLDER_RES_TREES / f'max_dist_m{int(CONFIG["MAX_RAY_DIST"])}/'
    if not os.path.isdir(FOLDER_RES_TREES_RAY):
        os.mkdir(FOLDER_RES_TREES_RAY)

    elev = dsm

    # Surface of the faces of the elements
    A = CONFIG['RES'] ** 2

    # Reduce the resolution of the elevation map by Box sampling
    elev_lr, = hf.reduce_res_mean(reduce_res, elev)

    # Set lowest elevation to zero
    elev_lr -= elev_lr.min()
    
    # Set the street_height to the lowest height (=0)
    street_height = elev.min()

    SHAPE_2D = elev.shape
    SHAPE_2D_LR = elev_lr.shape

    SHAPE_3D_LR = SHAPE_2D_LR + \
        (int(max((11, np.ceil(elev_lr / CONFIG['RES']).max() + 3))),)

    masks_map_lr_path = FOLDER_RES_TREES_RAY / 'masks_lr.png'
    if os.path.isfile(masks_map_lr_path):
        print("Loading lowres .png image to masks")

        img = cv2.cvtColor(cv2.imread(str(masks_map_lr_path)), cv2.COLOR_BGR2RGB)

        masks_lr = hf.img_to_masks(
            img, map_colors, load_dir=masks_map_lr_path, showfig=False) 
    else:
        # Reducing the resolution of the object masks by counting the ocurrence of
        # all masks per downsampeled pixel
        masks_lr = hf.reduce_res_mask(masks, reduce_res, shape=SHAPE_2D_LR)

    map_img_lr = plot2d.masks2img(
        masks_lr,
        map_colors,
        plot=False,
        save_dir=masks_map_lr_path,
    )

    if not CONFIG['TREES']:
        elev_lr[~masks_lr['buildings']] = 0

    area_els = np.shape(elev_lr)[0] * np.shape(elev_lr)[1]

    percentage_buildings = masks["buildings"].sum() / area_els * 100
    percentage_grass = masks["grass"].sum() / area_els * 100
    percentage_concrete = masks["concrete"].sum() / area_els * 100
    percentage_trees = masks["trees"].sum() / area_els * 100
    percentage_roads = masks["roads"].sum() / area_els * 100
    percentage_water = masks["water"].sum() / area_els * 100

    map_img_gradient_lr = plot2d.masks2img(
        masks=masks_lr,
        map_colors=map_colors_gradient,
        height2d=elev_lr,
        plot=True,
        # save_dir=MASKS_MAP_PATH,
    )

    #  Mask which sensors will be used to log results
    if os.path.isfile(FOLDER / 'sensor_mask.png'):
        update_sensor_mask = False
    else:
        update_sensor_mask = True

    border_mask = hf.create_circular_mask(elev.shape[0], elev.shape[1], radius=elev.shape[1]*(3/8))
    

    
    try:
        sensor_mask_img = cv2.imread(str(FOLDER / 'sensor_mask.png'), cv2.IMREAD_GRAYSCALE)
        if sensor_mask_img is None:
            raise FileNotFoundError

    except FileNotFoundError:
        sensor_mask_img = map_img * (0.2 + 0.8 * border_mask)[..., np.newaxis]

        plt.imshow(sensor_mask_img / 255)
        plt.title('sensor_mask.png')
        plt.show(block=False)
        
        imageio.imwrite(FOLDER / 'sensor_mask.png', sensor_mask_img / 255)
        
        if update_sensor_mask:
            input('Now you can set a custom mask for the location of UHI calculation. \n Press enter to continue...')
        sensor_mask_img = cv2.imread(str(FOLDER / 'sensor_mask.png'), cv2.IMREAD_GRAYSCALE)

    sensor_mask_bools, contours = hf.img_to_threshold_mask(sensor_mask_img)

    sensor_mask_bools_lr = hf.reduce_res_mean(reduce_res, sensor_mask_bools)[0]
    
    
    sensor_mask_img_lr = map_img_lr * \
        (0.2 + 0.8 * sensor_mask_bools_lr)[..., np.newaxis]
    imageio.imwrite(FOLDER_RES_TREES_RAY / 'sensor_mask_lr.png', sensor_mask_img_lr)
    sensor_mask_img_lr = cv2.imread(
        str(FOLDER_RES_TREES_RAY / 'sensor_mask_lr.png'),
        cv2.IMREAD_GRAYSCALE)

    sensor_mask_bools_lr, contours_lr = hf.img_to_threshold_mask(sensor_mask_img_lr)


    # masks_sensor_lr, = hf.reduce_res_mean(reduce_res, mask_sensors)

    # image_cont = cv2.drawContours(
    #     map_img_gradient.copy(), contours, 0, (1, 0, 0), 1)

    # contours_lr, _ = cv2.findContours(mask_sensors_lr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # %% Calculate coordinates, faces and identify street and building elements

    envir = dict()
    envir_no_trees = dict()
    envir_vf = dict()
    sensors = dict()
    sensors_vf = dict()

    start = time.time()

    


    # %%% Calculate or load environment topology data

    # Compare the above calculated coordinates list with the (if present)
    # coordinates list from the data. If they are equal, all other setupdata
    # will be loaded, if not it must be calculated.
    # ---- Load environment topology data
    try:
        # if np.array_equal(envir_no_trees['coo'], setup_data['cl_no_trees']):
        if not FORCE_RECALC_ENVIR:

            envir = np.load(
                f'{FOLDER_RES_TREES_RAY}/envir.npy',
                allow_pickle=True).item()
            envir_no_trees = np.load(
                f'{FOLDER_RES_TREES_RAY}/envir_no_trees.npy',
                allow_pickle=True).item()

            print("Loaded environment topology")

    except BaseException:
        FORCE_RECALC_ENVIR = True

    # ---- Calculate environment topology data
    if FORCE_RECALC_ENVIR:
        print("Calculating coordinates list...")

        # Calculate the coordinates list (coo), "orig" indicates that all
        # elements are present. Then, it is elavuated which elements are the
        # inner elements and these elements are then removed in order to save
        # computation time.
        envir_no_trees['coo_with_inside'] = hf.elevationmap_to_coo(
            np.where(masks_lr['buildings'], elev_lr, 0), street_height, CONFIG['RES'])
        envir_no_trees['faces_lst_with_inside'], mask_outer_elements = hf.identify_els_outer_faces(
            envir_no_trees['coo_with_inside'], street_height)  # remove inner elements from cl
        envir_no_trees['coo'], envir_no_trees['faces_lst'] = envir_no_trees['coo_with_inside'][
            mask_outer_elements], envir_no_trees['faces_lst_with_inside'][mask_outer_elements]

        envir['coo_with_inside'] = hf.elevationmap_to_coo(
            (np.where(masks_lr['buildings'], elev_lr, 0) +
             np.where(masks_lr['trees'], elev_lr, 0)),
            street_height,
            CONFIG['RES'])
        faces_lst_orig, mask_outer_elements = hf.identify_els_outer_faces(
            envir['coo_with_inside'], street_height=street_height)  # remove inner elements from cl
        envir['coo'] = envir['coo_with_inside'][mask_outer_elements]
        envir['faces_lst'] = faces_lst_orig[mask_outer_elements]
        envir['faces_idcs'] = hf.calc_faces_idcs(
            envir['coo'], envir['faces_lst'])
        
        # in ray_tracing, an error occurred that was resolved by making the envir['faces_idcs'] a np array.
        # TypeError: list indices must be integers or slices, not list
        envir['faces_idcs'] = np.array(envir['faces_idcs'])

        envir_no_trees['faces_idcs'] = hf.calc_faces_idcs(
            envir_no_trees['coo'], envir_no_trees['faces_lst'])

        envir['coo_int'] = (
            envir['coo'] if CONFIG['TREES'] else envir_no_trees['coo']).astype(int)
        envir['idcs_3D'] = hf.coo_to_idcs3D(coo_int=envir['coo_int'],
                                         shape=SHAPE_3D_LR)
        envir['idcs_3D_masked'] = np.ma.masked_equal(envir['idcs_3D'], -1)
        coords_all_ngbrs = hf.coords_all_adjacent_elements(
            coo_int=envir['coo_int'], shape=envir['idcs_3D'].shape)
        
        envir['idcs_ngbrs'] = hf.coos_to_idcs(coos=np.array(coords_all_ngbrs), idcs_3D=envir['idcs_3D'])

        np.save(f'{FOLDER_RES_TREES_RAY}/envir.npy', envir)
        np.save(f'{FOLDER_RES_TREES_RAY}/envir_no_trees.npy', envir_no_trees)

    # make a numpy matrix with neighbour elements 
    idcs_ngbrs_mtx = np.array(
        list(
            itertools.zip_longest(
                *np.array(envir['idcs_ngbrs'], dtype=object),
                fillvalue=None))).T
    
    idcs_ngbrs_mask = (idcs_ngbrs_mtx == None)
    
    idcs_ngbrs_mtx[idcs_ngbrs_mtx == None] = 0
    idcs_ngbrs_mtx = idcs_ngbrs_mtx.astype(int)
    idcs_ngbrs_ma = np.ma.array(idcs_ngbrs_mtx, mask=idcs_ngbrs_mask)
    
    nr_ngbrs = (~idcs_ngbrs_ma.mask).sum(1)

    if not CONFIG['TREES']:
        envir['coo'] = envir_no_trees['coo']
        envir['faces_lst'] = envir_no_trees['faces_lst']
        envir['faces_idcs'] = envir_no_trees['faces_idcs']

    types_dict = {"buildings": masks_lr["buildings"],
                  "trees": masks_lr["trees"],
                  "roads": masks_lr["roads"],
                  "grass": masks_lr["grass"],
                  "concrete": masks_lr["concrete"],
                  "water": masks_lr["water"],
                  }
    
    envir['idcs_dict'] = hf.masks_to_idcs_dict(
        envir['idcs_3D'],
        types_dict,
        envir['faces_idcs'])

    masks_lst = {}

    for surf_type in envir['idcs_dict']:
        masks_lst.update({surf_type: hf.idcs_to_booleanlist(
            len(envir['coo']), envir['idcs_dict'][surf_type])})
        
    front_dens_tree, front_dens_building, front_dens_total = hf.frontal_areas(
       idcs_dict=envir['idcs_dict'],
       faces_idcs=envir['faces_idcs'],
       # H=elev_lr[elev_lr > 2].mean(),  
       area_map=area_els * A, 
       wind_dir_deg=0, 
       area_face=CONFIG['RES']*CONFIG['RES'], 
       # height=np.arange(envir['idcs_3D'].shape[2])*CONFIG['RES'],
       _print=True) 

    # %%% Visualize elevation data with map colors

    masks_3d = {}
    masks_3d.update({"height": hf.array2d_to_array3d(
        np.ones((len(envir['coo']))), envir['coo'], shape=SHAPE_3D_LR).astype(bool)})

    for surf_type in masks_lst:
        masks_3d.update({surf_type: hf.array2d_to_array3d(
            masks_lst[surf_type], envir['coo'], shape=SHAPE_3D_LR).astype(bool)})

    height_mask3d = hf.array2d_to_array3d(
        np.ones((len(envir['coo']))), envir['coo'], shape=SHAPE_3D_LR).astype(bool)

    elev_round = hf.roundto(elev, CONFIG['RES'])
    elev_round_lr = hf.roundto(elev_lr, CONFIG['RES'])

    topology_plot = True
        
    # ---- 2.5D Barplot of masks
    if 'bar' in CONFIG['PLOT_TOPOLOGY']:
        start = time.time()
        if not os.path.isfile(f'{FOLDER_RES_TREES_RAY}/masks_3d.png'):
            bar.plot(
                height2d=elev_lr,
                img=map_img_gradient_lr,
                gridsize=GRIDSIZE,
                res=CONFIG['RES'],
                view_zen=60,
                view_azi=-ROT,
                colorbar=False,
                save_path=f'{FOLDER_RES_TREES_RAY}/masks_3d.png',
                dpi=500,
                saveimg=True,
            )

        if not os.path.isfile(f'{FOLDER}/masks_3d_HD.png'):
            print('Plot full-res 3d masks image')
            start = time.time()

            bar.plot(
                height2d=elev,
                img=map_img_gradient,
                gridsize=GRIDSIZE,
                res=CONFIG['RES'],
                save_path=f'{FOLDER}/masks_3d_HD',
                colorbar=False,
                saveimg=True,
                dpi=500,
                # view_zen=45,
                view_zen=60,
                view_azi=-ROT,
            )

        if not os.path.isfile(
                f'{FOLDER}/sat_3d_HD.png') and sat_img is not None and sat_img_HD is not None and sat_img_HD.shape[0] < 500:
            print('Plot full-res 3d satellite image')
            bar.plot(
                height2d=dsm_HD,
                img=sat_img_HD,
                gridsize=GRIDSIZE,
                res=0.25,
                view_zen=60,
                view_azi=-ROT,
                colorbar=False,
                save_path=f'{FOLDER}/sat_3d_HD',
                dpi=500,
                saveimg=True,
            )
        else:
            print('Skipped full-res 3d satellite image')

        rotate_map = True
        if rotate_map:
            fn_rotatemap = FOLDER / 'rotate map'
            if not os.path.isdir(fn_rotatemap):
                os.mkdir(fn_rotatemap)
                for view_azi in tqdm(np.linspace(0, 360, 180+1)):
                    bar.plot(
                        height2d=elev_round_lr,
                        img=map_img_gradient_lr,
                        gridsize=GRIDSIZE,
                        res=CONFIG['RES'],
                        save_path=str(fn_rotatemap / ('azi' + str(int(view_azi)))),
                        colorbar=False,
                        saveimg=True,
                        dpi=300,
                        view_zen=60,
                        view_azi=view_azi)

            print('Topology plot, HighRes: ' + hf.duration(start))
        else:
            print('Map too large for full-res bar plot')

        print('Topology plot, LowRes: ' + hf.duration(start))

    # ---- 3D Voxelsplot of masks
    day = '2001-08-26'

    terrain_hr = {
        "roads": {
            "data_proj": masks["roads"],
            "color": map_colors_gradient["roads"],
            "mask": masks["roads"]},
        "grass": {
            "data_proj": masks["grass"],
            "color": map_colors_gradient["grass"],
            "mask": masks["grass"]},
        "concrete": {
            "data_proj": masks["concrete"],
            "color": map_colors_gradient["concrete"],
            "mask": masks["concrete"]},
        "trees": {
            # "data_proj": trees_elev,
            "data_proj": elev,
            "color": map_colors_gradient["trees"],
            "mask": masks["trees"] if CONFIG['TREES'] else np.zeros(
                masks["trees"].shape,
                dtype=bool)},
        "buildings": {
            # "data_proj": buildings_elev,
            "data_proj": elev,
            "color": map_colors_gradient["buildings"],
            "mask": masks["buildings"]},
        "water": {
            "data_proj": masks["water"],
            "color": map_colors_gradient["water"],
            "mask": masks["water"]},
    }

    terrain_lr = {
        "roads": {
            "data_proj": masks_lr["roads"],
            "color": map_colors_gradient["roads"],
            "mask": masks_lr["roads"],
            "mask3d": masks_3d["roads"]},
        "grass": {
            "data_proj": masks_lr["grass"],
            "color": map_colors_gradient["grass"],
            "mask": masks_lr["grass"],
            "mask3d": masks_3d["grass"]},
        "concrete": {
            "data_proj": masks_lr["concrete"],
            "color": map_colors_gradient["concrete"],
            "mask": masks_lr["concrete"],
            "mask3d": masks_3d["concrete"]},  # new
        "trees": {
            "data_proj": np.where(masks_lr['trees'], elev_lr, 0),
            "color": map_colors_gradient["trees"],
            "mask": masks_lr["trees"] if CONFIG['TREES'] else np.zeros(
                masks_lr["trees"].shape,
                dtype=bool),
            "mask3d": masks_3d["trees"]},
        "buildings": {
            "data_proj": np.where(masks_lr['buildings'], elev_lr, 0),
            "color": map_colors_gradient["buildings"],
            "mask": masks_lr["buildings"],
            "mask3d": masks_3d["roofs"] + masks_3d["walls"]},
        "water": {
            "data_proj": masks_lr["water"],
            "color": map_colors_gradient["water"],
            "mask": masks_lr["water"],
            "mask3d": masks_3d["water"]},
    }

    print('Environment has ' + str(len(envir['coo'])) + ' elements')
    if 'voxels' in CONFIG['PLOT_TOPOLOGY']:
        print('Starting voxels-plot of the environment...')
        if (len(envir['coo']) > 75000):
            print('Skipped sensors voxels plot because it has too many elements.')
        else:
            print('Starting voxels-plot of of the environment from the side...')
            plot3d.voxels_plot(
                height3d=height_mask3d,
                plot_data=terrain_lr,
                cl=envir['coo'],
                t=50,
                rot_north=ROT,
                sun_zen_rad=math.radians(20),
                sun_azi_rad=math.radians(145),
                el_size=CONFIG['RES'],
                view_dir=[30, ROT + (45 if ADDRESS == 'custom' else 0)],  # zen, azi
                figtitle='Environment voxels plot',
                grid_visibility=0.01,
                disp_sun=False,
                shade_faces=True,
                show=False,
                saveimg=True,
                dpi=1000,
                save_path=FOLDER_RES_TREES / 'environment voxels plot',
                save_ext='.png',
                face_visibility=.99,
            )
        
    if 'voxels_side' in CONFIG['PLOT_TOPOLOGY']:
        
        if (len(envir['coo']) > 75000):
            print('Skipped sensors voxels plot because it has too many elements.')
        else:
            print('Starting voxels-plot of of the environment from the side...')
            plot3d.voxels_plot(
                height3d=height_mask3d,
                plot_data=terrain_lr,
                cl=envir['coo'],
                t=50,
                rot_north=ROT,
                sun_zen_rad=math.radians(20),
                sun_azi_rad=math.radians(145),
                el_size=CONFIG['RES'],
                view_dir=[90, 0.001],  # zen, azi
                figtitle='Environment voxels plot',
                grid_visibility=0.01,
                disp_sun=False,
                shade_faces=True,
                show=False,
                saveimg=True,
                dpi=1000,
                save_path=FOLDER_RES_TREES / 'environment voxels plot from side',
                save_ext='.png',
                face_visibility=.99,
            )
        
    # topology_plot = True
    

    # %% Height / width calculation and plotting

    if CONFIG['CALC_HW_MAP'] and masks_lr["buildings"].any():
        
        start = time.time()
        try:
            if not FORCE_RECALC_ENVIR:
                hw_map = np.load(f'{FOLDER}/HW.npz')['hw_map']
                print('Loaded H/W map.')
            else:
                raise()
        except:
            hw_map = height_width.calc_hw_map(envir_no_trees['coo'],
                                    SHAPE_2D_LR,
                                    envir_no_trees['faces_lst'],
                                    save_dir=FOLDER_RES_TREES_RAY,
                                    dz=1)
        if time.time() - start > 2:
            print('Calculating H/W map took: ' + hf.duration(start))

        try:
            hw_map_smooth = cv2.medianBlur(np.float32(hw_map), ksize=5)
            hw_map_smooth_smooth = cv2.medianBlur(hw_map_smooth, ksize=5)
            hw_map_smooth_smooth[masks_lr["buildings"]] = 0
            
            if ADDRESS == 'custom':
                hw_map_plot = hw_map
            else:
                hw_map_plot = hw_map_smooth_smooth
    
            hw_ave = np.average(hw_map_smooth_smooth,
                                weights=hw_map_smooth_smooth > 0)
        except:
            hw_ave=None
            pass

        start = time.time()
        
        img_hw, colorbar = plot2d.data_over_img(
            img=map_img_gradient_lr,
            data=hw_map_plot,
            # data_mask=~(masks_lr['buildings']
            #             | masks_lr['trees']),
            data_mask=~(masks_lr['buildings']),
            colorbar_lims=(0, 1.5),
            # data_alpha=0,
            cmap='viridis')
        
        plot_data = {
            "buildings": {
                "data_proj": np.where(
                    masks_lr['buildings'],
                    elev_lr,
                    0),
                "color": "grey",
                "mask": masks_lr["buildings"]},
            "hw_map": {
                "data_proj": hw_map_plot,
                "color": "Reds",
                "mask": np.invert(
                    masks_lr["buildings"])}}
        # from side
        if 'bar' in CONFIG['PLOT_HW_MAP']:
            bar.plot(
                height2d=elev_round_lr - np.where(masks_lr['trees'], elev_lr, 0),
                img=img_hw,
                gridsize=GRIDSIZE,
                res=CONFIG['RES'],
                save_path=str(FOLDER_RES_TREES_RAY / 'hw_map'),
                saveimg=True,
                plotimg=True,
                colorbar=colorbar,
                view_zen=45,
                view_azi=-ROT)

        # from top
        if 'bar_top' in CONFIG['PLOT_HW_MAP']:
            bar.plot(
                height2d=elev_round_lr - np.where(masks_lr['trees'], elev_lr, 0),
                img=img_hw,
                gridsize=GRIDSIZE,
                res=CONFIG['RES'],
                save_path=str(FOLDER_RES_TREES_RAY / 'hw_map_top'),
                saveimg=True,
                plotimg=True,
                colorbar=colorbar,
                view_zen=0,
                view_azi=0)

    # else:
    hw_ave = None

    # %% Make 'sensors'
    
    sensor_masks3d = {}
    sensors['idcs_dict'] = {}
    
    if CONFIG['SENSOR_LOCATION'].startswith('top canyon') and hw != 0:
        sensor_mask_lr = hw_map == hw_map.max()
        sensor_xy = np.array(np.where(sensor_mask_lr)).T
    
        if CONFIG['SENSOR_LOCATION'].endswith('slice'):
            sensor_xy = sensor_xy[sensor_xy[:, 1]==SHAPE_2D[1]/2]
        
        # midpoint is 1 higher than midpoint of highest elevation point
        sensor_height = envir['coo'][:, 2].max() + 1
        sensors['coo'] = np.column_stack(
            (sensor_xy + 0.5, sensor_height * np.ones(len(sensor_xy))))
        
        # swap cols
        sensors['coo'][:, [0, 1]] = sensors['coo'][:, [1, 0]]
    
    
    elif CONFIG['SENSOR_LOCATION'] == 'ground':
        sensor_mask_lr, sensor_xy_withoud_mod = hf.make_sensor_mask(
            masks=masks_lr, 
            masks_keys=['roads', 'grass', 'concrete'], 
            shape=SHAPE_2D_LR, 
            initial_mask=sensor_mask_bools_lr)
        
        sensor_xy = np.array(np.where(sensor_mask_lr)).T
        sensor_height = 2.5
        sensors['coo'] = np.column_stack(
            (sensor_xy + 0.5, sensor_height * np.ones(len(sensor_xy))))
        
        # swap cols
        sensors['coo'][:, [0, 1]] = sensors['coo'][:, [1, 0]]
        
    elif CONFIG['SENSOR_LOCATION'] == 'street and floating':
        
        # street sensors
        try:
            sensor_mask_img = cv2.cvtColor(cv2.imread(str(FOLDER_RES_TREES / 'street_sensors_locations.png')), cv2.COLOR_BGR2RGB)
            print(f"loaded {FOLDER_RES_TREES.stem}/street_sensors_locations.png")
            plt.imshow(sensor_mask_img)
            plt.title('Loaded street_sensors_locations.png')
            plt.show()  
            
            sensor_mask_lr = hf.img_to_masks_sat(sensor_mask_img, palette={'sensors':[1,0,0], 'rest':[1,1,1]})['sensors']
        except:
            
            sensor_mask_lr, _ = hf.make_sensor_mask(
                masks=masks_lr, 
                masks_keys=['roads', 'grass', 'concrete'], 
                shape=SHAPE_2D_LR, 
                initial_mask=sensor_mask_bools_lr)
            
            if hw == 0:
                sensor_mask_lr *= hf.mask_modulus(sensor_mask_bools_lr.shape, mod=3)
            
            sensor_mask_img = map_img_lr.copy()
            sensor_mask_img[sensor_mask_lr] = [1, 0, 0]
            plt.imshow(sensor_mask_img)
            plt.title('Newly created street_sensors_locations.png')
            plt.show()
            
            imageio.imwrite(FOLDER_RES_TREES / 'street_sensors_locations.png', sensor_mask_img)
            input('Failed to load "street_sensors_locations.png", created a new one. Adjust or press enter to continue...')
            sensor_mask_img = cv2.cvtColor(cv2.imread(str(FOLDER_RES_TREES / 'street_sensors_locations.png')), cv2.COLOR_BGR2RGB)
            sensor_mask_lr_loaded = hf.img_to_masks_sat(sensor_mask_img, palette={'sensors':[1,0,0], 'rest':[1,1,1]})['sensors']
            
            if not np.all(sensor_mask_lr_loaded == sensor_mask_lr):
                plt.imshow(sensor_mask_img)
                plt.title('Adjusted street_sensors_locations.png')
                plt.show()
            
            sensor_mask_lr = sensor_mask_lr_loaded
            
        sensor_mask_lr_street = sensor_mask_lr.copy()
        
        sensor_xy = np.array(np.where(sensor_mask_lr)).T
        sensor_height = 2.5
        coo_street = np.column_stack(
            (sensor_xy + 0.5, sensor_height * np.ones(len(sensor_xy))))
        
        sensor_masks3d['street'] = hf.array2d_to_array3d(
            np.ones((len(coo_street))), 
            coo_street[:, [1, 0, 2]],
            shape=SHAPE_3D_LR).astype(bool)
        
        sensors['idcs_dict']['street'] = np.arange(len(coo_street))
        
        # floating sensors above urban environment
        try:
            sensor_mask_img = cv2.cvtColor(cv2.imread(str(FOLDER_RES / 'floating_sensors_locations.png')), cv2.COLOR_BGR2RGB)
            print(f"Loaded {FOLDER_RES.stem}/floating_sensors_locations.png")
            
            if show_all_plots:
                plt.imshow(sensor_mask_img)
                plt.title('Loaded floating_sensors_locations.png')
                plt.show()  
            
            sensor_mask_lr = hf.img_to_masks_sat(sensor_mask_img, palette={'sensors':[0,0,1], 'rest':[1,1,1]})['sensors']
        except:
            
            sensor_mask_lr = sensor_mask_bools_lr * hf.mask_modulus(sensor_mask_bools_lr.shape, mod=3)
            
            sensor_mask_img = map_img_lr.copy()
            sensor_mask_img[sensor_mask_lr] = [0, 0, 1]
            plt.imshow(sensor_mask_img)
            plt.title('Newly created floating_sensors_locations.png')
            plt.show()
            
            imageio.imwrite(FOLDER_RES / 'floating_sensors_locations.png', sensor_mask_img)
            input('Failed to load "floating_sensors_locations.png", created a new one. Adjust or press enter to continue...')
            sensor_mask_img = cv2.cvtColor(cv2.imread(str(FOLDER_RES / 'floating_sensors_locations.png')), cv2.COLOR_BGR2RGB)
            sensor_mask_lr_loaded = hf.img_to_masks_sat(sensor_mask_img, palette={'sensors':[0,0,1], 'rest':[1,1,1]})['sensors']
            
            if not np.all(sensor_mask_lr_loaded == sensor_mask_lr):
                plt.imshow(sensor_mask_img)
                plt.title('Adjusted floating_sensors_locations.png')
                plt.show()
            
            sensor_mask_lr = sensor_mask_lr_loaded
        
        sensor_xy = np.array(np.where(sensor_mask_lr)).T
        sensor_height = envir['coo'][..., 2].max() + 2
        if hw == 0:
            sensor_height += 1
        coo_floating = np.column_stack(
            (sensor_xy + 0.5, sensor_height * np.ones(len(sensor_xy))))
        
        sensor_masks3d['floating'] = hf.array2d_to_array3d(
            np.ones((len(coo_floating))), 
            coo_floating[:, [1, 0, 2]],
            shape=SHAPE_3D_LR).astype(bool)
        
        sensors['idcs_dict']['floating'] = np.arange(len(coo_street), len(coo_street)+len(coo_floating))
        
        
        coo_street[:, [0, 1]] = coo_street[:, [1, 0]]
        coo_floating[:, [0, 1]] = coo_floating[:, [1, 0]]
        
        # coo list of both floating and street sensors
        sensors['coo'] = np.concatenate([coo_street, coo_floating])
        
        
        
        
        
    elif CONFIG['SENSOR_LOCATION'] == 'img' or hw == 0:
        while True:
            try:
                sensor_mask_img = cv2.cvtColor(cv2.imread(str(FOLDER / 'sensors_locations.png')), cv2.COLOR_BGR2RGB)
                break
            except:
                input('Failed to load "sensors_locations.png", please fix and press enter to retry...')
        
        sensor_mask = hf.img_to_masks_sat(sensor_mask_img, palette={'sensors':[1,0,0], 'rest':[1,1,1]})
        sensor_mask_lr = hf.reduce_res_mask(sensor_mask, reduce_res, shape=SHAPE_2D_LR)['sensors']
        
        # sensor_mask_lr, sensor_xy_withoud_mod = hf.make_sensor_mask(
        #     masks=masks_lr, 
        #     masks_keys=['roads', 'grass', 'concrete'], 
        #     shape=SHAPE_2D_LR, 
        #     initial_mask=sensor_mask_bools_lr)
        
        sensor_xy = np.array(np.where(sensor_mask_lr)).T
    
        if masks_lr['buildings'].any() and (elev_lr > 0).any():
            # sensor_height = int(elev_lr.max() / CONFIG['RES']) + 1
            sensor_height = envir['coo_int'][..., 2].max() + 2 
        else:
            sensor_height = int(5 / CONFIG['RES']) + 1
    
        sensors['coo'] = np.column_stack(
            (sensor_xy + 0.5, sensor_height * np.ones(len(sensor_xy))))
        
        # swap cols
        sensors['coo'][:, [0, 1]] = sensors['coo'][:, [1, 0]]

    


    # filter sensors coordinates using sensor_mask_lr
    # idcs_sensors_full = hf.coo_to_ndarray(
    #     np.column_stack([sensors['coo'][:, :2], np.arange(len(sensors['coo']))]),
    #     shape=SHAPE_2D_LR)

    # idcs_sensors = idcs_sensors_full[sensor_mask_lr].astype(int)

    # sensors['coo'] = sensors['coo'][idcs_sensors]

    sensors['idcs_3D'] = hf.coo_to_idcs3D(
        coo_int=sensors['coo'].astype(int),
        shape=(SHAPE_3D_LR[0], SHAPE_3D_LR[1], SHAPE_3D_LR[2]))

    sensors['idcs_3D_masked'] = np.ma.masked_equal(sensors['idcs_3D'], -1)

    idcs_sensor_map_all = hf.coo_to_ndarray(
        np.column_stack([sensors['coo'][:, :2], np.arange(len(sensors['coo']))]),
        shape=SHAPE_2D_LR)
    idcs_sensor_map_select = idcs_sensor_map_all[sensor_mask_lr]
    idcs_sensors_domain = idcs_sensor_map_select[~np.isnan(
        idcs_sensor_map_select)].astype(int)

    # indices of elements beneath the sensors
    idcs_ground = envir['idcs_3D'][sensor_mask_lr, 0]

    coo_idcs_under_sensor = envir['idcs_3D'][..., 0].take(np.ravel_multi_index(
        sensors['coo'][:, :2].astype(int).T, SHAPE_2D_LR))
    
    idcs_under_floating_sensor = envir['idcs_3D'][coo_floating[:, 0].astype(int), coo_floating[:, 1].astype(int)]
    idcs_under_floating_sensor = idcs_under_floating_sensor[idcs_under_floating_sensor>-1]
    
    idcs_under_cropped_area = envir['idcs_3D'][coo_street[:, 0].astype(int), coo_street[:, 1].astype(int)]
    idcs_under_cropped_area = idcs_under_cropped_area[idcs_under_cropped_area>-1]
    
    
    # sensor_mask3d = hf.array2d_to_array3d(
    #     np.ones((len(sensors['coo']))), sensors['coo'], shape=SHAPE_3D_LR).astype(bool)
    
    # %%% voxels plot of sensors
    
    if 'voxels' in CONFIG['PLOT_SENSOR_LOCATION']:
        try:
            if (len(envir['coo'])+len(sensors['coo']) > 100000):
                print('Skipped sensors voxels plot because it has too many elements.')
            
            else:
            # print('Start voxels plot...')
                terrain_lr_sensor = terrain_lr.copy()
                terrain_lr_sensor.update({
                    "floating sensors": {
                        "data_proj": sensor_mask_lr,
                        "color": [0, 0, 1, 0.15],
                        "mask": sensor_mask_lr,
                        "mask3d": sensor_masks3d['floating']},
                    # "street sensors": {
                    #     "data_proj": sensor_mask_lr,
                    #     "color": [1, 0, 0, 0.3],
                    #     "mask": sensor_mask_lr,
                    #     "mask3d": sensor_masks3d['street']},
                    })
                
                print('Starting voxels-plot of the environment and sensors (in red)...')
                
                plot3d.voxels_plot(
                    height3d=(
                        sensor_masks3d['floating']
                        # +sensor_masks3d['street']
                        +height_mask3d),
                    plot_data=terrain_lr_sensor,
                    cl=np.concatenate([envir['coo'], sensors['coo']]),
                    t=50,
                    rot_north=ROT,
                    sun_zen_rad=math.radians(20),
                    sun_azi_rad=math.radians(145),
                    el_size=CONFIG['RES'],
                    view_dir=[30, ROT],  # zen, azi
                    figtitle=f'{CONFIG["ADDRESS_STR"]}, \n Floating sensor elements in blue \n Pedestrians sensors in red',
                    grid_visibility=0.01,
                    disp_sun=False,
                    shade_faces=True,
                    show=False,
                    saveimg=True,
                    dpi=1000,
                    save_path=FOLDER_RES_TREES / 'Voxels plot with sensors',
                    save_ext='.png',
                    face_visibility=.99,
                    highlight_els={'green': idcs_under_floating_sensor} if ('model 1' in str(FOLDER).lower() or 'model 2' in str(FOLDER).lower()) else {},
                    overwrite_existing=False,
                )
                
                if 'custom' in CONFIG['ADDRESS'].lower() and hw == 1:
                    plot3d.voxels_plot(
                        height3d=(
                            sensor_masks3d['floating']
                            # +sensor_masks3d['street']
                            +height_mask3d),
                        plot_data=terrain_lr_sensor,
                        cl=np.concatenate([envir['coo'], sensors['coo']]),
                        t=50,
                        rot_north=ROT,
                        sun_zen_rad=math.radians(20),
                        sun_azi_rad=math.radians(145),
                        el_size=CONFIG['RES'],
                        view_dir=[45, 30],  # zen, azi
                        figtitle=f'{CONFIG["ADDRESS_STR"]}, \n Floating sensor elements in blue \n Pedestrians sensors in red',
                        grid_visibility=0.01,
                        disp_sun=False,
                        shade_faces=True,
                        show=False,
                        saveimg=True,
                        dpi=1000,
                        save_path=FOLDER_RES_TREES / 'Voxels plot with sensors side',
                        save_ext='.png',
                        face_visibility=1,
                        labels=False,
                        ticks=False,
                        highlight_els={'green': idcs_under_floating_sensor} if ('model 1' in str(FOLDER).lower() or 'model 2' in str(FOLDER).lower()) else {},
                        overwrite_existing=False,
                    )
                else:
                    print('real world, so skipped the voxelsplot from the side.')
        except Exception as e:
            print(e)
            pass
            

    # %% Physics and simulation settings

    # %%% Phyiscal principles toggles

    interaction = CONFIG['ACTIVE_PHYSICS']['interaction']

    sw_reflection = CONFIG['ACTIVE_PHYSICS']['sw_out_refl']

    sw_in_diff = CONFIG['ACTIVE_PHYSICS']['sw_in_diff']
    sw_in_dir = CONFIG['ACTIVE_PHYSICS']['sw_in_dir']

    lw_in_sky = CONFIG['ACTIVE_PHYSICS']['lw_in_sky']
    lw_out_emm = CONFIG['ACTIVE_PHYSICS']['lw_out_emm']
    lw_out_refl = CONFIG['ACTIVE_PHYSICS']['lw_out_refl']
    
    convection = CONFIG['ACTIVE_PHYSICS']['convection']
    conduction_surface = CONFIG['ACTIVE_PHYSICS']['conduction_surface']
    anthropogenic = CONFIG['ACTIVE_PHYSICS']['anthropogenic']

    T_trees_eqls_air = True # TODO move elsewhere?

    labels = []
    if sw_in_dir:
        labels.append('sw_in_dir')
    if sw_in_diff:
        labels.append('sw_diff')
    if sw_reflection:
        labels.append('refl')
    if lw_out_emm:
        labels.append('lw_out_emm')
    if lw_out_refl:
        labels.append('lw_out_refl')
    if conduction_surface:
        labels.append('cond')
    if convection:
        labels.append('conv')
    if anthropogenic:
        labels.append('antr')
    if T_trees_eqls_air:
        labels.append('tree_eq_air')

    # %%% Solar settings

    path_solar_angles = (FOLDER / CONFIG['SOLAR_ANGLES']['filename'])

    try:
        if CONFIG['RECALC_SUNLIT_ELS']:
            raise Exception()
        
        # path_solar_angles = FOLDER / 'solar_angles_location.txt'
        sun_pos = pd.read_csv(
            path_solar_angles,
            index_col='Datetime',
            parse_dates=['Datetime']).asfreq(FREQ_STR).interpolate()
        recalc_sun = False

    except (FileNotFoundError, Exception):
        recalc_sun = True
        sun_pos = pd.DataFrame()
        for _dates in dates_lst:
            if CONFIG['SOLAR_ANGLES']['location'] == True:
                _sun_pos = solar_angles.location_sun_pos(
                    latitude, 
                    longitude, 
                    CONFIG['T_STEP'], 
                    start_date=_dates[0], 
                    end_date=_dates[-1],
                    tz=CONFIG['SOLAR_ANGLES'].get('timezone', 'Europe/Amsterdam'),
                    )
    
            elif CONFIG['SOLAR_ANGLES']['location'] == False:
                _sun_pos = solar_angles.custom_sun_pos(
                    t_step=CONFIG['T_STEP'], 
                    zenith=CONFIG['SOLAR_ANGLES']['zenith'],
                    azimuth=CONFIG['SOLAR_ANGLES']['azimuth'],
                    tz=CONFIG['SOLAR_ANGLES'].get('timezone', 'Europe/Amsterdam'),
                    start_date=_dates[0],
                    end_date=_dates[-1])
                
            sun_pos = sun_pos.append(_sun_pos)
            
        sun_pos["azi_rad"] = np.deg2rad(sun_pos["azi_deg"]) % (2 * math.pi)
        sun_pos["zen_rad"] = np.deg2rad(sun_pos["zen_deg"])
            
        sun_pos.to_csv(path_solar_angles)
    
    if isinstance(ROT, (int, float)):
        sun_pos["azi_deg TRUE"] = sun_pos["azi_deg"].copy() % 360
        sun_pos["azi_deg"] = (sun_pos["azi_deg"] + ROT) % 360
    
    sun_pos["azi_rad"] = np.deg2rad(sun_pos["azi_deg"]) % (2 * math.pi)
    
    sun_pos.index = sun_pos.index + \
        (datetime.datetime(dates_lst[0][0].year, 1, 1, 0, 0) - datetime.datetime(sun_pos.index[0].year, 1, 1, 0, 0))

    sunrise = solar_angles.sunrise(sun_pos)
    sunset = solar_angles.sunset(sun_pos)
    sunhighpoint = solar_angles.sunhighpoint(sun_pos)
    
    fig, ax = plt.subplots() 

    sun_pos.plot(y='zen_deg', label='Zenith', ax=ax) 
    sun_pos.plot(y='azi_deg', label='Azimuth (corrected for rotation)', ax=ax, secondary_y = True)
    plt.title('Zenith and azimuth (corrected for rotation) angle')
    plt.show()

    # %%% Weather settings

    # weather_source = 'wow-344'
    weather_source = 'bsrn'
    if weather_source == 'bsrn':
        
        if CONFIG['START_DATES'][0][0] == 2019:
            bsrn_file = 'data/CAB_radiation_2019-07.tab'
            bsrn_df = pd.read_csv(bsrn_file, sep='\t', skiprows=41, index_col='Date/Time', parse_dates=['Date/Time'])
            
        if CONFIG['START_DATES'][0][0] == 2023:
            bsrn_file = 'data/CAB_radiation_2023-09.tab'
            bsrn_df = pd.read_csv(bsrn_file, sep='\t', skiprows=42, index_col='Date/Time', parse_dates=['Date/Time'])
            
        bsrn_df.index = bsrn_df.index.tz_localize('UTC').tz_convert('Europe/Amsterdam').tz_localize(None)
    if weather_source == 'wow-344':

        bsrn_file='data/knmi-wow-RdmAirport-2023-09.csv'
        bsrn_df = pd.read_csv(bsrn_file, sep=';',index_col='datum', parse_dates=['datum'])
    
    bsrn_df = bsrn_df.resample(FREQ_STR).mean().interpolate() 
    bsrn_df = bsrn_df.loc[sun_pos.index[0]: sun_pos.index[-1]]
    
    # load tud weather data (for wind speed and ambient temperature)
    weather_file = 'data/weather/tud/Delfshaven.csv'
    weather_tud_df = pd.read_csv(weather_file, sep=',', skiprows=0, index_col='DateTime', parse_dates=['DateTime'],low_memory=False)
    weather_tud_df.index = weather_tud_df.index.tz_localize('UTC').tz_convert('Europe/Amsterdam').tz_localize(None)
    weather_tud_df = weather_tud_df.loc[sun_pos.index[0]: sun_pos.index[-1]]
    object_columns = weather_tud_df.select_dtypes(include=['object']).columns
    for column in object_columns:
        try:
            weather_tud_df[column] = weather_tud_df[column].astype(float)
        except ValueError:
            # Handle exceptions for non-convertible values
            print(f"Column '{column}' contains non-convertible values.")
    
    weather_tud_df = weather_tud_df.resample(FREQ_STR).mean().interpolate()
    weather_tud_df = weather_tud_df.loc[sun_pos.index[0]: sun_pos.index[-1]]
    
    
    
    closest_stn = hf.find_closest_stn(latitude, longitude)

    dates_str = dates.astype(str)

    try:
        if SIMULATE_AVERAGES:
            knmi_file = 'data/knmi_%i-%i.csv' % tuple(CONFIG['AVERAGE_YEARS'])

        if not SIMULATE_AVERAGES:
            knmi_file = 'data/knmi_data_' + str(closest_stn) + '.txt'
        
        knmi_data = pd.read_csv(knmi_file,
                                index_col='Date',
                                parse_dates=['Date'])
        
        # check if all dates are in the loaded file, if not throw error
        if not all([date in knmi_data.index for date in dates_str]):
            raise FileNotFoundError('Found file does not contain all dates.')

        knmi_data = knmi_data.sort_index()
        knmi_data = knmi_data.asfreq(FREQ_STR).interpolate()
        
        knmi_data.to_csv(knmi_file)
        
        # only sliced dates to be simulated
        knmi_data = knmi_data[np.isin(
            knmi_data.index.strftime('%Y-%m-%d'), dates_str)]

    except FileNotFoundError:
        knmi_data = pd.DataFrame() 
        for _date in dates:
            _knmi_data = get_data.knmi_weather(
                closest_stn=closest_stn, 
                t_step=CONFIG['T_STEP'], 
                start_date=_date,  
                end_date=_date,
                simulate_averages=SIMULATE_AVERAGES, 
                average_years=CONFIG['AVERAGE_YEARS'],
                timezone=CONFIG['SOLAR_ANGLES'].get('timezone', 'Europe/Amsterdam'),
                save_path=knmi_file)
            
            knmi_data = knmi_data.append(_knmi_data)

        knmi_data.to_csv(knmi_file)

    # t_max = sun_pos.index.max()
    # time_array = sun_pos.index.time.astype(str)

    # %%% Anthropogenic heat production
    if anthropogenic:
        if ADDRESS == 'custom' or True:
            if not SIMULATE_AVERAGES:
                anthropogenic_power_m2 = pd.read_csv(
                    'data/anthropogenic_power_m2.txt',
                    squeeze=True,
                    index_col='Date',
                    parse_dates=['Date']).asfreq(FREQ_STR).interpolate()
            else:
                anthropogenic_power_m2 = pd.read_csv(
                    'data/anthropogenic_power_ave_m2.txt',
                    squeeze=True,
                    index_col='Date',
                    parse_dates=['Date'])
        else:
            anthropogenic_power_m2 = get_data.anthropogenic_buildings(
                bag_gdf=bag_gdf, 
                air_temp=knmi_data['T'], 
                nr_buildings_els = masks_3d['buildings'].sum(),
                simulate_averages=SIMULATE_AVERAGES, 
                area=(CONFIG['RES'] * CONFIG['RES']))
        
    elif not anthropogenic:
        pass
    # %%% Physics constants

    U_i = 2  # brick, surface heat transmission coefficient
    # thickness = 0.1  # 10cm thickness wall layer
    SB_CONST = 5.67 * 10**(-8)

    CONFIG['THERMAL_DIFF'] = 4. * 10**-6  # Thermal diffusivity of steel, m2.s-1

    # dir_norm_irr_max = 1370 #direct normal irradiance = solar constant
    # diff_hor_irr_max = 200 #diffuse horizontal irradiance

    sol_const = 1367
    # date = datetime.datetime(year=2020, month=12, day=29)

    # %%% Load weather data and load/set forcings
    
    weather = pd.DataFrame(index=sun_pos.index)
    
    if CONFIG['T_AIR'] == 'knmi':
        weather['T_air'] = (knmi_data["T"] + 273.15).rolling('2h', center=True)
    elif CONFIG['T_AIR'] == 'bsrn':
        weather['T_air'] = (bsrn_df['T2 [°C]'] + 273.15).rolling('2h', center=True)
    elif CONFIG['T_AIR'] == 'tud-oost':
        weather['T_air'] = (weather_tud_df['Tair_{Avg}'] + 273.15).rolling('2h', center=True).mean()
    else:
        list_to_dataseries(df=weather, var='T_air', data=CONFIG['T_AIR'])
        
    if CONFIG['WIND_SPEED'] == 'tud-oost':
        weather['wind_speed'] = (weather_tud_df['WindSpd_{Avg}']).rolling('2h', center=True).mean()
        
    if CONFIG['WIND_SPEED'] == 'knmi':
        weather['wind_speed'] = knmi_data["FH"].rolling('2h', center=True).mean()
        
        
        
    else:
        list_to_dataseries(df=weather, var='wind_speed', data=CONFIG['WIND_SPEED'])
        
    if CONFIG['REL_HUM'] == 'knmi':
        weather['rel_hum'] = knmi_data["U"]
    elif CONFIG['REL_HUM'] == 'bsrn':
        weather['rel_hum'] = bsrn_df['RH [%]']
    else:
        list_to_dataseries(df=weather, var='rel_hum', data=CONFIG['REL_HUM'])
        
    if CONFIG['CLOUD_COVER'] == 'knmi':
        weather['cloud_cover'] = knmi_data["N"]
    else:
        list_to_dataseries(df=weather, var='cloud_cover', data=CONFIG['CLOUD_COVER'])
    
    # weather['T_dewpoint_celcius'] = physics.physics.dewpoint_approximation(weather['T_air'], weather['rel_hum'])
    
    # if CONFIG['EMISSIVITY_SKY'] == 'calculate':
    #     # weather['emissivity_sky'] = physics.physics.emissivity_sky(weather['T_dewpoint_celcius'], time=weather['T_dewpoint_celcius'].index)
    #     weather['emissivity_sky'] = physics.physics.emissivity_sky(weather['T_dewpoint_celcius'])
    # else:
    #     list_to_dataseries(df=weather, var='emissivity_sky', data=CONFIG['EMISSIVITY_SKY'])
    
    
    # weather['T_sky_eff'] = np.power((weather['emissivity_sky'] * weather['T_air']**4), 1 / 4)



    sun_pos["azi_rad"] = np.where(
        sun_pos["azi_rad"] > math.pi,
        sun_pos["azi_rad"] - 2 * math.pi,
        sun_pos["azi_rad"])


    # forcings
    forcings = pd.DataFrame(index=sun_pos.index)

    if 'G_no_atm_hor' in CONFIG['RADIATIONS']:
        forcings['G_no_atm_hor'] = CONFIG['RADIATIONS']['G_no_atm_hor']
    else:
        forcings['G_no_atm_hor'] = physics.physics.glob_rdtn_no_atmosph(
            sun_pos['zen_rad'], sun_pos['azi_rad'], sun_pos.index, latitude, sol_const)
    
    if 'G_cloudless_hor' in CONFIG['RADIATIONS']:
        forcings['G_cloudless_hor'] = CONFIG['RADIATIONS']['G_cloudless_hor']
    else:
        forcings['G_cloudless_hor'] = physics.physics.glob_rdtn_cloudless(
            forcings['G_no_atm_hor'], sun_pos['zen_rad'])
        
    if 'G_hor' in CONFIG['RADIATIONS']:
        forcings['G_hor'] = CONFIG['RADIATIONS']['G_hor']
    else:
        if CONFIG['G_HOR'] == 'calculate':
            # forcings['G_hor'] = np.clip((1000 * np.cos(sun_pos['zen_rad'])), 0, None) 
            forcings['G_hor'] = physics.physics.glob_rdtn(
                forcings['G_cloudless_hor'], weather['cloud_cover'])
        elif CONFIG['G_HOR'] == 'bsrn':
            forcings['G_hor'] = bsrn_df['SWD [W/m**2]']
            

    if 'Diff_hor' in CONFIG['RADIATIONS']:
        forcings['Diff_hor'] = CONFIG['RADIATIONS']['Diff_hor']
    else:
        if CONFIG['DIFF_HOR'] == 'calculate':
            forcings['Diff_hor'] = np.clip(physics.physics.diff_radiation(
                forcings['G_hor'], forcings['G_no_atm_hor'], sun_pos['zen_rad'], method='deJong'), a_min=0, a_max=None)
            print('\n\n\nUSING DIFF = 0.177 * G_HOR\n\n\n')
            forcings['Diff_hor'] = CONFIG.get('DIFFUSION_FACTOR', 0.177) * forcings['G_hor']
            
        elif CONFIG['DIFF_HOR'] == 'bsrn':
            forcings['Diff_hor'] = bsrn_df['DIF [W/m**2]']
        
    if 'Dir_hor' in CONFIG['RADIATIONS']:
        forcings['Dir_hor'] = CONFIG['RADIATIONS']['Dir_hor']
    else: 
        if CONFIG['DIR_HOR'] == 'calculate':
            forcings['Dir_hor'] = np.clip(forcings['G_hor'] - forcings['Diff_hor'], a_min=0, a_max=None)
        elif CONFIG['DIR_HOR'] == 'bsrn':
            forcings['Dir_hor'] = bsrn_df['DIR [W/m**2]'] * np.cos(sun_pos['zen_rad'])
    
    if 'Dir_perp' in CONFIG['RADIATIONS']:
        forcings['Dir_perp'] = CONFIG['RADIATIONS']['Dir_perp']
    else:
        if CONFIG['DIR_PERP'] == 'calculate':
            forcings['Dir_perp'] = forcings['Dir_hor'] / np.cos(sun_pos['zen_rad'])
        elif CONFIG['DIR_PERP'] == 'bsrn':
            forcings['Dir_perp'] = bsrn_df['DIR [W/m**2]']
            
    if 'LW_sky' in CONFIG['RADIATIONS']:
        forcings['LW_SKY'] = CONFIG['RADIATIONS']['LW_sky']
    else:
        if CONFIG['LW_SKY'] == 'calculate':
            forcings['LW_sky'] = physics.physics.radiative_pwr(emm=0.8, T=weather['T_air'])
        elif CONFIG['LW_SKY'] == 'bsrn':
            forcings['LW_sky'] = bsrn_df['LWD [W/m**2]']
            
    forcings['T_air'] = weather['T_air']
    forcings['wind_speed'] = weather['wind_speed']

    FOLDER_RES_LOG = FOLDER_RES_TREES_RAY / 'log_data/'
    if not os.path.isdir(FOLDER_RES_LOG):
        os.mkdir(FOLDER_RES_LOG)

    # if address == 'custom':
    FOLDER_RES_LOG_ROT = FOLDER_RES_LOG / ('rotated' + str(ROT))
    if not os.path.isdir(FOLDER_RES_LOG_ROT):
        os.mkdir(FOLDER_RES_LOG_ROT)

    FOLDER_RES_LOG_ROT_AVE = FOLDER_RES_LOG_ROT / ('average%i-%i' % tuple(CONFIG['AVERAGE_YEARS']) if SIMULATE_AVERAGES else 'real')
    if not os.path.isdir(FOLDER_RES_LOG_ROT_AVE):
        os.mkdir(FOLDER_RES_LOG_ROT_AVE)

    FOLDER_RES_LOG_ROT_CC_WS = FOLDER_RES_LOG_ROT_AVE / '&'.join(labels)

    if not os.path.isdir(FOLDER_RES_LOG_ROT_CC_WS):
        os.mkdir(FOLDER_RES_LOG_ROT_CC_WS)

    # %%% Load environmental parameters

    skin_thickness_dict = hf.load_dict('constants/skin_thickness.json')  # [m]
    albedo_dict = hf.load_dict('constants/albedo.json')  # [-]
    emissivity_dict = hf.load_dict('constants/emissivity.json')  # [-]
    # density_dict = hf.load_dict('constants/density.sjon')  # [kg/m3]
    # spec_heat_cap_dict = hf.load_dict('constants/spec_heat_cap.txt')  # [J/(kg K)]
    vol_heat_cap_dict = hf.load_dict('constants/vol_heat_cap.json')  # [J/(m3 K)]
    thermal_cond_dict = hf.load_dict('constants/thermal_cond.txt')  # [W/(m K)]

    for key, dct in {'emissivity': emissivity_dict, 
                     'albedo_sw': albedo_dict, 
                     # 'density': density_dict,
                     'vol_heat_cap': vol_heat_cap_dict, 
                     'skin_thickness': skin_thickness_dict,
                     'thermal_cond': thermal_cond_dict}.items():
    
        if key in kwargs:
            val = float(kwargs[key])
            if isinstance(val, (float, int)):
                for surf_type in albedo_dict:
                    dct[surf_type] = val
                if __debug__:
                    print(f'Updated all {key} values to {val}')
            # elif isinstance(kwargs[key], (dict)):
            #     for surf_type in kwargs[key]:
            #         dct[surf_type] = kwargs[key][surf_type]
            #         if __debug__:
            #             print(f'Updated {key} values of {surf_type} to {kwargs[key][surf_type]}')
            # else:
            #     raise ValueError(f'Incorrect custom input for {key}: {kwargs[key]} has type {type(kwargs[key])}, must be int, float or dict.')

    albedo_sw_std = 0

    emissivity_std = 1
    emissivity_sky = CONFIG['EMISSIVITY_SKY']

    materials = dict()

    materials['emissivity'] = emissivity_std * np.ones(len(envir['coo']))
    materials['albedo_sw'] = albedo_sw_std * np.ones(len(envir['coo']))
    # materials['density'] = 1 * np.ones(len(envir['coo']))
    materials['vol_heat_cap'] = 1 * np.ones(len(envir['coo']))
    materials['skin_thickness'] = 1 * np.ones(len(envir['coo']))
    materials['thermal_diff'] = 1 * np.ones(len(envir['coo']))

    for surf_type in envir['idcs_dict']:
        if surf_type == 'buildings':
            continue
        materials['emissivity'][envir['idcs_dict'][surf_type]
                                ] = emissivity_dict[surf_type]
        materials['albedo_sw'][envir['idcs_dict'][surf_type]] = albedo_dict[surf_type]
        # materials['density'][envir['idcs_dict'][surf_type]] = density_dict[surf_type]
        materials['vol_heat_cap'][envir['idcs_dict'][surf_type]
                                   ] = vol_heat_cap_dict[surf_type]
        materials['skin_thickness'][envir['idcs_dict'][surf_type]
                               ] = skin_thickness_dict[surf_type]
        materials['thermal_diff'][envir['idcs_dict'][surf_type]] = thermal_cond_dict[surf_type] / \
            (vol_heat_cap_dict[surf_type])

    if CONFIG['TREES']:
        materials['albedo_sw'][envir['idcs_dict']["trees"]] = albedo_dict["trees"]
        

    materials_person = dict()

    materials_person['emissivity'] = 0.97 * \
        np.ones(len(sensors['coo']))  # P. Schrijvers
    materials_person['albedo_sw'] = 0.3 * \
        np.ones(len(sensors['coo']))  # P. Schrijvers

    albedo_set = np.array([0])

    materials['albedo_lw'] = (1 - materials['emissivity'])

    hw_set = np.array([1])

    if conduction_surface:
        max_dt = physics.heat.max_dt((CONFIG['RES'], CONFIG['RES'], CONFIG['RES']), materials["thermal_diff"])
        if CONFIG['T_STEP'] > max_dt / 2 and CHECK_TIMESTEP:
            raise ValueError(
                'Timestep is too big, this will cause ' +
                'instabilities in the 3D heat diffusion. ' +
                'Please reduce timestep. Max dt is: ' +
                str(max_dt))


    # %% View-factors environmental elements


    # %%% Calculate or load VF-matrix

    if sw_reflection or sw_in_diff or lw_out_emm or lw_out_refl:
        # ---- Load view-factor matrix

        if not FORCE_RECALC_ENVIR:
            try:
                print('Loading F_matrix...')
                start = time.time()

                envir_vf['coords_triu'] = np.load(
                    FOLDER_RES / 'vf_coo_triu_RAW.npz')['coords_triu']
                envir_vf['data_triu'] = np.load(
                    FOLDER_RES / 'vf_coo_triu_RAW.npz')['data_triu']
                envir_vf['dense_svf'] = np.load(
                    FOLDER_RES / 'svf_dense.npz')['dense_svf']

                if __debug__:
                    envir_vf['sparse_triu'] = sparse.COO(
                        coords=envir_vf['coords_triu'].T.astype(np.int64),
                        data=envir_vf['data_triu'],
                        shape=(len(envir['coo']), len(envir['coo']), 6, 6))
                    
                    envir_vf['sparse'] = envir_vf['sparse_triu'] + \
                        envir_vf['sparse_triu'].transpose((1, 0, 3, 2))

                print('Succesfully loaded F_matrix: ' + hf.duration(start))
                start = time.time()

            except (FileNotFoundError, KeyError):
                FORCE_RECALC_ENVIR = True
                
        # ---- Calculate view-factor matrix
        if FORCE_RECALC_ENVIR:
            print('\n\nStarted view factor calculation for ' +
                  str(len(envir['coo'])) + ' elements... Please wait...\n\n')

            start_VF_calc = time.time()

            # Calculate COOrdinate list (via sparse matrix) of the viewing
            # factors between all faces. Note that only the upper triangle
            # faces are calculated but that due to symmetry (equal surfaces)
            # the outcome can be transposed.
            envir_vf['coords_triu'], envir_vf['data_triu'] = vf_j.calc_matrix_mp(
                envir_from=envir,
                envir_to=envir,
                min_vf_value=CONFIG['MIN_VF_VALUE'],
                skip_bottom=True,
                # max_dist=MAX_RAY_ELS, # set max raylength (in elements)
                max_dist=None,
                multiprocessing='starmap',
                # processes=8,
            )

            time_VF_calc = time.time() - start_VF_calc
            

            np.savez_compressed(FOLDER_RES / 'vf_coo_triu_RAW.npz',
                                coords_triu=envir_vf['coords_triu'],
                                data_triu=envir_vf['data_triu'])

            # reduce computational costs of the simulation by removing
            # a few low value vief factors.
            if CONFIG['REDUCE_ACCURACY_VF_PERC'] > 0:
                envir_vf['coords_triu'], envir_vf['data_triu'], off_perc = hf.coo_reduce_accuracy(
                    coords=envir_vf['coords_triu'], data=envir_vf['data_triu'], reduce_acc=CONFIG['REDUCE_ACCURACY_VF_PERC'])

            envir_vf['sparse_triu'] = sparse.COO(
                coords=envir_vf['coords_triu'].T.astype(np.int64),
                data=envir_vf['data_triu'],
                shape=(len(envir['coo']), len(envir['coo']), 6, 6))


            print('Started sparse...')
            start_sparse = time.time()        
            
            # transpose the upper triangle indices to retreive all indices and
            # have a full COOrdinate list
            envir_vf['sparse'] = envir_vf['sparse_triu'] + \
                envir_vf['sparse_triu'].transpose((1, 0, 3, 2))
                
            print(hf.duration(start_sparse))

            print('Started todense...') 
            start_todense = time.time()        

            envir_vf['dense_svf'] = envir['faces_lst'].astype(
                int) - envir_vf['sparse'].sum((1,3)).todense()
            print(hf.duration(start_todense))
            
            # set bottom-faces to zero (they should have no contribution to
            # svf)
            envir_vf['dense_svf'][:, 2] = 0

            if not __debug__:
                del envir_vf['sparse_triu'], envir_vf['sparse']

            print('Started saving', FOLDER_RES / 'svf_dense.npz')
            start_save = time.time()
            
            np.savez_compressed(
                FOLDER_RES / 'svf_dense.npz',
                dense_svf=envir_vf['dense_svf'])
            
            print('Saving took: ', hf.duration(start_save))
            
            print(f'Calculating and vf-matrix takes: {hf.duration(start)} (as will be appended in csv file)')


        # Sky view factor
        sky_vf_sum = envir_vf['dense_svf'].sum(1)

        ave_svf_ground = envir_vf['dense_svf'][list(envir['idcs_dict']["roads"]) +
                                               list(envir['idcs_dict']["grass"]) +
                                               list(envir['idcs_dict']["concrete"]), 3].mean()


    else:
        min_vf = None
        off_perc = None
        sky_view_factors = None
        sky_vf_sum = None
        sky_exposure_factor = None
        F_matrix, F_matrix_sum, F_matrix_sum123 = None, None, None

    try:
        try:
            VF_df = pd.read_csv(
                FOLDER /
                'View Factor data.csv',
                index_col=[
                    'hw',
                    'res',
                    'trees']).sort_index()
        except ValueError:
            VF_df = pd.read_csv(
                FOLDER /
                'View Factor data.csv',
                index_col=[
                    'hw',
                    'res',
                    'use_trees']).sort_index()
            VF_df = VF_df.rename(columns={'use_trees': 'trees'})
    except FileNotFoundError:
        VF_df = pd.DataFrame(index=pd.MultiIndex.from_product(
            [[1.0, 2.0, 'real world'],
             [0.5, 1.0, 2.0, 4.0],
             [True, False]],
            names=['hw', 'res', 'trees']))

        VF_df = pd.DataFrame(
            index=pd.MultiIndex.from_product([[], [], []],
                                             names=['hw', 'res', 'trees']))

    try:
        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES'])]
    except BaseException:
        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'VF calc time'] = None
        VF_df = VF_df.sort_index()

    if 'time_VF_calc' in locals():
        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'VF calc time'] = time_VF_calc

    try:
        sky_vf_domain = envir_vf['dense_svf'][coo_idcs_under_sensor[idcs_sensors_domain], 3]

        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'perc off'] = off_perc
        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'min_VF'] = min_vf
        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'nnz'] = len(envir_vf['coo_triu']) * 2

        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'sky_vf_min'] = sky_vf_domain.min()
        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'sky_vf_max'] = sky_vf_domain.max()
        VF_df.loc[(hw, CONFIG['RES'], CONFIG['TREES']), 'sky_vf_mean'] = sky_vf_domain.mean()

        VF_df.sort_index().to_csv(FOLDER /'View Factor data.csv')

    except BaseException:
        pass

    # %%% Get SVF from literature and compare to calculated SVF

    try:

        svf_sum_2d = sky_vf_sum[envir['faces_idcs'][3]].reshape(SHAPE_2D_LR)
        svf_face3_2d = envir_vf['dense_svf'][envir['faces_idcs'][3], 3].reshape(
            SHAPE_2D_LR)
    except BaseException:
        pass

    if ADDRESS != 'custom':

        svf_dsr_non = get_data.svf(
            bbox=bbox,
            folder=FOLDER,
            save_name='svf_raw',
            download_loc='data/maps/SVF/',
            rotation=ROT,
            fill=False,
            fill_max_sigma=1)

        svf_dsr_non_ma = svf_dsr_non.read(1, masked=True)
        if show_all_plots:
            plt.title(f'{address_str} \n Not-filled SVF')
            plt.imshow(svf_dsr_non_ma)
            plt.clim(0, 1)
            plt.colorbar()
            plt.show()

        load_svf = True

        # %%% Download SVF from literature (where it was calculated for 16 directions and a search radius of 100 m)
        
        if load_svf:
            svf_dsr = get_data.svf(
                bbox,
                FOLDER,
                save_name='svf',
                download_loc='data/maps/SVF',
                rotation=ROT,
                fill=True,
                fill_max_sigma=1)

            # ---- 2D plot of loaded SVF, not masked
            svf_dsr_ma = svf_dsr.read(1, masked=True)
            if show_all_plots:
                plt.imshow(svf_dsr_ma)
                plt.clim(0, 1)
                plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f'{FOLDER_RES}/svf_2d_load.png')
                plt.title(
                    f'{address_str} \n Filled SVF, ave SVF load: {svf_dsr_ma.mean():.3f}')
                plt.tight_layout()
                plt.show()

            # ---- 2D plot of loaded SVF, masked
            svf_dsr_ma = svf_dsr.read(1, masked=True)
            svf_dsr_ma.mask[~border_mask] = True
            if show_all_plots:
                plt.imshow(svf_dsr_ma)
                
                plt.clim(0, 1)
                plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)
                plt.colorbar()            
                plt.savefig(f'{FOLDER_RES}/svf_2d_load_crop.png')
                plt.tight_layout()
                plt.title(
                    f'{address_str} \n Filled SVF, ave SVF load: {svf_dsr_ma.mean():.3f}')
                plt.tight_layout()
                plt.show()

        # %%% Plot the downloaded and calculated SVF map
    
    # svf_arr = hf.coo_to_ndarray(
    #     np.column_stack((sensors['coo'][idcs_sensors_domain, :2], sky_vf_domain)),
    #     plot=False,
    #     interpolate_missing=False,
    #     shape=SHAPE_2D_LR,
    #     cmap=plt.get_cmap('viridis'))
    
    # svf_arr = hf.coo_to_ndarray(sensors['coo']['street'],
    #     plot=False,
    #     interpolate_missing=False,
    #     shape=SHAPE_2D_LR,
    #     cmap=plt.get_cmap('viridis'))
    
    if CONFIG['PLOT_SVF']:
        
        img_svf_sum, colorbar = plot2d.data_over_img(
            img=map_img_gradient_lr,
            data=svf_sum_2d,
            # data_mask=~(masks_lr['buildings']
            #             | masks_lr['trees']),
            colorbar_lims=(0, 4.5),
            # data_alpha=0,
            cmap='turbo')

        # if not sw_reflection:
        #     continue
        start = time.time()
            
    # ---- 2D plot of calculated SVF
    if '2d' in CONFIG['PLOT_SVF']:
        plt.imshow(svf_face3_2d)
        # plt.title(
        #     f'{address_str} \n average SVF on street, calc: {np.nanmean(svf_face3_2d):.3f}')
        plt.clim(0, 1)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
        plt.savefig(f'{FOLDER_RES}/svf_2d_calc.png')
        
        plt.show()

    # ---- 2.5D barplot of calculated svf
    if 'bar' in CONFIG['PLOT_SVF'] and not os.path.isfile(FOLDER_RES / 'sky_vf.png'):
        # from side

        bar.plot(
            height2d=elev_round_lr,
            img=img_svf_sum,
            gridsize=GRIDSIZE,
            res=CONFIG['RES'],
            save_path=str(FOLDER_RES_TREES / 'sky_vf'),
            plot=True,
            colorbar=colorbar,
            view_zen=30,
            view_azi=-ROT,
        )

        # from top
        bar.plot(
            height2d=elev_round_lr,
            img=img_svf_sum,
            gridsize=GRIDSIZE,
            res=CONFIG['RES'],
            save_path=str(FOLDER_RES_TREES / 'sky_vf_top'),
            plot=True,
            colorbar=colorbar,
            view_zen=0.001,
            view_azi=0,
        )

    # ---- 3D voxels of calculated svf
    if 'voxels' in CONFIG['PLOT_SVF']:
        
        plot_data = {
            'svf': {
                'data': envir_vf['dense_svf'].max(-1),
                'color': 'viridis',
                'colorbar_lims': [0,1]}}
        print('Starting voxels-plot of Sky View Factor...')
        plot3d.voxels_plot(
            height3d=height_mask3d,
            plot_data=plot_data,
            cl=envir['coo'],
            t=50,
            rot_north=ROT,
            # sun_zen_rad=sun_pos['zen_rad'],
            # sun_azi_rad=sun_pos['azi_rad'],
            el_size=CONFIG['RES'],
            view_dir=[30, ROT + (45 if ADDRESS == 'custom' else 0)],  # zen, azi
            figtitle='Sky View Factor',
            grid_visibility=0.01,
            disp_sun=False,
            shade_faces=False,
            show=False,
            saveimg=True,
            dpi=1000,
            save_path=FOLDER_RES_TREES / 'sky view factor',
            save_ext='.png',
            face_visibility=.99,
        )
                
    if CONFIG['PLOT_SVF']:         
        print('Plotting view-factor map took: ' + hf.duration(start))

    # %%% Calculate the difference between SVF from own calculations and literature
    if ADDRESS != 'custom':
        
        cutout_meters = 1
        cutout_pixels = int(cutout_meters*CONFIG['RES'])
        
        # ---- 2D plot of the delta SVF
        
        svf_loaded = svf_dsr.read(1, masked=True)
        sat_img_lr = sat_img.copy()

        scale = int(svf_loaded.shape[0] / svf_face3_2d.shape[0])
        sat_img_lr = np.dstack(hf.reduce_res_mean(
            scale, sat_img_lr[..., 0], sat_img_lr[..., 1], sat_img_lr[..., 2]))

        svf_loaded_lr = hf.reduce_res_mean(scale, svf_loaded.data)[0]

        border_mask_lr = hf.reduce_res_mean(scale, border_mask)[0]
        mask = border_mask_lr & hf.shrink_mask(
            (masks_lr["roads"]+masks_lr["concrete"]+masks_lr["grass"]), pixels=cutout_pixels, out_shape=(SHAPE_2D_LR))[0]

        svf_loaded_lr_ground = np.ma.where(mask, svf_loaded_lr, y=np.nan)

        data = (svf_face3_2d - svf_loaded_lr_ground)

        max_val = np.abs([np.nanmin(data), np.nanmax(data)]).max()
        if show_all_plots:
            plt.imshow(data, cmap='PuOr_r', vmin=-max_val, vmax=max_val)
            plt.title(
                f'SVF calculated minus SVF loaded, \nmean: {np.nanmean(data):.2f}')
            plt.colorbar()
            plt.show(block=False)
        

        # 2.5D barplot with delta SVF on the ground, where delta SVF is the bar-height
        svf_delta_img, svf_delta_colorbar = plot2d.data_over_img(
            img=sat_img_lr,
            data=data,
            # data_alpha=1,
            cmap='PuOr_r')

        if show_all_plots:
            bar.plot(
                height2d=elev_lr +
                np.where(
                    ~data.mask,
                    data,
                    0) *
                np.nanmax(elev_lr) /
                np.nanmin(data),
                img=svf_delta_img,
                gridsize=GRIDSIZE,
                aspect=(1/7),
                res=CONFIG['RES'],
                view_zen=60,
                view_azi=-ROT,
                colorbar=svf_delta_colorbar,
                dpi=500,
                saveimg=False,
                cmap='PuOr')

        data = (svf_face3_2d - svf_loaded_lr_ground)
        data_alpha = mpl.colors.Normalize(
            0, np.nanmax(
                np.abs(data)))(
            np.abs(
                data.data))**1
        data_alpha = np.where(data_alpha < 0, 0, data_alpha)
        
        # 2.5D barplot with delta SVF on the ground, plotted on the discretized map 
        svf_delta_img, svf_delta_colorbar = plot2d.data_over_img(
            img=map_img_gradient_lr,
            data=data,
            # data_alpha=0,
            cmap='PuOr_r',
            # data_alpha=data_alpha,
        )

        if show_all_plots:
            bar.plot(
                height2d=elev_round_lr,
                img=svf_delta_img,
                gridsize=GRIDSIZE,
                aspect=1 / 7,
                res=CONFIG['RES'],
                view_zen=60,
                view_azi=-ROT,
                figtitle='SVF calculated minus SVF loaded',
                colorbar=svf_delta_colorbar,
                dpi=500,
                saveimg=False,
                cmap='PuOr_r'
            ) 

        # ---- SVF on all surfaces within the given area

        scale = int(svf_dsr_ma.data.shape[0] / svf_face3_2d.shape[0])
        svf_loaded_non = hf.reduce_res_mean(scale, svf_dsr_ma.data)[0]

        mask_xyplot = ~hf.reduce_res_mean(scale, svf_dsr_ma.mask)[
            0] * sensor_mask_lr_street
        x_masked = np.ma.array(svf_loaded_non, mask=~mask_xyplot)
        y_masked = np.ma.array(svf_face3_2d, mask=~mask_xyplot)


        if show_all_plots:
            plt.imshow(x_masked)
            plt.clim(0, 1)
            plt.colorbar()
            plt.title(f'Loaded SVF \n on all surfaces \n average: {np.mean(x_masked)}')
            plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
            plt.tight_layout()
            plt.savefig(f'{FOLDER_RES}/svf_all_loaded.png')
            plt.show()
    
            plt.imshow(y_masked)
            plt.clim(0, 1)
            plt.colorbar()
            plt.title(f'Calculated SVF \n on all surfaces \n average: {np.mean(y_masked)}')
            plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
            plt.tight_layout()
            plt.savefig(f'{FOLDER_RES}/svf_all_calculated.png')
            plt.show()
            
            ax = plot2d.density_scatter(
                x=x_masked.flatten(), y=y_masked.flatten(), s=2)
            ax.set_aspect('equal')
            ax.plot([0, 1], [0, 1])
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel(r'SVF$_{\mathrm{loaded}}$')
            ax.set_ylabel(r'SVF$_{\mathrm{calculated}}$')
    
            rms = sklearn.metrics.mean_squared_error(
                x_masked.compressed(), y_masked.compressed(), squared=False)
            mae = np.abs(x_masked.compressed() - y_masked.compressed()).mean()
            ax.set_title(
                f'SVF \n on all points \n RMSE = {rms:.3f}, MAE = {mae:.3f}',
                fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{FOLDER_RES}/svf_all_errors.png')
            plt.show()

         # ---- SVF on all surfaces, except trees, within the given area

        scale = int(svf_dsr_ma.data.shape[0] / svf_face3_2d.shape[0])
        svf_loaded_non = hf.reduce_res_mean(scale, svf_dsr_ma.data)[0]

        mask2, _ = hf.shrink_mask(masks_lr["roads"] +
                                  masks_lr["grass"] +
                                  masks_lr["buildings"] +
                                  masks_lr["concrete"], pixels=cutout_pixels, out_shape=(SHAPE_2D_LR))

        mask_xyplot = ~hf.reduce_res_mean(scale, svf_dsr_ma.mask)[
            0] * mask2 * sensor_mask_lr_street
        x_masked = np.ma.array(svf_loaded_non, mask=~mask_xyplot)
        y_masked = np.ma.array(svf_face3_2d, mask=~mask_xyplot)

        if show_all_plots:

            plt.imshow(x_masked)
            plt.clim(0, 1)
            plt.colorbar()
            plt.title(f'Loaded SVF \n on all surfaces except trees \n average: {np.mean(x_masked)}')
            plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
            plt.tight_layout()
            plt.savefig(f'{FOLDER_RES}/svf_all_but_trees_loaded.png')
            plt.show()
    
            plt.imshow(y_masked)
            plt.clim(0, 1)
            plt.colorbar()
            plt.title(f'Calculated SVF \n on all surfaces except trees \n average: {np.mean(y_masked)}')
            plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
            plt.tight_layout()
            plt.savefig(f'{FOLDER_RES}/svf_all_but_trees_calculated.png')
            plt.show()
            ax = plot2d.density_scatter(
                x=x_masked.flatten(), y=y_masked.flatten(), s=2)
            ax.set_aspect('equal')
            ax.plot([0, 1], [0, 1])
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('SVF$_{\mathrm{loaded}}$')
            ax.set_ylabel('SVF$_{\mathrm{calculated}}$')
    
            rms = sklearn.metrics.mean_squared_error(
                x_masked.compressed(), y_masked.compressed(), squared=False)
            mae = np.abs(x_masked.compressed() - y_masked.compressed()).mean()
            ax.set_title(
                f'SVF \n all except trees \n RMSE = {rms:.3f}, MAE = {mae:.3f}',
                fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{FOLDER_RES}/svf_all_but_trees_errors.png')
            plt.show()
        

        # ---- SVF on all ground-level surfaces, within the given area

        scale = int(svf_dsr_ma.data.shape[0] / svf_face3_2d.shape[0])
        svf_loaded_non = hf.reduce_res_mean(scale, svf_dsr_ma.data)[0]


        meters_cutout = 2
        mask3, _ = hf.shrink_mask(masks_lr["roads"] +
                                  masks_lr["grass"] +
                                  masks_lr["concrete"], pixels=cutout_pixels, out_shape=(SHAPE_2D_LR))

        mask_xyplot = ~hf.reduce_res_mean(scale, svf_dsr_ma.mask)[
            0] * mask3 * sensor_mask_lr_street
        x_masked = np.ma.array(svf_loaded_non, mask=~mask_xyplot)
        y_masked = np.ma.array(svf_face3_2d, mask=~mask_xyplot)

        plt.imshow(x_masked)
        plt.colorbar()
        plt.clim(0, 1)        
        plt.title(f'Loaded SVF \n on all ground-level surfaces \n average: {np.mean(x_masked)}')
        plt.tick_params(left = False, right = False , labelleft = False , 
            labelbottom = False, bottom = False)
        plt.tight_layout()
        plt.savefig(f'{FOLDER_RES}/svf_ground_loaded.png')

        plt.show()

        plt.imshow(y_masked)
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(f'Calculated SVF \n on all ground-level surfaces \n average: {np.mean(y_masked)}')
        plt.tick_params(left = False, right = False , labelleft = False , 
            labelbottom = False, bottom = False)
        plt.tight_layout()
        plt.savefig(f'{FOLDER_RES}/svf_ground_calculated.png')
        plt.show()
        ax = plot2d.density_scatter(
            x=x_masked.flatten(), y=y_masked.flatten(), s=2)
        ax.set_aspect('equal')
        ax.plot([0, 1], [0, 1])
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('SVF$_{\mathrm{loaded}}$')
        ax.set_ylabel('SVF$_{\mathrm{calculated}}$')

        rms = sklearn.metrics.mean_squared_error(
            x_masked.compressed(), y_masked.compressed(), squared=False)
        mae = np.abs(x_masked.compressed() - y_masked.compressed()).mean()
        ax.set_title(
            f'SVF \n svf on ground-level surfaces \n RMSE = {rms:.3f}, MAE = {mae:.3f}',
            fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{FOLDER_RES}/svf_ground_errors.png')
        plt.show()
        
        
        # ---- Compare the error to the distance to the borders
        
        # dist = SHAPE_2D_LR[0]/2-np.abs(np.linspace(-SHAPE_2D_LR[0]/2, SHAPE_2D_LR[0]/2, SHAPE_2D_LR[0]))
        # xx, yy = np.meshgrid(dist[:, None], dist[None, :])
        
        # dist_arr = np.min(np.stack([xx,yy]), axis=0)
        
        # tmp_load = np.ma.array(svf_loaded_non, mask=~mask3)
        # tmp_calc = svf_face3_2d
        # data = np.sqrt((tmp_calc - tmp_load)**2)
        # plt.title('RMSE')
        # plt.imshow(data, cmap='cividis')
        # plt.colorbar()
        # plt.show(block=False)
        # mask = ~np.isnan(data)
        # plt.scatter(dist_arr[mask].flatten(), data[mask].flatten(), s=1)
        # plt.show()
        # plot2d.density_scatter(dist_arr[mask].flatten(), data[mask].flatten(), s=1)
        # plt.show()
        
    else:
        
        y_masked = np.ma.array(svf_face3_2d, mask=~sensor_mask_lr_street)
        
        plt.imshow(y_masked)
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(f'Calculated SVF \n on all ground-level surfaces \n average: {np.mean(y_masked)}')
        plt.tick_params(left = False, right = False , labelleft = False , 
            labelbottom = False, bottom = False)
        plt.tight_layout()
        plt.savefig(f'{FOLDER_RES}/svf_ground_calculated.png')
        plt.show()

    # %% View factors sensor elements
    
    time.sleep(1)

    # F. Lindberg, B. Holmer, S. Thorsson
    geom_fact = np.array([0.22, 0.22, 0.06, 0.06, 0.22, 0.22])
    # Solweig 1.0 – modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings
    # Int. J. Biometeorol., 52 (2008), pp. 697-713

    sensors['faces_idcs'] = 6 * [np.arange(len(sensors['coo']))]
    sensors['faces_idcs'] = np.array(sensors['faces_idcs'])

    if True:
        # ---- Load view-factor matrix
        try:
            sensors_vf['sparse_sum13'] = sparse.load_npz(
                f'{FOLDER_RES}/sensors_vf_sparse_sum13.npz')
            
            _load_len = len(sensors_vf['sparse_sum13'])
            
            if _load_len != len(sensors['coo']):
                raise Exception(f'Loaded view-factor for sensors has length {_load_len}, must be {len(sensors["coo"])}')
            
            sensors_vf['coords'] = np.load(
                f'{FOLDER_RES}/sensors_vf_coo_COMPLETE.npz')['coords']
            
            sensors_vf['data'] = np.load(
                f'{FOLDER_RES}/sensors_vf_coo_COMPLETE.npz')['data']

            sensors_vf['sparse_persons'] = sparse.load_npz(
                f'{FOLDER_RES}/sensors_vf_sparse_persons.npz')


        # ---- Calculate view-factor matrix
        except (FileNotFoundError, ValueError, Exception):
            start = time.time()
            sensors_vf['coords'], sensors_vf['data'] = vf_j.calc_matrix_mp(
                envir_from=sensors, 
                envir_to=envir, 
                min_vf_value=CONFIG['MIN_VF_VALUE'],
                max_dist=MAX_RAY_ELS,
                multiprocessing='starmap',
            )

            np.savez_compressed(
                f'{FOLDER_RES}/sensors_vf_coo_COMPLETE.npz',
                coords=sensors_vf['coords'],
                data=sensors_vf['data'])

            if CONFIG['REDUCE_ACCURACY_VF_PERC'] > 0:
                sensors_vf['coords'], sensors_vf['data'], off_perc = hf.reduce_coo_accuracy(
                    sensors_vf['coords'], sensors_vf['data'], reduce_acc=CONFIG['REDUCE_ACCURACY_VF_PERC'])
            # np.savez_compressed(
            #     f'{FOLDER_RES}/sensors_vf_coo.npz',
            #     coords=sensors_vf['coords'],
            #     data=sensors_vf['data'])

            sensors_vf['sparse'] = sparse.COO(
                coords=sensors_vf['coords'].T.astype(np.int64),
                data=sensors_vf['data'],
                shape=(len(sensors['coo']), len(envir['coo']), 6, 6))

            sensors_vf['sparse_sum13'] = sensors_vf['sparse'].sum((1, 3))
            sparse.save_npz(
                f'{FOLDER_RES}/sensors_vf_sparse_sum13',
                sensors_vf['sparse_sum13'])

            sensors_vf['sparse_persons'] = (
                sensors_vf['sparse'] * geom_fact[np.newaxis, np.newaxis, :, np.newaxis]).sum(2)
            sparse.save_npz(
                f'{FOLDER_RES}/sensors_vf_sparse_persons',
                sensors_vf['sparse_persons'])

        # # not sure about this
        # sensors_vf['coo_persons'] = sensors_vf['coo'].copy()
        # sensors_vf['coo_persons'][sensors_vf['coo_persons'][:, 2] == 0, -1] *= geom_fact[0]
        # sensors_vf['coo_persons'][sensors_vf['coo_persons'][:, 2] == 1, -1] *= geom_fact[1]
        # sensors_vf['coo_persons'][sensors_vf['coo_persons'][:, 2] == 2, -1] *= geom_fact[3]
        # sensors_vf['coo_persons'][sensors_vf['coo_persons'][:, 2] == 3, -1] *= geom_fact[4]
        # sensors_vf['coo_persons'][sensors_vf['coo_persons'][:, 2] == 4, -1] *= geom_fact[5]

        # not sure about this --- pre 12 oct 
        # sensors_vf['data_persons'] = sensors_vf['data'].copy()
        # sensors_vf['data_persons'][sensors_vf['coords']
        #                            [:, 2] == 0] *= geom_fact[0]
        # sensors_vf['data_persons'][sensors_vf['coords']
        #                            [:, 2] == 1] *= geom_fact[1]
        # sensors_vf['data_persons'][sensors_vf['coords']
        #                            [:, 2] == 2] *= geom_fact[3]
        # sensors_vf['data_persons'][sensors_vf['coords']
        #                            [:, 2] == 3] *= geom_fact[4]
        # sensors_vf['data_persons'][sensors_vf['coords']
        #                            [:, 2] == 4] *= geom_fact[5]
        
        # not sure about this --- pre 12 oct 
        sensors_vf['data_persons'] = sensors_vf['data'].copy()
        sensors_vf['data_persons'][sensors_vf['coords']
                                   [:, 2] == 0] *= geom_fact[0]
        sensors_vf['data_persons'][sensors_vf['coords']
                                   [:, 2] == 1] *= geom_fact[1]
        sensors_vf['data_persons'][sensors_vf['coords']
                                   [:, 2] == 2] *= geom_fact[2]
        sensors_vf['data_persons'][sensors_vf['coords']
                                   [:, 2] == 3] *= geom_fact[3]
        sensors_vf['data_persons'][sensors_vf['coords']
                                   [:, 2] == 4] *= geom_fact[4]
        sensors_vf['data_persons'][sensors_vf['coords']
                                   [:, 2] == 5] *= geom_fact[5]


        # if len(sensors_vf['coords_persons']) != len(sensors_vf['coords']):
        #     print("Loaded Sky View Factors for sensors is for a different sensor setup.\n Calculating new SVFs...")
        #     raise ValueError

        sensors_vf['sparse_svf'] = (
            np.ones(
                np.shape(
                    sensors_vf['sparse_sum13'])) -
            sensors_vf['sparse_sum13'])

        sensors_vf['dense_svf'] = sensors_vf['sparse_svf'].todense()
        sensors_vf['dense_svf'][:,2] = 0 # set SVF to zero at the bottom of sensor elements
        

        sensors_vf['dense_svf_person'] = (
            sensors_vf['dense_svf'] * geom_fact).sum(1)

        # if sensors_vf['sparse_persons'].density > 0.05:
        #     VF_persons_sp = VF_persons.copy()
        #     VF_persons = VF_persons.todense()

        # Plot the SVF of the sensors
        # if True:
        # VF_sensors_img = coo_to_ndarray(np.column_stack([sensors_vf['coords'][:, :2], sensors_vf['dense_svf_person']]), interpolate_missing=True, shape=SHAPE_2D_LR)
        VF_sensors_img = hf.coo_to_ndarray(np.column_stack([sensors['coo'][:, :2], sensors_vf['dense_svf_person']]), interpolate_missing=False, shape=SHAPE_2D_LR)
        
        
        if not 'sparse' in sensors_vf:
            sensors_vf['sparse'] = sparse.COO(
                coords=sensors_vf['coords'].T.astype(np.int64),
                data=sensors_vf['data'],
                shape=(len(sensors['coo']), len(envir['coo']), 6, 6))
        
        xs = sensors['coo'][sensors['idcs_dict']['street']][:, 0].astype(int)
        ys = sensors['coo'][sensors['idcs_dict']['street']][:, 1].astype(int)
        els_under_street = envir['idcs_3D'][xs, ys, 0]
        
        plt.imshow(hf.coo_to_ndarray(np.column_stack([sensors['coo'][sensors['idcs_dict']['street'], :2], sensors_vf['sparse'][sensors['idcs_dict']['street'], els_under_street, 2,3].todense()]), interpolate_missing=False, shape=SHAPE_2D_LR));\
        plt.colorbar() 
        plt.title('VF to ground')
        plt.show()
        
        del sensors_vf['sparse']
        # VF_data_img, colorbar = plot2d.data_over_img(
        #     img=map_img_gradient_lr,
        #     data=VF_sensors_img,
        #     data_mask=~(masks_lr['buildings']
        #                 | masks_lr['trees']),
        #     colorbar_lims=(0, 1),
        #     data_alpha=0,
        #     cmap='gray')
        # plt.imshow(VF_data_img)
        # plt.colorbar(colorbar)
        # plt.show(block=False)

    # %% Save all topology data

    try: 
        svf_log = pd.read_csv(FOLDER / 'svf_log.csv',   index_col=('hw', 'res'))
    except:
        svf_log = pd.DataFrame(
            # index=[hw, CONFIG['RES']],
            columns=[
                'hw',
                'res',
                'street mean',
                'street min',
                'street max', 
                'ped mean',
                'ped min',
                'ped max',
                ]).astype(float)
        
        svf_log = svf_log.set_index(['hw', 'res'])
        
    svf_log.loc[(hw, CONFIG['RES']), :] = np.nan
        
    # idcs_street_under_floating_sensor = np.intersect1d(idcs_under_cropped_area, envir['idcs_dict']['roads'])
    if len(idcs_under_cropped_area)>0: 
        svf_street = envir_vf['dense_svf'][idcs_under_cropped_area, 3]
        svf_log.loc[(hw, CONFIG['RES']), ['street min', 'street max', 'street mean']] = svf_street.min(), svf_street.max(), svf_street.mean()

    svf_pedestrian = sensors_vf['dense_svf_person'][sensors['idcs_dict']['street']]
    if len(svf_pedestrian) > 0:
        svf_log.loc[(hw, CONFIG['RES']), ['ped min', 'ped max', 'ped mean']] = svf_pedestrian.min(), svf_pedestrian.max(), svf_pedestrian.mean()
    
    svf_log.sort_index().to_csv(FOLDER / 'svf_log.csv')
    
    
    print('Average sky-view-factor on all ground is', ave_svf_ground)
    print('Average sky-view-factor on cropped ground is', svf_street.mean())
    print('Average sky-view-factor for pedestrians is', svf_pedestrian.mean())

    if FORCE_RECALC_ENVIR and sw_reflection:
        filesize = round(os.path.getsize(
            f'{FOLDER_RES}/sensors_vf_coo_COMPLETE.npz') / (1024**2), 2)

        log_dict = {
            'address': [address_str],
            'gridsize': [GRIDSIZE],
            'reduce_res': [reduce_res],
            'use_trees': [CONFIG['TREES']],
            'nr els': [len(envir['coo'])],
            'calc time': [(str(datetime.timedelta(seconds=time_VF_calc))
                           .split('.')[0])],
            'filesize (MB)': [filesize],
            'SVF ground ave': [svf_street.mean()],
            'h/w ave': [hw_ave],
            'trees [%]': [percentage_trees],
            'grass [%]': [percentage_grass],
            'water [%]': [percentage_water],
            'buildings [%]': [percentage_buildings],
            'roads [%]': [percentage_roads]}

        df_log = pd.DataFrame(log_dict)

        fn_VF_calculations = data_dir + '/' + 'VF_calculations' + '.csv'
        df_log.to_csv(fn_VF_calculations,
                      mode='a',
                      header=False if os.path.isfile(
                          fn_VF_calculations) else True,
                      index=False)

        # %% Settings and initialization for simulation

    show_time = False

    cl_corner = np.add(envir['coo'], -0.5)

    custom_zen = np.arange(0, 90, 90 / len(sun_pos))
    custom_azi = np.arange(0, 360, 360 / len(sun_pos))

    title_str = [''] * len(sun_pos)

    i_hw = 0

    # %% Plot settings
    if SIMULATE_AVERAGES:
        FOLDER_RURAL = 'demos/CUSTOM - LCZ A/res4.0/trees/log_data/rotated0.0/average%i-%i/sw_dir&sw_diff&refl&lw&cond&conv&antr&tree_eq_air/' % tuple(CONFIG['AVERAGE_YEARS'])
    else:
        FOLDER_RURAL = 'demos/Rural grass reference/hw1.0/res1.0/no_trees/log_data/rotated0.0/sw_dir&sw_diff&refl&lw&cond&conv&antr&tree_eq_air/'

    if UHI_BOOL:
        utci_rural = (pd.read_csv(
            FOLDER_RURAL + 'utci_log.csv',
            index_col='Unnamed: 0',
            parse_dates=['Unnamed: 0'])
            .asfreq(FREQ_STR).astype(float).interpolate()['mean'])

        T_mr_rural = (pd.read_csv(
            FOLDER_RURAL + 'T_mr_log.csv',
            index_col='Unnamed: 0',
            parse_dates=['Unnamed: 0'])
            .asfreq(FREQ_STR).astype(float).interpolate()['mean'])

    shape_dashboard = (4, 1)

    # %%% Select some times to plot

    # allow ALL times to be plotted
    plot_bools = sun_pos.index.values.astype(bool)
    
    if not any([bool(plot_time) for plot_time in [CONFIG['PLOT_DAYS'], CONFIG['PLOT_HOURS'], CONFIG['PLOT_MINUTES'], CONFIG['PLOT_TIME']]]):
        plot_bools[:] = False

    # at a specific day of the month
    if CONFIG['PLOT_DAYS']:
        plot_bools *= np.sum([sun_pos.index.day == _day for _day in CONFIG['PLOT_DAYS']], 0).astype(bool)
        
    # at a specific hours of the day
    if CONFIG['PLOT_HOURS']:
        plot_bools *= np.sum([sun_pos.index.hour == _hour for _hour in CONFIG['PLOT_HOURS']], 0).astype(bool)
    
    # at a specific minute of the hour
    if CONFIG['PLOT_MINUTES']:
        plot_bools *= np.sum([sun_pos.index.minute == _minute for _minute in CONFIG['PLOT_MINUTES']], 0).astype(bool)

    if CONFIG['PLOT_TIME']:
        plot_bools *= np.sum([sun_pos.index.strftime("%H:%M") == _time for _time in CONFIG['PLOT_TIME']], 0).astype(bool)
        
    plot_times = list(sun_pos.index[plot_bools].astype(str))

    data_input = CONFIG['PLOT_DATA_INPUT']
    data_colorbar_lims = CONFIG['PLOT_DATA_COLORBAR_LIMS']
    data_cmap = CONFIG['PLOT_DATA_CMAP']

    sensor_inputs = CONFIG['PLOT_SENSOR_DATA_INPUT']
    sensor_colorbar_lims = [None if cbar == 'None' else cbar for cbar in CONFIG['PLOT_SENSOR_COLORBAR_LIMS']]

    
    sensor_cmaps = CONFIG['PLOT_SENSOR_CMAP']
    sensor_custom_colorbar = True

    # %%% Create folders for saving plots
    
    for plot_type in CONFIG['PLOT_DATA']:
        fp_dataplot = f'{FOLDER_RES}/{data_input}'
        if not os.path.isdir(fp_dataplot):
            os.mkdir(fp_dataplot)

        fp_dataplot += f'/rotated{ROT}'
        if not os.path.isdir(fp_dataplot):
            os.mkdir(fp_dataplot)

        fp_dataplot += f'/{plot_type}'
        if not os.path.isdir(fp_dataplot):
            os.mkdir(fp_dataplot)

    for plot_type in CONFIG['PLOT_SENSOR_DATA']:
        for sensor_input in sensor_inputs:
            fp_sensorplot = f'{FOLDER_RES}/{sensor_input}'
            if not os.path.isdir(fp_sensorplot):
                os.mkdir(fp_sensorplot)

            fp_sensorplot += f'/rotated{ROT}'
            if not os.path.isdir(fp_sensorplot):
                os.mkdir(fp_sensorplot)

            fp_sensorplot += f'/{plot_type}'
            if not os.path.isdir(fp_sensorplot):
                os.mkdir(fp_sensorplot)

    for plot_type in CONFIG['PLOT_SHADE']:
        fp_shadeplot = f'{FOLDER_RES}/shade'
        if not os.path.isdir(fp_shadeplot):
            os.mkdir(fp_shadeplot)

        fp_shadeplot += f'/rotated{ROT}'
        if not os.path.isdir(fp_shadeplot):
            os.mkdir(fp_shadeplot)

        fp_shadeplot += f'/{plot_type}'
        if not os.path.isdir(fp_shadeplot):
            os.mkdir(fp_shadeplot)

    # %%  Set which data to log and save

    x_cross_slice = (slice(None), int(SHAPE_2D_LR[0]/2), slice(0, None))

    save_ave_data = ["q_sw['in_diff']",
                     "q_sw['in_dir']",
                     "q_sw['in']",
                     # "q_sw['in_from_sky']",
                     # "q_sw['out_els']",
                      "q_sw['out_els_refl']",
                     "q_sw['in_from_els']",
                      "q_sw['abs']",
                      "q_lw['net']",
                     # "q_lw['in']",
                      "q_lw['in_from_els']",
                      "q_lw['in_from_sky']",
                      "q_lw['out_rad']",
                     # "q_lw['out_els']",
                     # "q_lw['out_refl']",
                      "q_lw['abs']",
                      "q_convection",
                     #  "q_anthropogenic",
                      "T"
                     ]
    
    save_ave_data_sensors = ["q_sw_sensors['in_diff']",
                     "q_sw_sensors['in_dir']",
                     "q_sw_sensors['in']",
                      # "q_sw_sensors['in_from_sky']",
                     # "q_sw_sensors['out_els']",
                     # "q_sw_sensors['out_els_refl']",
                     "q_sw_sensors['in_from_els']",
                     # "q_sw_sensors['abs']",
                     # "q_lw_sensors['net']",
                     "q_lw_sensors['in']",
                     "q_lw_sensors['in_from_els']",
                     "q_lw_sensors['in_from_sky']",
                     # "q_lw_sensors['out_rad']",
                     # "q_lw_sensors['net']",
                     # "q_lw_sensors['out_els']",
                     # "q_lw_sensors['out_refl']",
                     # "q_lw_sensors['abs']"
                     "T_mr",
                     
                     ]

    if 'north' in CONFIG['ADDRESS_STR'].lower() or 'model 1' in CONFIG['ADDRESS_STR'].lower():
        slice_name = ' (y-slice)'
    elif 'east' in CONFIG['ADDRESS_STR'].lower() or 'model 2' in CONFIG['ADDRESS_STR'].lower():
        slice_name = ' (x-slice)'
    else:
        slice_name = ''
    

    save_raw_dict = {
        f"canyon street{slice_name}": {
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        "street": {
            "T",
            },
        'ground cropped': {
            'T',
            },
        f"north-facing wall{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        f"south-facing wall{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        f"east-facing wall{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        f"west-facing wall{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        f"north roof{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        f"south roof{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        f"east roof{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        f"west roof{slice_name}": {
            # "q_sw['in_dir']",
            "q_sw['out_els_refl']",
            "q_sw['in']", 
            "q_sw['in_from_els']",
            "q_sw['in_diff']",
            "q_sw['in_dir']",
            "q_lw['in_from_els']", 
            "q_lw['in_from_sky']", 
            "q_lw['out_rad']", 
            "q_lw['net']",
            "q_sw['abs']",
            "T",
            "q_convection",
            },
        
        
    
    
        f'bottom{slice_name}': {
            "q_sw_sensors['in_dir']",
            "q_sw_sensors['in']"},
        f'street{slice_name}': {
            'T_mr'},
        'grass{slice_name}': {
            'T_mr'}
        }
    if 'chimney' in str(FOLDER).lower():
        save_raw_dict.update({'east-facing wall': {'T'},
                              'west-facing wall': {'T'},
                              'north-facing wall': {'T'},
                              'south-facing wall': {'T'},
                              'upwards-facing roof': {'T'},
                              })

    if 'canyon' in CONFIG['ADDRESS_STR'].lower() and hw != 0:
        
        
        if 'north' in CONFIG['ADDRESS_STR'].lower() or 'model 1' in CONFIG['ADDRESS_STR'].lower():
            idcs_canyon_wall_face_0, idcs_canyon_wall_face_1 = np.unique(np.where((envir_vf['sparse'][:, :, 0, 1] > 0)), axis=1)
            idcs_canyon_wall_face_0 = np.unique(idcs_canyon_wall_face_0)
            idcs_canyon_wall_face_1 = np.unique(idcs_canyon_wall_face_1)
            
            wall_face_0_x = envir['coo_int'][idcs_canyon_wall_face_0[0], 0]
            wall_face_1_x = envir['coo_int'][idcs_canyon_wall_face_1[0], 0]
            idcs_canyon = envir['idcs_3D_masked'][(slice(wall_face_1_x, wall_face_0_x + 1), slice(None), slice(None))].compressed()
            idcs_slice_y = idcs_under_floating_sensor 
            
            idcs_roof_west = envir['idcs_3D_masked'][(slice(0, wall_face_1_x+1), slice(None), slice(envir['coo_int'][..., 2].max(), envir['coo_int'][..., 2].max()+1))].compressed()
            idcs_roof_east = envir['idcs_3D_masked'][(slice(wall_face_0_x, None), slice(None), slice(envir['coo_int'][..., 2].max(), envir['coo_int'][..., 2].max()+1))].compressed()
            idcs_ground = envir['idcs_3D_masked'][(slice(None), slice(None), slice(0, 1))].compressed()
            idcs_above_ground = envir['idcs_3D_masked'][(slice(None), slice(None), slice(0, 1))].compressed()
            
            idcs_canyon_ground_sliced = reduce(np.intersect1d, (idcs_ground, idcs_slice_y, idcs_canyon))
            idcs_canyon_walls_sliced = reduce(np.intersect1d, (idcs_ground, idcs_slice_y, idcs_above_ground))
            
            idcs_canyon_wall_face_1_sliced = reduce(np.intersect1d, (idcs_slice_y, idcs_canyon_wall_face_1))
            idcs_canyon_wall_face_0_sliced = reduce(np.intersect1d, (idcs_slice_y, idcs_canyon_wall_face_0))
            idcs_roof_west_sliced = reduce(np.intersect1d, (idcs_slice_y, idcs_roof_west))
            idcs_roof_east_sliced = reduce(np.intersect1d, (idcs_slice_y, idcs_roof_east))
            
            results_to_log = {
                f"east-facing wall{slice_name}": {'idcs': list(idcs_canyon_wall_face_1_sliced), 'faces': [1]},
                f"west-facing wall{slice_name}": {'idcs': list(idcs_canyon_wall_face_0_sliced), 'faces': [0]},
                f"canyon street{slice_name}": {'idcs': list(idcs_canyon_ground_sliced), 'faces': [3]},
                f"west roof{slice_name}": {'idcs': list(idcs_roof_west_sliced), 'faces': [3]},
                f"east roof{slice_name}": {'idcs': list(idcs_roof_east_sliced), 'faces': [3]},
            }
            
            idcs_sensors_sliced_y = sensors['idcs_3D_masked'][(slice(None), int(SHAPE_2D_LR[0]/2), slice(None))].compressed()
            
            sensors_to_log = {
                f'top{slice_name}': {'idcs': list(idcs_sensors_sliced_y), 'faces': [3]},
                f'bottom{slice_name}': {'idcs': list(idcs_sensors_sliced_y), 'faces': [2]}}
            
        elif 'east' in CONFIG['ADDRESS_STR'].lower() or 'model 2' in CONFIG['ADDRESS_STR'].lower():
            idcs_canyon_wall_face_4, idcs_canyon_wall_face_5 = np.unique(np.where((envir_vf['sparse'][:, :, 4, 5] > 0)), axis=1)
            idcs_canyon_wall_face_4 = np.unique(idcs_canyon_wall_face_4)
            idcs_canyon_wall_face_5 = np.unique(idcs_canyon_wall_face_5)
            
            wall_face_4_y = envir['coo_int'][idcs_canyon_wall_face_4[0], 1]
            wall_face_5_y = envir['coo_int'][idcs_canyon_wall_face_5[0], 1]
            idcs_canyon = envir['idcs_3D_masked'][(slice(None), slice(wall_face_5_y, wall_face_4_y + 1), slice(None))].compressed()
            idcs_slice_x = idcs_under_floating_sensor 
            
            idcs_roof_north = envir['idcs_3D_masked'][(slice(None), slice(0, wall_face_5_y+1), slice(envir['coo_int'][..., 2].max(), envir['coo_int'][..., 2].max()+1))].compressed()
            idcs_roof_south = envir['idcs_3D_masked'][(slice(None), slice(wall_face_4_y, None), slice(envir['coo_int'][..., 2].max(), envir['coo_int'][..., 2].max()+1))].compressed()
            idcs_ground = envir['idcs_3D_masked'][(slice(None), slice(None), slice(0, 1))].compressed()
            idcs_above_ground = envir['idcs_3D_masked'][(slice(None), slice(None), slice(0, 1))].compressed()
            
            idcs_canyon_ground_sliced = reduce(np.intersect1d, (idcs_ground, idcs_slice_x, idcs_canyon))
            idcs_canyon_walls_sliced = reduce(np.intersect1d, (idcs_ground, idcs_slice_x, idcs_above_ground))
            
            idcs_canyon_wall_face_4_sliced = reduce(np.intersect1d, (idcs_slice_x, idcs_canyon_wall_face_4))
            idcs_canyon_wall_face_5_sliced = reduce(np.intersect1d, (idcs_slice_x, idcs_canyon_wall_face_5))
            idcs_roof_north_sliced = reduce(np.intersect1d, (idcs_slice_x, idcs_roof_north))
            idcs_roof_south_sliced = reduce(np.intersect1d, (idcs_slice_x, idcs_roof_south))
            
            results_to_log = {
                f"south-facing wall{slice_name}": {'idcs': list(idcs_canyon_wall_face_5_sliced), 'faces': [5]},
                f"north-facing wall{slice_name}": {'idcs': list(idcs_canyon_wall_face_4_sliced), 'faces': [4]},
                f"canyon street{slice_name}": {'idcs': list(idcs_canyon_ground_sliced), 'faces': [3]},
                f"north roof{slice_name}": {'idcs': list(idcs_roof_north_sliced), 'faces': [3]},
                f"south roof{slice_name}": {'idcs': list(idcs_roof_south_sliced), 'faces': [3]},
            }
            
            idcs_sensors_sliced_x = sensors['idcs_3D_masked'][(int(SHAPE_2D_LR[0]/2), slice(None), slice(None))].compressed()
            
            sensors_to_log = {
                f'top{slice_name}': {'idcs': list(idcs_sensors_sliced_x), 'faces': [3]},
                f'bottom{slice_name}': {'idcs': list(idcs_sensors_sliced_x), 'faces': [2]}}
            
        elif 'model 3' in CONFIG['ADDRESS_STR'].lower():
            results_to_log = {}
            
            sensors_to_log = {
                f'top{slice_name}': {'idcs': list(sensors['faces_idcs'][3]), 'faces': [3]},
                f'bottom{slice_name}': {'idcs': list(sensors['faces_idcs'][2]), 'faces': [2]}}
            
        else:
            raise Exception('Please specify ADDRESS_STR as "North-South"- or "East-West"-oriented in the config file')
        
    elif 'well' in CONFIG['ADDRESS_STR'].lower() and hw != 0:
        
        wall_face_0_idcs, wall_face_1_idcs = np.unique(np.where((envir_vf['sparse'][:, :, 0, 1] > 0)), axis=1)
        wall_face_0_idcs = np.unique(wall_face_0_idcs)
        wall_face_1_idcs = np.unique(wall_face_1_idcs)
        wall_face_0_x = envir['coo_int'][wall_face_0_idcs[0], 0]
        wall_face_1_x = envir['coo_int'][wall_face_1_idcs[0], 0]
        
        wall_face_4_idcs, wall_face_5_idcs = np.unique(np.where((envir_vf['sparse'][:, :, 4, 5] > 0)), axis=1)
        wall_face_4_idcs = np.unique(wall_face_4_idcs)
        wall_face_5_idcs = np.unique(wall_face_5_idcs)
        wall_face_4_y = envir['coo_int'][wall_face_4_idcs[0], 1]
        wall_face_5_y = envir['coo_int'][wall_face_5_idcs[0], 1]
        
        slice_x = slice(wall_face_1_x, wall_face_0_x + 1)
        slice_y = slice(wall_face_5_y, wall_face_4_y + 1)
        
        ground_idcs = np.unique(np.where((envir_vf['sparse'][:, :, 3, :] > 0).sum(-1))[0])
        
        idcs_well = envir['idcs_3D_masked'][(slice_x, slice_y, slice(None))].compressed()
        idcs_well_ground = reduce(np.intersect1d, (idcs_well, ground_idcs))
        
        results_to_log = {
            "east-facing wall": {'idcs': list(wall_face_1_idcs), 'faces': [1]},
            "west-facing wall": {'idcs': list(wall_face_0_idcs), 'faces': [0]},
            "north-facing wall": {'idcs': list(wall_face_4_idcs), 'faces': [4]},
            "south-facing wall": {'idcs': list(wall_face_5_idcs), 'faces': [5]},
            "canyon street": {'idcs': list(idcs_well_ground), 'faces': [3]},
        }
        
    elif hw == 0 and 'REAL' not in str(FOLDER):
        
        idcs_slice_x = envir['idcs_3D_masked'][(int(SHAPE_2D_LR[0]/2), slice(None), slice(None))].compressed()
        
        idcs_canyon_ground_sliced = reduce(np.intersect1d, (coo_idcs_under_sensor, idcs_slice_x))        
        
        results_to_log = {
            f"canyon street{slice_name}": {'idcs': list(idcs_under_floating_sensor), 'faces': [3]},
            'grass': {'idcs': envir['idcs_dict']['grass'], 'faces': [3]}
            }
        
        idcs_sensors_sliced_x = sensors['idcs_3D_masked'][(int(SHAPE_2D_LR[0]/2), slice(None), slice(None))].compressed() # does not have to be x-sliced actually
        
        sensors_to_log = {
            f'top{slice_name}': {'idcs': list(idcs_sensors_sliced_x), 'faces': [3]},
            f'bottom{slice_name}': {'idcs': list(idcs_sensors_sliced_x), 'faces': [2]},
            }
        
    elif 'single' in CONFIG['ADDRESS_STR'].lower():
        
        wall_face_0_idcs, _ = np.unique(np.where((envir_vf['sparse'][:, :, 0, 3] > 0)), axis=1)
        wall_face_0_idcs = np.unique(wall_face_0_idcs)
        
        wall_face_1_idcs, _ = np.unique(np.where((envir_vf['sparse'][:, :, 1, 3] > 0)), axis=1)
        wall_face_1_idcs = np.unique(wall_face_1_idcs)
        
        wall_face_4_idcs, _ = np.unique(np.where((envir_vf['sparse'][:, :, 4, 3] > 0)), axis=1)
        wall_face_4_idcs = np.unique(wall_face_4_idcs)
        
        wall_face_5_idcs, _ = np.unique(np.where((envir_vf['sparse'][:, :, 5, 3] > 0)), axis=1)
        wall_face_5_idcs = np.unique(wall_face_5_idcs)
        
        
        idcs_roof = reduce(np.intersect1d, (envir['idcs_dict']['buildings'], envir['faces_idcs'][3]))
        
        idcs_outer = reduce(np.union1d, [envir['faces_idcs'][0], envir['faces_idcs'][1], envir['faces_idcs'][4], envir['faces_idcs'][5]])
        idcs_roof_inner = np.setdiff1d(idcs_roof, idcs_outer)
        
        
        results_to_log = {
            "east-facing wall": {'idcs': list(wall_face_1_idcs), 'faces': [1]},
            "west-facing wall": {'idcs': list(wall_face_0_idcs), 'faces': [0]},
            "north-facing wall": {'idcs': list(wall_face_4_idcs), 'faces': [4]},
            "south-facing wall": {'idcs': list(wall_face_5_idcs), 'faces': [5]},
            "upwards-facing roof": {'idcs': list(idcs_under_floating_sensor), 'faces':[3]}, # not the roof but just the idcs under floating sensors
            }
        
        idcs_sensors_sliced_x = sensors['idcs_3D_masked'][(int(SHAPE_2D_LR[0]/2), slice(None), slice(None))].compressed() # does not have to be x-sliced actually
        
        sensors_to_log = {
            f'top{slice_name}': {'idcs': list(sensors['faces_idcs'][3]), 'faces': [3]},
            f'bottom{slice_name}': {'idcs': list(sensors['faces_idcs'][2]), 'faces': [2]}}
        
    elif 'REAL' in str(FOLDER):
        results_to_log = {
            "ground": {'idcs': list(idcs_under_floating_sensor), 'faces': [3]},
            "ground cropped": {'idcs': list(idcs_under_cropped_area), 'faces': [3]},
            'grass': {'idcs': envir['idcs_dict']['grass'], 'faces': [3]},
            'roads': {'idcs': envir['idcs_dict']['roads'], 'faces': [3]},
            # 'walls': {'idcs': envir['idcs_dict']['walls'], 'faces': [0,1,4,5]},
            }
        
    try:
        sensors_to_log = {
            f'floating top{slice_name}': {'idcs': sensors['idcs_dict']['floating'], 'faces': [3]},
            f'floating bottom{slice_name}': {'idcs': sensors['idcs_dict']['floating'], 'faces': [2]},
            }
    except:
        pass
    
    try:
        sensors_to_log.update({f'street{slice_name}': {'idcs': sensors['idcs_dict']['street'], 'faces': [0,1,2,3,4,5]}})
    except:
        pass
    try:
        sensors_to_log.update({f'grass{slice_name}': {'idcs': sensors['idcs_dict']['grass'], 'faces': [0,1,2,3,4,5]}})
    except:
        pass
                                   
        


    plot_vars = ["q_sw_in_diff",
                 "q_sw['in_dir']",
                 "q_sw_in_from_els",
                 "q_sw_abs",
                 "q_lw['in_from_els']",
                 "q_lw['in_from_sky",
                 "q_lw['out_rad",
                 "q_convection",
                 "q_anthropogenic",
                 "T",
                 "knmi_T",
                 "knmi_FH",
                 "knmi_Q"]

    
    df_columns = pd.MultiIndex.from_product([['nr els', 'results sum', 'results raw', 'simulation steps', 'external forcings'], [1.0], [1.0], [0.4], [0.85], [0.25], [2000000], [0], [''], [''], [1.0]], 
                                               names=['category', 'emissivity sky', 'hw', 'albedo', 'emissivity', 'skin thickness', 'vol. heat cap.', 'wind speed', 'location', 'variable', 'res'])
    
    if CONFIG['SOLAR_ANGLES']['location']:
        df_index_now = pd.MultiIndex.from_arrays((
            sun_pos[dates_str[0]:dates_str[-1]].index, 
            sun_pos['azi_deg TRUE'][dates_str[0]:dates_str[-1]], 
            sun_pos['zen_deg'][dates_str[0]:dates_str[-1]]), names=['datetime', 'azimuth', 'zenith'])
    
    else:
        df_index_now = pd.MultiIndex.from_product([
            # [0], list(range(90))],
            CONFIG['SOLAR_ANGLES']['azimuth'][0], CONFIG['SOLAR_ANGLES']['zenith'][0]],
            names=['azimuth', 'zenith'])
    
    
    # make dataframe that will store the results per zenith and azimuth solar angle (and per timestep)
    # results_log_now = pd.DataFrame(
    #     index=df_index_now,
    #     columns=df_columns)
    
    try:
        if CONFIG['SOLAR_ANGLES']['location']:
            results_log_load = pd.read_pickle(FOLDER / 'results_log.pkl')
            df_index_load = results_log_load.index
            df_index_add = df_index_now.difference(df_index_load)
            
            if len(df_index_add) > 0:
                results_log_add = pd.DataFrame(
                    index=df_index_add,
                    columns=df_columns)
                
                results_log = pd.concat([results_log_load, results_log_add])
            else:
                results_log = results_log_load
            
            
        else:
            results_log_load = pd.read_pickle(FOLDER / 'results_log.pkl')
            idx_names = ['datetime', 'azimuth', 'zenith']

            new_idcs_datetime = False
            
            # add missing indices
            new_idcs_zen = list(set(sun_pos['zen_deg'].unique()).difference(results_log.index.get_level_values('zenith')))
            new_idcs_azi = list(set(sun_pos['azi_deg TRUE'].unique()).difference(results_log.index.get_level_values('azimuth')))
            
            if new_idcs_zen:
                add_indices = pd.MultiIndex.from_product(
                    [results_log.index.unique('azimuth'), new_idcs_zen],
                    names=idx_names)
                
                results_log_add = pd.DataFrame(index=add_indices, columns=results_log.columns)
                results_log = pd.concat([results_log, results_log_add]).sort_index()
                
            if new_idcs_azi:
                add_indices = pd.MultiIndex.from_product(
                    [new_idcs_azi, results_log.index.unique('zenith')],
                    names=idx_names)
                
                results_log_add = pd.DataFrame(index=add_indices, columns=results_log.columns)
                results_log = pd.concat([results_log, results_log_add]).sort_index()
        
        results_log = results_log.sort_index()
        
    except Exception as e:
        print(e)
        
        results_log = pd.DataFrame(
            index=df_index_now,
            columns=df_columns)
        
    for idcs_loc, idcs_list in results_to_log.items():
        results_log.loc[:, ('nr els', '', hw, '', '', '', '', '', idcs_loc, '', CONFIG['RES'])] = len(idcs_list['idcs'])
        
    for idcs_loc, idcs_list in sensors_to_log.items():
        results_log.loc[:, ('nr els', '', hw, '', '', '', '', '', idcs_loc, '', CONFIG['RES'])] = len(idcs_list['idcs'])
    
    data_log_bool = False
    if data_log_bool:
        try:
            data_log = pd.read_pickle(
                FOLDER_RES_LOG_ROT_CC_WS / 'data_log.pkl').asfreq(FREQ_STR).astype(float).interpolate()
            if not str(data_log.index[-1]) == (dates_str[-1] + ' 23:45:00'):
                raise ValueError('Loaded data_log.pkl has incorrect end-date, so a new empty DataFrame is generated.')
        except (BaseException, ValueError):
            header = pd.MultiIndex.from_product([list(ave_idcs.keys()),
                                                 save_ave_data],
                                                names=['idcs', 'data'])
            data_log = pd.DataFrame(
                index=pd.date_range(
                    dates_str[0],
                    dates_str[-1] + ' 23:59:00',
                    freq=FREQ_STR),
                columns=header)
            
        try:
            data_log_sensors = pd.read_pickle(
                FOLDER_RES_LOG_ROT_CC_WS / 'data_log_sensors.pkl').asfreq(FREQ_STR).astype(float).interpolate()
            if not str(data_log_sensors.index[-1]) == (dates_str[-1] + ' 23:45:00'):
                raise ValueError('Loaded data_log_sensor.pkl has incorrect end-date, so a new empty DataFrame is generated.')
        except (BaseException, ValueError):
            header = pd.MultiIndex.from_product([['top', 'bottom'],
                                                 save_ave_data_sensors],
                                                names=['idcs', 'data'])
            data_log_sensors = pd.DataFrame(
                index=pd.date_range(
                    dates_str[0],
                    dates_str[-1] + ' 23:59:00',
                    freq=FREQ_STR),
                columns=header)

    try:
        utci_load = pd.read_csv(FOLDER_RES_LOG_ROT_CC_WS + 'utci_log.csv',
                                index_col='Unnamed: 0',
                                parse_dates=['Unnamed: 0'])

        utci_log = utci_load.astype(float)
        utci_log.index = pd.to_datetime(utci_log.index)

    except BaseException:
        utci_log = pd.DataFrame(
            # index=pd.date_range(
            #     '2020-01-01',
            #     '2021-12-31 23:59',
            # freq=FREQ_STR),
            columns=[
                'mean',
                'min',
                'max']).astype(float)


    if UHI_BOOL:
        uhi_log = utci_log.subtract(utci_rural, axis='index')

    shadow_methods = ['rt']

    loaded_q_sw_in = []
    loaded_sunny_els = []
    
    if not os.path.isdir(FOLDER_RES_LOG / 'sunlit_els'):
        os.mkdir(FOLDER_RES_LOG / 'sunlit_els')

    print('\n\n Starting solar- and heat calculations...\n\n')
    print('Address = ', address_str)
    if ADDRESS == 'custom':
        print('Height/width = ', hw)
    print('Rotation = ', ROT)
    print('Resolution = ', CONFIG['RES'])
    
    # %% Initialize empty numpy arrays to store data

    utci_means = []
    utci_mins = []
    utci_maxs = []

    empty_data_array = np.zeros((STEPS_PER_DAY, len(envir['coo'])))
    empty_data_array_faces = np.zeros((STEPS_PER_DAY, len(envir['coo']), 6))

    sunny_els_arr = empty_data_array.copy().astype(int)
    sunny_sens_arr = empty_data_array.copy()

    q_sw = {}
    q_sw['in_dir'] = np.copy(empty_data_array_faces)
    q_sw['in_diff'] = np.copy(empty_data_array_faces)
    q_sw['in'] = np.copy(empty_data_array_faces)
    q_sw['in_from_sky'] = np.copy(empty_data_array_faces)
    q_sw['in_from_els'] = np.copy(empty_data_array_faces)
    q_sw['out_els_refl'] = np.copy(empty_data_array_faces)
    # q_sw['out_els'] = np.copy(empty_data_array_faces)
    q_sw['abs'] = np.copy(empty_data_array_faces)

    q_lw = {}
    q_lw['net'] = np.copy(empty_data_array_faces)
    q_lw['in_from_els'] = np.copy(empty_data_array_faces)
    q_lw['in_from_sky'] = np.copy(empty_data_array_faces)
    q_lw['in'] = np.copy(empty_data_array_faces)
    q_lw['out_sky'] = np.zeros(STEPS_PER_DAY)
    q_lw['out_rad'] = np.copy(empty_data_array_faces)
    q_lw['out_els'] = np.copy(empty_data_array_faces)
    q_lw['out_refl'] = np.copy(empty_data_array_faces)
    q_lw['abs'] = np.copy(empty_data_array_faces)

    empty_sensors_data_faces = np.zeros(
        (STEPS_PER_DAY, len(sensors['coo']), 6))
    empty_sensors_data = np.zeros((STEPS_PER_DAY, len(sensors['coo'])))
    q_sw_sensors = {}
    q_sw_sensors['in_dir'] = np.copy(empty_sensors_data_faces)
    q_sw_sensors['in_diff'] = np.copy(empty_sensors_data_faces)
    q_sw_sensors['in'] = np.copy(empty_sensors_data_faces)
    q_sw_sensors['in_from_sky'] = np.copy(empty_sensors_data_faces)
    q_sw_sensors['in_from_els'] = np.copy(empty_sensors_data_faces)
    q_sw_sensors['out_els_refl'] = np.copy(empty_sensors_data_faces)
    q_sw_sensors['out_els'] = np.copy(empty_sensors_data_faces)
    q_sw_sensors['abs'] = np.copy(empty_sensors_data_faces)

    q_lw_sensors = {}
    q_lw_sensors['net'] = np.copy(empty_sensors_data_faces)
    q_lw_sensors['in_from_els'] = np.copy(empty_sensors_data_faces)
    q_lw_sensors['in_from_sky'] = np.copy(empty_sensors_data_faces)
    q_lw_sensors['in'] = np.copy(empty_sensors_data_faces)
    q_lw_sensors['out_sky'] = np.zeros(STEPS_PER_DAY)
    q_lw_sensors['out_rad'] = np.copy(empty_sensors_data_faces)
    q_lw_sensors['out_els'] = np.copy(empty_sensors_data_faces)
    q_lw_sensors['out_refl'] = np.copy(empty_sensors_data_faces)
    q_lw_sensors['abs'] = np.copy(empty_sensors_data_faces)

    q_sw_persons = {}
    q_sw_persons['in_dir'] = np.copy(empty_sensors_data)
    q_sw_persons['in_diff'] = np.copy(empty_sensors_data)
    q_sw_persons['in'] = np.copy(empty_sensors_data)
    q_sw_persons['in_from_sky'] = np.copy(empty_sensors_data)
    q_sw_persons['in_from_els'] = np.copy(empty_sensors_data)
    q_sw_persons['out_els_refl'] = np.copy(empty_sensors_data)
    q_sw_persons['out_els'] = np.copy(empty_sensors_data)
    q_sw_persons['abs'] = np.copy(empty_sensors_data)

    q_lw_persons = {}
    q_lw_persons['net'] = np.copy(empty_sensors_data)
    q_lw_persons['in_from_els'] = np.copy(empty_sensors_data)
    q_lw_persons['in_from_sky'] = np.copy(empty_sensors_data)
    q_lw_persons['in'] = np.copy(empty_sensors_data)
    q_lw_persons['out_sky'] = np.zeros(STEPS_PER_DAY)
    q_lw_persons['out_rad'] = np.copy(empty_sensors_data)
    q_lw_persons['out_els'] = np.copy(empty_sensors_data)
    q_lw_persons['out_refl'] = np.copy(empty_sensors_data)
    q_lw_persons['abs'] = np.copy(empty_sensors_data)

    q_convection = np.copy(empty_data_array_faces)
    q_anthropogenic = np.copy(empty_data_array_faces)
    q_conduction = np.copy(empty_data_array)
    
    # mean radiant temperature initialisation
    T_mr = np.copy(empty_sensors_data)
    mean_rad_flux = np.copy(empty_sensors_data)
    utci_t = np.copy(empty_sensors_data)
    T_ngbrs = np.copy(empty_data_array)
    T_surf = np.copy(empty_data_array_faces)

    utci_means = np.zeros((STEPS_PER_DAY))
    utci_maxs = np.zeros((STEPS_PER_DAY))
    utci_mins = np.zeros((STEPS_PER_DAY))

    T = np.zeros((STEPS_PER_DAY, len(envir['coo'])))
    for surf_type in envir['idcs_dict']:
        if surf_type == 'buildings':
            continue
        T[0, envir['idcs_dict'][surf_type]] = CONFIG['T_INIT'][surf_type]

    if not CONFIG['TREES']:
        T[0, envir['idcs_dict']["trees"]] = CONFIG['T_INIT']["grass"]

    # %% Start simulation
    # for i_day, date in enumerate(dates[12:]):
    for i_day, date_obj in enumerate(tqdm(dates)):

        # set all shortwave to zero
        for key in q_sw:
            q_sw[key][:] = 0
            
        # set all longwave to zero
        for key in q_lw:
            q_lw[key][:] = 0
            
        sw_reflections_lst = np.zeros((STEPS_PER_DAY))
        lw_reflections_lst = np.zeros((STEPS_PER_DAY))

        to_next_day = False        

        # date_obj = date.copy()
        date = date_obj.strftime('%Y-%m-%d')
        month_str = date_obj.strftime('%Y-%m')
        
        _azi = sun_pos['azi_deg'][date].values
        _zen = sun_pos['zen_deg'][date].values
        
        # _alb = float(kwargs['albedo_sw']) if 'albedo_sw' in kwargs else 'custom'
        _emm = float(kwargs['emissivity']) if 'emissivity' in kwargs else 'custom'
        _res = float(kwargs['RES']) if 'RES' in kwargs else 'custom'
        
        if 'albedo_sw' in kwargs:
            _alb = float(kwargs['albedo_sw'])
        else:
            _alb = ', '.join([f'{loc}={albedo_dict[loc]}' for loc in ['roofs', 'walls', 'roads']])
            
        if 'emissivity' in kwargs:
            _emm = kwargs['emissivity']
        else:
            _emm = ', '.join([f'{loc}={emissivity_dict[loc]}' for loc in ['roofs', 'walls', 'roads']])

        if not CONFIG['FORCE_RECALC_DATA']:
            # try to load the mean radiant temperature data, if succesful, 
            # continue to next day
            try:
                if any(pd.isnull(results_log.loc[(sun_pos[date]['azi_deg TRUE'], sun_pos[date]['zen_deg']), ('results', hw, _alb, _emm, 'element', slice(None), _res)])):
                # if any(pd.isnull(T_mr_log.loc[date])) and not T_mr_log.loc[date].empty:
                    print(
                        f'Skipped ... because data was already calculated.')
                    continue
            except BaseException:
                pass

        try:
            dashboard_T_lim = (
                np.min([np.min(d) for d in [utci_rural.loc[month_str],
                                            utci_log.loc[month_str]["mean"],
                                            knmi_data["T"].loc[month_str]]]),
                np.max([np.max(d) for d in [utci_rural.loc[month_str],
                                            utci_log.loc[month_str]["mean"],
                                            knmi_data["T"].loc[month_str]]]))
        except BaseException:
            dashboard_T_lim = (0, 40)
        dashboard_T_lim = (dashboard_T_lim[0] - (dashboard_T_lim[0] % 5),
                           dashboard_T_lim[1] + (5 - dashboard_T_lim[1] % 5))

        dashboard_irr_lims = (
            0, np.max([np.max(forcings.loc[month_str]["Dir_hor"]),
                       np.max(forcings.loc[month_str]["Dir_perp"])]))

        year, month, day = tuple([int(_) for _ in date.split("-")])
        last_day = calendar.monthrange(year, month)[1]
        if day == last_day:
            plot_month = True
        else:
            plot_month = False

        try:
            if recalc_sun:
                raise Exception()
            envir_sunlit = hf.load_pickled_list(
                path=FOLDER_RES_LOG / ('sunlit_els/envir_' + date + '.npz'), 
                name='sunlit_els', length=STEPS_PER_DAY)
        except (FileNotFoundError, EOFError, Exception):
            envir_sunlit = [[] for _ in range(STEPS_PER_DAY)]

        try:
            if recalc_sun:
                raise Exception()
            # raise FileNotFoundError
            sensors_sunlit = hf.load_pickled_list(
                path=FOLDER_RES_LOG / ('sunlit_els/sensors_' + date + '.npz'), 
                name='sunlit_els', length=STEPS_PER_DAY)
        except (FileNotFoundError, EOFError, KeyError, Exception):
            sensors_sunlit = [[] for _ in range(STEPS_PER_DAY)]

         # %% New timestep    
        for t in tqdm(range(STEPS_PER_DAY)):
   
            if skip_steps:
                T[t+1] = T[t]
                if t < int(STEPS_PER_DAY * 5/6):
                    continue
                
                print(f'\n\n Fast-Forwarded to t={t} \n\n')
                skip_steps = False
   
            time_now = sun_pos.loc[date].index[t]
            time_str = time_now.strftime("%Y-%m-%d %H%:%M:%S")
            
            if sun_pos.loc[date].index[t] not in utci_log.index:
                utci_log = utci_log.append(pd.DataFrame(
                    index=[sun_pos.loc[date].index[t]]))

            azi_deg_t = (sun_pos['azi_deg'][time_now]) % 360
            azi_rad_t = np.deg2rad(azi_deg_t)
            zen_deg_t = sun_pos['zen_deg'][time_now]
            zen_rad_t = np.deg2rad(zen_deg_t)
                
            # %%% Wind speed calculation

            # windspeed varies with height
            if not CONFIG['UNIFORM_WINDSPEED'] and hw > 0:
                wind_speed_t = physics.physics.urban_wind_speed(
                    u10=weather['wind_speed'][time_now],
                    H=elev_lr[elev_lr > 2].mean(),
                    area_map=area_els * A,
                    front_dens_tree=front_dens_tree,
                    front_dens_building=front_dens_building,
                    front_dens_total=front_dens_total,
                    height=np.arange(envir['idcs_3D'].shape[2]) * CONFIG['RES'])

            # uniform windspeed throughout environment
            else:
                wind_speed_t = weather['wind_speed'][time_now]

            # %%% Radiation calculation
            
            
            # if t==6:
            #     print('t = 6')
            
            # ---- Shortwave radiation
            if sw_in_dir or sw_in_diff:
                q_sw, envir_sunlit, nr_refl = physics.physics.shortwave(
                    t=t,
                    envir_from=envir,
                    envir_to=envir,
                    q_sw=q_sw,
                    physics=CONFIG['ACTIVE_PHYSICS'],
                    albedo=materials['albedo_sw'],
                    sun_pos_t=sun_pos.loc[time_now],
                    sunlit_els=envir_sunlit,
                    shadow_method=CONFIG['SHADOW_METHOD'],
                    radiations_t=forcings.loc[time_now],
                    envir_vf=envir_vf,
                    max_reflections=20,
                    error_reflections=CONFIG['ERROR_REFLECTIONS'],
                    force_recalc=CONFIG['FORCE_RECALC_SUN'])
                
                # for _t in range()
                # if date_obj.month > 5 and len(envir_sunlit[t]) > 0:
                #     plt.title(f't={time_now}, hw={hw}')
                #     plt.imshow(np.isin(envir['idcs_3D'][:, :, 0], envir_sunlit[t]).T)
                #     plt.show()
                
                sw_reflections_lst[t] = nr_refl
            
            # ---- Longwave irradiation calculation

            if lw_in_sky or lw_out_emm or lw_out_refl:
                q_lw, nr_refl = physics.physics.longwave(
                    t=t,
                    q_lw=q_lw,
                    envir=envir,
                    envir_vf=envir_vf,
                    materials=materials,
                    physics=CONFIG['ACTIVE_PHYSICS'],
                    lw_out_sky=forcings['LW_sky'][time_now],
                    T=T,
                    # emissivity_sky=weather['emissivity_sky'][time_now],
                    # T_sky=weather['T_air'][time_now],
                    reflections=True,
                    max_reflections=50,
                    error_reflections=CONFIG['ERROR_REFLECTIONS'])
                
                lw_reflections_lst[t] = nr_refl
                
                

            # !!! not implemented correctly. Needs revision.
            if CONFIG['ACTIVE_PHYSICS']['conduction_inwards']:
                q_conduction[t, :] = physics.physics.conduction_inwards(U_i,
                                                          CONFIG['T_BUILDING_INTERIOR'],
                                                          T[t, :]) 

            # %%% Other physics calculation
            if CONFIG['ACTIVE_PHYSICS']['convection']:
                if not CONFIG['UNIFORM_WINDSPEED'] and hw > 0:
                    q_convection[t] = physics.physics.convection(wind_speed_t[envir['coo_int'][:, -1]],
                                                      weather['T_air'][time_now],
                                                      T[t],
                                                      envir['faces_lst'],
                                                      windy_faces=CONFIG['WINDY_FACES'])
                else:
                    q_convection[t] = physics.physics.convection(
                        wind_speed_t,
                        weather['T_air'][time_now],
                        T[t],
                        envir['faces_lst'],
                        windy_faces=CONFIG['WINDY_FACES'])

            if anthropogenic and len(envir['idcs_dict']["buildings"]) > 0:
                q_anthropogenic[t, envir['idcs_dict']["buildings"]] = (
                    float(anthropogenic_power_m2[time_str])
                    * envir['faces_lst'][envir['idcs_dict']["buildings"]])

            # !!!  Temperature updating. To calcualte eq. Temp. Needs to be revised when used.
            # T_surf[t] = phys.surface_equilbrium_temp(q_sw_abs=q_sw_abs[t],
            #                                          q_lw['abs=q_lw['abs[t],
            #                                          heat_transf_coeff=0,
            #                                          albedos=albedo_sw,
            #                                          emissivities=emissivity,
            #                                          SB_CONST=SB_CONST,
            #                                          T_sky_eff=T_sky_eff,
            #                                          T_air=T_air)

            # %%% Sensor data calculation 

            if CONFIG['CALC_UTCI']:

                # Solar radiation
                sensors_sunlit[t] = np.where(
                    np.isin(
                        coo_idcs_under_sensor,
                        sensors_sunlit[t]))[0]

                q_sw_sensors, sensors_sunlit, _ = physics.physics.shortwave(
                    t=t,
                    envir_from=envir, 
                    envir_to=sensors,
                    envir_vf=sensors_vf,
                    q_sw=q_sw_sensors,
                    q_sw_out_els_refl_prev_t=q_sw['out_els_refl'][t],
                    albedo=materials_person['albedo_sw'],
                    physics=CONFIG['ACTIVE_PHYSICS'],
                    sun_pos_t=sun_pos.loc[time_now],
                    sunlit_els=sensors_sunlit,
                    shadow_method=CONFIG['SHADOW_METHOD'],
                    radiations_t=forcings.loc[time_now],
                    reflections=False,
                    force_recalc=CONFIG['FORCE_RECALC_SUN'])

                for name, item in q_sw_sensors.items():
                    q_sw_persons[name][t] = (
                        q_sw_sensors[name][t] * geom_fact).sum(1)

                # Longwave radiation
                q_lw_sensors['in_from_sky'][t] = sensors_vf['dense_svf'] * q_lw['out_sky'][t]
                q_lw_sensors['in_from_els'][t] = hf.coo_multiplication(
                    coo_arr=sensors_vf['coords'],
                    vf_arr=sensors_vf['data'],
                    data=q_lw['out_els'][t],
                    nr_els=int(len(q_lw_sensors['in_from_els'][t])),
                    upper_triangular=False)
                q_lw_sensors['in'][t] = q_lw_sensors['in_from_sky'][t] + \
                    q_lw_sensors['in_from_els'][t]
                
                
                q_lw_persons['in_from_els'][t] = (
                    q_lw_sensors['in_from_els'][t] * geom_fact).sum(1)

                q_lw_persons['in_from_sky'][t] = sensors_vf['dense_svf_person'] * \
                    q_lw['out_sky'][t]
                q_lw_persons['in'][t] = q_lw_persons['in_from_sky'][t] + \
                    q_lw_persons['in_from_els'][t]

                mean_rad_flux[t] = (q_sw_persons['in'][t] *
                                 (1 -
                                  materials_person['albedo_sw']) +
                                 q_lw_persons['in'][t] *
                                 materials_person['emissivity'])
                
                if t == 95 and __debug__ and False:
                    print('__debug__ is:', __debug__)
                    
                    test=[]
                    data = q_sw['in']
                    
                    ave_on_ground = data[:, ground_idcs].sum((-1)).mean(axis=1)
                    ave_on_canyon = data[:, canyon_idcs].sum((-1)).mean(axis=1)
                    
                    for _t in list(range(0, t, 1)):
                        test.append(plot2d.cross_section(
                            data=data[_t].sum((-1)), 
                            slice_xyz=canyon_slice, 
                            idcs_3d=envir['idcs_3D'], vmax=data.max(), 
                            title=f't={_t}\n Cross section at x={int(SHAPE_2D_LR[0]/2)}\n SW in, reflected from els \n Average on canyon elements: {ave_on_canyon[_t]:.2f}, \nAverage on ground-elements {ave_on_ground[_t]:.2f}'))
                    
                    
                    plt.plot(sun_pos[date]['zen_deg'], ave_on_ground, label='ave_on_ground');\
                    plt.plot(sun_pos[date]['zen_deg'], ave_on_canyon, label='ave_on_canyon');\
                    plt.legend()
                    plt.show()
                    
                # !!! is emissivity_person needed?
                T_mr[t] = (
                    (mean_rad_flux[t] / (materials_person['emissivity'] * SB_CONST)) ** (1 / 4))

                utci_t[t] = utci.utci(
                    ta=weather['T_air'][time_now] - 273.15, tr=T_mr[t] - 273.15,
                    vel=weather['wind_speed'][time_now], rh=weather['rel_hum'][time_now])

                # filter out the sensors relevant for UHI
                utci_domain = utci_t[t][idcs_sensors_domain]
                
                # if date != dates[0]: #!!! to check: do not set values for the first day,
                utci_log.loc[time_str, 'mean'] = utci_domain.mean()
                utci_log.loc[time_str, 'max'] = utci_domain.max()
                utci_log.loc[time_str, 'min'] = utci_domain.min()
                utci_log.index = pd.to_datetime(utci_log.index)

                if UHI_BOOL:
                    uhi_log.loc[time_str, 'mean'] = float(
                        utci_log.loc[time_str, 'mean'] - utci_rural.loc[time_str])
                    uhi_log.loc[time_str, 'min'] = float(
                        utci_log.loc[time_str, 'min'] - utci_rural.loc[time_str])
                    uhi_log.loc[time_str, 'max'] = float(
                        utci_log.loc[time_str, 'max'] - utci_rural.loc[time_str])

                # T_mr_log.loc[time_str,
                #              'mean'] = T_mr[idcs_sensors_domain].mean()
                # T_mr_log.loc[time_str,
                #              'max'] = T_mr[idcs_sensors_domain].max()
                # T_mr_log.loc[time_str,
                #              'min'] = T_mr[idcs_sensors_domain].min()

            # %%% Update temperature, using fluxes

            q_net = (q_conduction[t]
                     + np.sum((q_sw['abs'][t]
                               + q_lw['net'][t]
                               + q_convection[t]
                               + q_anthropogenic[t]), 1))

            dTdt = physics.physics.dTdt(
                             vol_heat_cap=materials['vol_heat_cap'],
                             area=A,
                             thickness=materials['skin_thickness'],
                             q_net=q_net)

            if conduction_surface: #!!! needs to be checkd if correct when doing conduction AFTER delta T calc. Needs revision
                dTdt_heatdiff = physics.heat.heat_diffusion_lst(
                    T0_lst=T[t, ...], 
                    idcs_ngbrs_ma=idcs_ngbrs_ma, 
                    nr_ngbrs=nr_ngbrs,
                    gridres=CONFIG['RES'],
                    D=materials['thermal_diff'])
                dTdt += dTdt_heatdiff
                
            if not CONFIG['FIXED_TEMP']:
                if not ((t + 1) == STEPS_PER_DAY and dates[-1]==date_obj):
                    T[(t + 1) % STEPS_PER_DAY] = T[t, ...] + dTdt * CONFIG['T_STEP']
    
                    if T_trees_eqls_air:
                        T[(t + 1) % STEPS_PER_DAY, envir['idcs_dict']["trees"]] = (
                            weather['T_air'][time_now])
            else:
                T[(t + 1) % STEPS_PER_DAY] = T[t, ...]
                
                

            # %%% Plot per timestep

            # ---- Calculate data for plotting
            
            if str(time_now) in plot_times: 
                # continue

                if any([pl.startswith(('voxels_shaded'))
                        for pl in [*CONFIG['PLOT_SENSOR_DATA'], *CONFIG['PLOT_SHADE'], *CONFIG['PLOT_DATA']]]):
                    q_sw_in3d = q_sw['in'][t]
                    shade3d = hf.intensity_all_faces(array_3d=q_sw_in3d,
                                                     normalize=True,
                                                     max_intensity=1400,
                                                     min_visibility=0.3)
                    
                if any([pl.startswith(('bar', '2d', 'voxels'))
                        for pl in [*CONFIG['PLOT_SENSOR_DATA'], *CONFIG['PLOT_SHADE'], *CONFIG['PLOT_DATA']]]):
    
                    # Light intensity (scale 0-1) on the faces of the barplot.
                    # The labeling order differs from the labeling in the
                    # intensity calculation.
                    shadefaces = bar.illumination_faces(
                        math.radians(sun_pos['azi_deg'][time_now]-ROT),
                        sun_pos['zen_rad'][time_now],
                        beer_lambert=True,
                        min_visibility=0.7)
    
                    q_sw_in_top = (q_sw['in'][t, envir['faces_idcs'][3]]
                                   .max(1).reshape(SHAPE_2D_LR))
                    shade2d = hf.intensity_top_faces(array_2d=q_sw_in_top,
                                                   normalize=True,
                                                   max_intensity=1000,
                                                   min_visibility=0.7)
    
                # %%%% Plot data dashboard
    
                if CONFIG['SHOW_DASHBOARD']:
    
                    data_dashboard = {
                        "Temperature":
                            {"data":
                             {"Ambient T": knmi_data["T"][str(date)],
                              "UTCI ref. rural": utci_rural.loc[str(date)],
                              "UTCI calc.": utci_log.loc[str(date)]['mean'],
                              # "T_mr mean": T_mr_log.loc[str(date)]['mean'] - 273,
                              # "T_mr min": T_mr_log.loc[str(date)]['min'] - 273,
                              # "T_mr max": T_mr_log.loc[str(date)]['max'] - 273,
                              },
                             "lims":
                                 dashboard_T_lim},
                        "UHI":
                            {"data":
                             {"UHI mean": uhi_log['mean'].loc[str(date)],
                              "UHI fill": {"UHI min.": uhi_log['min'].loc[str(date)],
                                           "UHI max.": uhi_log['max'].loc[str(date)]}},
                             # "lims": sensor_colorbar_lims},
                             "lims": (-10, 10)},
                        "Solar irradiance":
                            {"data":
                             {"Diffuse horiz. irr": (forcings["Diff_hor"]
                                                     .loc[str(date)]),
                              "Direct perp. irr": (forcings["Dir_perp"].loc[str(date)]),
                              "LW rad. trap": q_lw['in_from_els'][:, idcs_ground].sum(2).mean(1),
                              "SW rad. abs": q_sw['abs'][:, idcs_ground].sum(2).mean(1),
                              "SW rad. trap": q_sw['in_from_els'][:, idcs_ground].sum(2).mean(1)},
                             "lims": dashboard_irr_lims},
                        "Wind speed":
                            {"data":
                             {"Wind speed": knmi_data["FH"].loc[str(date)]},
                             "lims": (knmi_data["FH"][month_str].min(),
                                      knmi_data["FH"][month_str].max())}
                    }
    
                    if any([len(pl)
                            for pl in [CONFIG['PLOT_SENSOR_DATA'], CONFIG['PLOT_DATA'], CONFIG['PLOT_SHADE']]]):
                        width_dashboard = 2
                        width_plot = 4
                        width_total = width_dashboard + width_plot
    
                        height = 4
                        gs = gridspec.GridSpec(height, 6)
                        gs.update(wspace=1, hspace=0.2)
                        fig = plt.figure(figsize=(2 * width_total, 2 * height))
                        fig.autofmt_xdate()
                        if shape_dashboard == (2, 2):
                            ax1 = plt.subplot(gs[:, 0:-2])
                            ax2 = plt.subplot(gs[0:2, -2:-1])
                            ax3 = plt.subplot(gs[2:4, -2:-1])
                            ax4 = plt.subplot(gs[0:2, -1:])
                            ax5 = plt.subplot(gs[2:4, -1:])
    
                        elif shape_dashboard == (4, 1):
                            ax1 = plt.subplot(gs[:, 0:-2])
                            ax2 = plt.subplot(gs[0, -2:])
                            ax3 = plt.subplot(gs[1, -2:])
                            ax4 = plt.subplot(gs[2, -2:])
                            ax5 = plt.subplot(gs[3, -2:])
                            
                        else:
                            raise ValueError(f'Invalid shape_dashboard {shape_dashboard}, has to be (2,2) or (4,1)')
    
                        axs = [ax2, ax3, ax4, ax5]
                        data_title = None
                    else:
                        axs = None
                        data_title = str(date)
    
                    plot2d.dashboard(
                        data_dashboard,
                        t / STEPS_PER_DAY,
                        shape=shape_dashboard,
                        axs=axs,
                        data_title=data_title,
                        hide_spines=[
                            'right',
                            'top'])
                else:
                    ax1 = None
    
                # %%%% Plot data environmental elements
                for plot_type in CONFIG['PLOT_DATA']:
    
                    fn_dataplot = f'{FOLDER_RES}/{data_input}/rotated{ROT}/{plot_type}/{time_str}_{"&".join(labels)}'
    
                    start = time.time()
    
                    data = hf.get_data(locals_dict=locals(), var_str=data_input)
    
                    # data = locals()[data_input].copy()
                    
                    if data_input.startswith('T'):
                        data -= 273
    
                    if len(data.shape) == 3:
                        data = data.sum(-1)
    
                    if data_colorbar_lims:
                        colorbar_lims = data_colorbar_lims
                    if not data_colorbar_lims:
                        colorbar_lims = (data[data != 0].min(), data.max())
    
                    # Evaluate if the data to be plotted is time-dependent.
                    if data.shape[0] == STEPS_PER_DAY:
                        stop_plot = False
                        data = data[t, :]
                    else:
                        stop_plot = True
    
                    if isinstance(data, sparse._coo.core.COO):
                        data = data.todense()
    
                    data2d = data[envir['faces_idcs'][3]].reshape(SHAPE_2D_LR)
    
                    figtitle = f'{data_names.get(data_input, data_input)}-plot \n {GRIDSIZE} x {GRIDSIZE} m \n around {address_str.replace("_", " ")}'
                    if not stop_plot:
                        if SIMULATE_AVERAGES:
                            if time_now.day in [4,5]:
                                figtitle += f'\n coldest {time_now.strftime(format="%B %H:%M")}'
                            if time_now.day in [14,15]:
                                figtitle += f'\n average {time_now.strftime(format="%B %H:%M")}'
                            if time_now.day in [24,25]:
                                figtitle += f'\n hottest {time_now.strftime(format="%B %H:%M")}'
                        else:
                            figtitle += f'\n {time_now.strftime(format="%d %B, %Y %H:%M")}'
                        try:
                            print_uhi = False
                            if print_uhi:
                                figtitle += f'\n UHI = {uhi_str}'
                        except NameError:
                            pass
    
                    data_img, colorbar = plot2d.data_over_img(
                        img=map_img_gradient_lr,
                        data=data2d,
                        data_mask=~(
                            # masks_lr['buildings'] | 
                            masks_lr['trees']),
                        colorbar_lims=colorbar_lims,
                        data_alpha=0.7,
                        cmap=data_cmap)
    
                    if plot_type.endswith('shaded') and (
                            plot_type.startswith('2d') or plot_type.startswith('bar')):
                        data_img *= shade2d[..., np.newaxis]
    
                    # 2D plot
                    if plot_type.startswith('2d'):
                        plot2d.show_img(
                            data_img,
                            plot=True,
                            colorbar=colorbar,
                            save_path=fn_dataplot,
                            figtitle=figtitle,
                            show_axis=False,
                            # ax=ax1 if len(ax1.lines) == 0 else None
                            ax=ax1
                            )
                        # plt.show(block=False)
    
                    # 2.5D barplot
                    if plot_type.startswith('bar'):
    
                        start = time.time()
                        bar.plot(
                            height2d=elev_round_lr,
                            img=data_img,
                            gridsize=GRIDSIZE,
                            res=CONFIG['RES'],
                            # shade=(shade2d if apply_shade else None),
                            # shadefaces=shadefaces,
                            figtitle=figtitle,
                            plot=True,
                            saveimg=True,
                            colorbar=colorbar,
                            save_path=fn_dataplot,
                            save_ext='.png',
                            # view_zen=30,
                            # view_azi=-ROT,
                            view_zen=10,
                            view_azi=0.01,
                            ax=ax1
                        )
                    # 3D voxelsplot
                    
                        
                    if plot_type.startswith('voxels'):
                        if plot_type.endswith('shade'):
                            shade_2d = (q_sw['in'][t].max(1) / 1400) * 0.4 + 0.6
                        else:
                            shade_2d = None
                            
                        plot_data = {
                            data_input: {
                                'data': data,
                                'color': data_cmap,
                                'colorbar_lims': data_colorbar_lims}}
    
                        for r in [0 if ADDRESS != 'custom' else 20]:
                            plot3d.voxels_plot(
                                height3d=height_mask3d,
                                plot_data=plot_data,
                                cl=envir['coo'],
                                shade_2d=shade_2d,
                                t=t,
                                rot_north=ROT,
                                sun_zen_rad=sun_pos['zen_rad'][date],
                                sun_azi_rad=sun_pos['azi_rad'][date],
                                el_size=CONFIG['RES'],
                                view_dir=[30, -ROT + r],  # zen, azi
                                # view_dir=[45, 180-sun_pos['azi_deg'][date][t]],  # zen, azi
                                figtitle=figtitle,
                                grid_visibility=0.01,    
                                overwrite_existing=True,
                                disp_sun=sw_in_dir,
                                show=True,
                                saveimg=True,
                                dpi=600,
                                save_path=fn_dataplot,
                                save_ext='.png',
                                face_visibility=.99,
                                # ax=ax1,
                            )
    
                # %%%% Plot shadow environmental elements
                for plot_type in CONFIG['PLOT_SHADE']:
                    # if zen_deg_t >= 90:
                    #     continue
                    # print('\n\n Plotting shade on terrain\n\n')
    
                    start = time.time()
                    fn = f'{FOLDER_RES}/shade/rotated{ROT}/{plot_type}/{time_str}_{"&".join(labels)}'
    
                    figtitle = f'Shadowplot of {GRIDSIZE} x {GRIDSIZE} m \n \
                        "{address_str.replace("_", " ")}" \n \
                            {time_str}'
                    apply_shade = True
    
                    # 2D Plot
                    shade_img = map_img_gradient_lr.copy()
                    if plot_type.endswith('shaded'):
                        shade_img *= shade2d[..., np.newaxis]
    
                    if plot_type.startswith('2d'):
                        print('really plot')
                        plot2d.show_img(
                            shade_img,
                            plot=True,
                            colorbar=None,
                            contours=None,
                            save_path=fn,
                            figtitle=figtitle,
                            show_axis=False,
                            ax=None)
                        plt.show(block=False)
    
    
                    # 2.5D Barplot
                    if plot_type.startswith('bar'):
    
                        bar.plot(
                            height2d=elev_round_lr,
                            img=shade_img,
                            gridsize=GRIDSIZE,
                            res=CONFIG['RES'],
                            shadefaces=shadefaces,
                            colorbar=False,
                            save_path=fn,
                            plot=False,
                            saveimg=True,
                            figtitle=figtitle,
                            view_zen=10,
                            view_azi=0.01,
                            cam_scale=1)
    
                    # 3D Voxelsplot
                    if plot_type.startswith('voxel'):
                        if plot_type.endswith('shaded'):
                            shade_2d=(q_sw['in'][t].max(1) / 1400) * 0.4 + 0.6
                        else:
                            shade_2d=None
                        plot3d.voxels_plot(
                            height3d=height_mask3d,
                            plot_data=terrain_lr,
                            cl=envir['coo'],
                            shade_2d=shade_2d,
                            t=t,
                            rot_north=ROT,
                            sun_zen_rad=sun_pos['zen_rad'][date],
                            sun_azi_rad=sun_pos['azi_rad'][date],
                            el_size=CONFIG['RES'],
                            view_dir=[45, -ROT + r],  # zen, azi
                            figtitle=figtitle,
                            grid_visibility=0.01,
                            disp_sun=sw_in_dir,
                            # show=True,
                            show=False,
                            saveimg=True,
                            save_path=fn,
                            save_ext='.png',
                            face_visibility=.99,
                            # ax=ax1,
                        )
                        
    
                # %%%% Plot sensor elements data
                for i, plot_type in enumerate(CONFIG['PLOT_SENSOR_DATA']):
    
                    for j, sensor_input in enumerate(sensor_inputs):
                        try:
                            data = hf.get_data(locals(), var_str=sensor_input)
                        except KeyError:
                            if sensor_input == 'UHI':
                                data = utci_t - utci_rural.loc[time_str]
    
                        if sensor_input == 'T_mr':
                            data -= 273
    
                        if isinstance(data, sparse.COO):
                            data = data.todense()
                            
                        if len(data.shape) == 3:
                            data = data.sum(-1)
        
                        # Evaluate if the data to be plotted is time-dependent.
                        if data.shape[0] == STEPS_PER_DAY:
                            data = data[t, ...]
                            
                        data = data[sensors['idcs_dict']['street']]
    
                        if np.all(np.isnan(data)):
                            continue
    
                        sensor_data_coo = np.column_stack(
                            (coo_street[:, :2], data))
                        data_interp = hf.coo_to_ndarray(
                            sensor_data_coo,
                            plot=False,
                            interpolate_missing=False,
                            shape=SHAPE_2D_LR,
                            cmap=plt.get_cmap('hot'))
    
                        if sensor_custom_colorbar:
                            colorbar_lims = sensor_colorbar_lims[i]
                        if not sensor_custom_colorbar:
                            colorbar_lims = (data[data != 0].min(), data.max())
    
                        fn = f'{FOLDER_RES}/{sensor_input}/rotated{ROT}/{plot_type}/{time_str}_{"&".join(labels)}'
                        uhi_str = 'UNDEFINED'
                        try:
                            figtitle = f'{data_names[sensor_input]}-plot \n {GRIDSIZE} x {GRIDSIZE} m, center: {address_str.replace("_", " ")} \n {time_str} , UHI = {uhi_str}'
                        except KeyError:
                            figtitle = f'{sensor_input}-plot \n {GRIDSIZE} x {GRIDSIZE} m, center: {address_str.replace("_", " ")} \n {time_str} , UHI = {uhi_str}'
    
                        if len(CONFIG['PLOT_SENSOR_DATA']) > 0:
                            sensor_img, colorbar = plot2d.data_over_img(
                                img=map_img_gradient_lr,
                                data=data_interp,
                                data_mask=~(masks_lr['buildings']
                                            | masks_lr['trees']),
                                colorbar_lims=colorbar_lims,
                                # data_alpha=0.2,
                                data_alpha=1.0,
                                cmap=sensor_cmaps[i])
    
                        if plot_type.endswith('shaded'):
                            sensor_img *= shade2d[..., np.newaxis]
                            apply_shade = False
                        else:
                            apply_shade = True
    
                        # 2D plot
                        if plot_type.startswith('2d'):
    
                            plot2d.show_img(
                                sensor_img,
                                plot=True,
                                colorbar=colorbar,
                                # contours=contours_lr,
                                save_path=fn,
                                figtitle=figtitle,
                                show_axis=False,
                                # ax=ax1 if (
                                #     i == 0 and len(
                                #         ax1.lines) == 0) else None, #!!! werkte ff niet 1 mar '23
                                ax=None,
                            )
                            plt.show(block=False)
    
                        # 2.5D barplot 
                        if plot_type.startswith('bar'):
                            bar.plot(
                                height2d=elev_round_lr,
                                img=sensor_img,
                                gridsize=GRIDSIZE,
                                res=CONFIG['RES'],
                                colorbar=colorbar,
                                # contours=contours_lr,
                                shadefaces=shadefaces,
                                figtitle=figtitle,
                                # plot=True,
                                plot=False,
                                saveimg=True,
                                save_path=fn,
                                save_ext='.png',
                                view_zen=10,
                                view_azi=0.01,
                                ax=ax1
                            )
            # %%% possible break

            if to_next_day:
                print('break')
                break
            
            pass


            # %%% Save steady state results
            
            # print(f'sw sum differs {diff_sw} % with previous sw sum')
        if CONFIG['SAVE_STEADY_STATE']:
            
            wind_speed_colname = CONFIG['WIND_SPEED']
            
            if CONFIG['SOLAR_ANGLES']['location']:
                log_idcs = (sun_pos[date].index, sun_pos[date]['azi_deg TRUE'], sun_pos[date]['zen_deg'])
            else:
                log_idcs = (sun_pos[date]['azi_deg TRUE'], sun_pos[date]['zen_deg'])
                
            log_idx = tuple(zip(*log_idcs))
            
            emm_sky = CONFIG['EMISSIVITY_SKY']
            
            skin_thickness_mean = materials['skin_thickness'].mean().round(3)
            # density_mean = materials['density'].mean().round(1)
            vol_heat_cap_mean = materials['vol_heat_cap'].mean().round(1)
            
            skin_thickness_mean = ', '.join([f'{loc}={skin_thickness_dict[loc]}' for loc in ['roofs', 'walls', 'roads']])
            vol_heat_cap_mean = ', '.join([f'{loc}={vol_heat_cap_dict[loc]}' for loc in ['roofs', 'walls', 'roads']])
            
            for var_name, var_vals in forcings.items():
                try:
                    results_log.loc[log_idcs, ('external forcings', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, '', var_name, _res)] = var_vals.loc[date].values
                except:
                    try:
                        results_log.loc[log_idcs, ('external forcings', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, '', var_name, _res)] = var_vals.loc[date].values
                    except:
                        print('failed twice')
        # 'category', 'emissivity sky', 'hw', 'albedo', 'emissivity', 'skin thickness', 'density', 'spec. heat cap.', 'wind speed', 'location', 'variable', 'res'
            if CONFIG['ACTIVE_PHYSICS']['sw_out_refl']:
                results_log.loc[log_idcs, ('simulation steps', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, '', 'shortwave reflections', _res)] = sw_reflections_lst
            if CONFIG['ACTIVE_PHYSICS']['lw_out_refl']:
                results_log.loc[log_idcs, ('simulation steps', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, '', 'longwave reflections', _res)] = lw_reflections_lst
                
            results_log.loc[log_idcs, ('simulation steps', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, '', 'longwave reflections', _res)] = lw_reflections_lst
            
            for loc_name, loc_dict in results_to_log.items():
                for var in save_ave_data:
                    save_raw = True if var in save_raw_dict.get(loc_name, []) else False
                    
                    data_values = hf.get_data(locals(), var)
                    
                    if save_raw:
                        try:
                            results_log.loc[(slice(None), slice(None)), ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)]
                        except KeyError:
                            results_log.loc[(slice(None), slice(None)), ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = [None]*len(results_log.index)
                    
                    # if variable has values per face of elements
                    if len(data_values.shape) == 3:
                        results_log.loc[log_idcs, ('results sum', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = (data_values[:, loc_dict['idcs'], loc_dict['faces']].sum(1))
                        if save_raw:
                            for i in list(range(len(_zen))):
                                try:
                                    results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs'], loc_dict['faces']])
                                except:
                                    results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs'], loc_dict['faces']])
                    
                    # if variable has only 1 value per element (no faces)
                    elif len(data_values.shape) == 2:
                        results_log.loc[log_idcs, ('results sum', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = (data_values[:, loc_dict['idcs']].sum(1))
                        if save_raw:
                            for i in list(range(len(_zen))):
                                try:
                                    results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs']])
                                except:
                                    results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs']])
                        
            
            for loc_name, loc_dict in sensors_to_log.items():
                for var in save_ave_data_sensors:
                    save_raw = True if var in save_raw_dict.get(loc_name, []) else False
                    
                    data_values = hf.get_data(locals(), var)
                    
                    if len(data_values.shape) == 3:
                        if save_raw:
                            # raw data arrays top of sensor elements
                            try:
                                results_log.loc[(slice(None), slice(None)), ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)]
                            except KeyError:
                                results_log.loc[(slice(None), slice(None)), ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = [None]*len(results_log.index)
                            for i in list(range(len(_zen))):
                                try:
                                    results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs'], loc_dict['faces']])
                                except:
                                    try:
                                        results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs'], loc_dict['faces']])
                                    except:
                                        print('double fail')
                            
                        
                        # sum of variables, per face
                        try:
                            results_log.loc[log_idcs, ('results sum', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = (
                                data_values[:, loc_dict['idcs']][..., loc_dict['faces']].sum((1,2)))
                        except:
                            print('pause')   
                        
                        # sum of variables, per face
                        try:
                            results_log.loc[log_idcs, ('results sum', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = (
                                data_values[:, loc_dict['idcs']][..., loc_dict['faces']].sum((1,2)))
                        except:
                            print('pause')    
    
                    else:
                        if save_raw:
                            try:
                                results_log.loc[(slice(None), slice(None)), ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)]
                            except KeyError:
                                results_log.loc[(slice(None), slice(None)), ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = [None]*len(results_log.index)
                            for i in list(range(len(_zen))):
                                try:
                                    results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs']])
                                except:
                                    try:
                                        results_log.at[log_idx[i], ('results raw', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = list(data_values[i, loc_dict['idcs']])
                                    except:
                                        print('double fail')
                        
                        results_log.loc[log_idcs, ('results sum', emm_sky, hw, _alb, _emm, skin_thickness_mean, vol_heat_cap_mean, wind_speed_colname, loc_name, var, _res)] = (
                            data_values[:, loc_dict['idcs']]
                            .sum(1))
                    
            if CONFIG['SOLAR_ANGLES']['location']: #!!!
                results_log.to_pickle(FOLDER / 'results_log.pkl')
            else:
                results_log.to_pickle(FOLDER / 'results_log.pkl')
                


        # %%% Save timestep-data

        # ---- Save sunlit elements
        envir_sunlit_arr = empty_data_array.copy()
        sensors_sunlit_arr = empty_data_array.copy()

        for t, (s_els, s_sens) in enumerate(zip(envir_sunlit, sensors_sunlit)):
            envir_sunlit_arr[t, s_els] = True
            sensors_sunlit_arr[t, s_sens] = True

        np.savez_compressed(
            FOLDER_RES_LOG /
            ('sunlit_els/envir_' +
            date),
            sunlit_els=envir_sunlit_arr)

        np.savez_compressed(
            FOLDER_RES_LOG /
            ('sunlit_els/sensors_' +
            date),
            sunlit_els=sensors_sunlit_arr)

        
        # save data of elements averaged for sets of indices, per timestep
        if np.shape(q_sw['in_dir'])[0] >= 24 and data_log_bool: #!!! >= 24, why is that? 
            print('Start gathering data to log at elements...')
            start = time.time()
            
            for idcs in ave_idcs:
                for var in save_ave_data:
                    if '[' in var:
                        var_without_key = var.split('[')[0]
                        key = var.split('[')[-1].split(']')[0].strip("'").strip('"')
                        data_values = locals()[var_without_key][key]
                    else:
                        data_values = locals()[var]
                    if len(data_values.shape) == 3:
                        data_values = data_values.sum(2)
                    # data_log.at[str(date), (idcs, var)] = (
                    #     data_temp[:, ave_idcs[idcs]]
                    #     .mean(1).reshape(24, -1).mean(1))
                    data_log.at[str(date), (idcs, var)] = (
                        data_values[:, ave_idcs[idcs]]
                        .mean(1))
                  
            print(f'Gathering data for data_log took {hf.duration(start)}')
        
                    
        if data_log_bool:
            data_log.to_pickle(FOLDER_RES_LOG_ROT_CC_WS / 'data_log.pkl')
            
        # save data of elements averaged for sets of indices, per timestep
        if np.shape(q_sw_sensors['in_dir'])[0] >= 24 and data_log_bool: #!!! >= 24, why is that? 26/jan/23
            # for idcs in ave_idcs:
            for var in save_ave_data_sensors:
                if '[' in var:
                    var_without_key = var.split('[')[0]
                    key = var.split('[')[-1].split(']')[0].strip("'").strip('"')
                    data_values = locals()[var_without_key][key]
                else:
                    data_values = locals()[var]
                if len(data_values.shape) == 3:
                    data_log_sensors.at[str(date), ('top', var)] = (
                        data_values[:, :, 3]
                        .mean(1))
                    data_log_sensors.at[str(date), ('bottom', var)] = (
                        data_values[:, :, 2]
                        .mean(1))
                else:
                    data_log_sensors.at[str(date), ('top', var)] = (
                        data_values[:, :]
                        .mean(1))
                    data_log_sensors.at[str(date), ('bottom', var)] = (
                        data_values[:, :]
                        .mean(1))
                    
                    
        if data_log_bool:
            data_log_sensors.to_pickle(FOLDER_RES_LOG_ROT_CC_WS / 'data_log_sensors.pkl')

            # utci_log.index = pd.to_datetime(utci_log.index)
            # utci_log = utci_log.dropna().sort_index()
            # utci_log.to_csv(FOLDER_RES_LOG_ROT_CC_WS / 'utci_log.csv')
    
            # T_mr_log.index = pd.to_datetime(T_mr_log.index)
            # T_mr_log = T_mr_log.dropna().sort_index()
            # T_mr_log.to_csv(FOLDER_RES_LOG_ROT_CC_WS / 'T_mr_log.csv')

        # %%%  Plot monthly overview at the last day of the month

        date_slice = slice(str(year) + '-' + str(month))
        try:
            if plot_month:
                plot2d.overview_plot_multi(
                    data_log[str(year) + '-' + str(month)],
                    plot_vars,
                    plot_order=['q_sw', 'q_lw', 'T', ''],
                    plot_locs=["roads", "ground", "roofs"],
                    showfig=False,
                    savefig=True,
                    save_path=(FOLDER_RES_LOG_ROT_CC_WS
                               / (calendar.month_name[month] + '_lines')),
                    figtitle=calendar.month_name[month])

            if plot_month:
                plot2d.overview_bandwiths_multi(
                    data_log[str(year) + '-' + str(month)],
                    knmi_data[str(year) + '-' + str(month)],
                    plot_vars,
                    date_slice=date_slice,
                    plot_order=['q_sw', 'q_lw', 'T', '', 'knmi'],
                    plot_locs=["roads", "ground", "roofs"],
                    utci_log=utci_log[str(year) + '-' + str(month)],
                    savefig=True,
                    save_path=(FOLDER_RES_LOG_ROT_CC_WS
                               / calendar.month_name[month] + '_bandwiths'),
                    figtitle=ADDRESS + '\n ' + calendar.month_name[month])

        except Exception:
            pass

    # %% Make movie from plotted images

    if True:
        if len(CONFIG['PLOT_DATA']) > 0:
            hf.images_to_vid(
                data_input,
                load_path=fp_dataplot,
                save_path=FOLDER_RES,
                dates=list(dates_str),
                post_label='&'.join(labels),
                # vid_duration=len(dates_str) * 3,
                )
        if len(CONFIG['PLOT_SHADE']) > 0:
            hf.images_to_vid(
                'shade',
                load_path=fp_shadeplot,
                save_path=FOLDER_RES,
                dates=list(dates_str),
                post_label='&'.join(labels),
                vid_duration=len(dates_str) * 3,
                skip_ends=True)
        if len(CONFIG['PLOT_SENSOR_DATA']) > 0:
            for sensor_input in sensor_inputs:
                for plot_type in CONFIG['PLOT_SENSOR_DATA']:
                    hf.images_to_vid(
                        save_name=(
                            sensor_input + ' ' + plot_type),
                        load_path=f'{FOLDER_RES}/{sensor_input}/rotated{ROT}/{plot_type}/',
                        save_path=str(FOLDER_RES),
                        dates=list(dates_str),
                        post_label='&'.join(labels),
                        vid_duration=len(dates_str) * 3,
                        skip_ends=True)

    # %% Plot results 2D

    plot_2d = False
    if plot_2d:

        max_x = np.where(masks_lr['buildings'], elev_lr, 0).shape[0]

        if CONFIG['DAYS'] > 10 and True:
            STEPS_PER_DAY = int(24 * 60 * 60 / CONFIG['T_STEP'])

            x_data = sun_pos.index[:-1][::STEPS_PER_DAY]
            plt.plot(x_data, np.sum(sun_pos['Zenith'].to_numpy()[
                     :-1].reshape(CONFIG['DAYS'], -1) < 90, 1) * CONFIG['T_STEP'] / 60 / 60)
            plt.xlabel('Date')
            plt.ylabel('Sun up time [hours]')
            plt.show(block=False)

            plt.plot(x_data, sun_pos['Zenith'].to_numpy()
                     [:-1].reshape(CONFIG['DAYS'], -1).mean(1))
            plt.xlabel('Date')
            plt.ylabel('Average zenith')
            plt.show(block=False)

            data1 = np.mean(T, 1)
            plot2d.plot_data(
                data1,
                sun_pos.Datetime,
                albedo_set,
                hw_set,
                max_x,
                plot=True,
                save=False,
                xlabel='Time [hours]',
                ylabel1='T',
                ylabel2='',
                title='')

        x_data = sun_pos.index

        import matplotlib.dates as mdates

        hours = mdates.HourLocator()   # every year
        hours_fmt = mdates.DateFormatter('%m/%d %H:%M')

        fig, ax = plt.subplots()

        # TEMPERATURE

        fig, ax = plt.subplots()
        if len(envir['idcs_dict']["roads"]) > 0:
            ax.plot(x_data, np.mean(
                T[:, envir['idcs_dict']["roads"]], 1), label='roads')
        if len(envir['idcs_dict']["roofs"]) > 0:
            ax.plot(x_data, np.mean(
                T[:, envir['idcs_dict']["roofs"]], 1), label='roofs')
        if len(envir['idcs_dict']["walls"]) > 0:
            ax.plot(x_data, np.mean(
                T[:, envir['idcs_dict']["walls"]], 1), label='walls')
        if len(envir['idcs_dict']["grass"]) > 0:
            ax.plot(x_data, np.mean(
                T[:, envir['idcs_dict']["grass"]], 1), label='grass')
        if len(envir['idcs_dict']["trees"]) > 0:
            ax.plot(x_data, np.mean(
                T[:, envir['idcs_dict']["trees"]], 1), label='trees')
        if len(envir['idcs_dict']["water"]) > 0:
            ax.plot(x_data, np.mean(
                T[:, envir['idcs_dict']["water"]], 1), label='water')

        ax.plot(x_data, np.mean(T[:, :], 1), label='ave all')

        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hours_fmt)

        ax.get_xaxis().set_major_locator(mpl.ticker.LinearLocator(numticks=13))
        ax.set_xlim(sun_pos['Datetime'].min(), sun_pos['Datetime'].max())
        ax.set_ylim(285, 302)
        fig.autofmt_xdate()

        plt.legend()
        plt.show(block=False)

        plot_fluxes = True
        if plot_fluxes:
            plt.plot(q_sw['in_diff'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_sw_in_diff')
            plt.plot(q_sw['in_dir'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_sw_in_dir')
            plt.plot(q_sw['in_from_els'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_sw_in_from_els')
            plt.plot(q_sw['abs'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_sw_abs')
            plt.plot(q_lw['in_from_els'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_lw_in_from_els')
            plt.plot(q_lw['out_rad'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_lw_out_rad')
            plt.plot(q_lw['out_refl'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_lw_out_refl')
            plt.plot(q_lw['abs'][:, envir['idcs_dict']["roads"]].sum(
                2).mean(1), label='q_lw_abs')
            plt.plot(q_convection[:, envir['idcs_dict']["roads"]].mean(
                1), label='q_convection')
            plt.plot(T[:, envir['idcs_dict']["roads"]].mean(1), label='T')

            plt.legend()
            plt.show(block=False)

        # ground temperature plot
        fig, ax = plt.subplots()

        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hours_fmt)
        plt.xticks(rotation='vertical')
        ax.set_xlim(sun_pos['Datetime'].min(), sun_pos['Datetime'].max())
        ax.get_xaxis().set_major_locator(mpl.ticker.LinearLocator(numticks=13))
        plt.plot(x_data, T[:, np.concatenate(
            [envir['idcs_dict']["roads"], envir['idcs_dict']["grass"]])].mean(1))
        ax.set_ylim(250, 340)
        plt.savefig(FOLDER_RES / '' + 'T_ground ' +
                    dates_str[0] + ' to ' + dates_str[-1])
        plt.show(block=False)

# %% Print size of variables Mb, GB etc.

    for name, size in sorted(((name, sys.getsizeof(value))
                              for name, value in locals().items()), key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, hf.sizeof_fmt(size)))


# %% Run main
if __name__ == '__main__':
    print(sys.argv[2:])
    main(sys.argv[1], **dict(arg.split('=') for arg in sys.argv[2:]))
    

