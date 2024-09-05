#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Gijs Henstra, 2024
"""

import warnings
warnings.filterwarnings("ignore") # ignoring a deprecation warning that ocurred while running simulations for final presentation


run_or_debug = runfile
# run_or_debug = debugfile

res = 2.0
hws = [1.0] # set h/w values for custom idealised environments
albedos = [0.2]

start_dates = [[2019, 7, 22]] # hot unclouded day, simulation
days = 2 # days to simulate
start_dates_str = [",".join(str('['+",".join(str(elem) for elem in start_date)+']') for start_date in start_dates)] # make string from startdays


for scenario_folder in ['REAL - LCZ 5 - Demo - TU Delft - 150m']:

    for alb in albedos if 'custom' in scenario_folder.lower() else ['custom']:
        for hw in hws if 'custom' in scenario_folder.lower() else [1.0]:
                        
            # specify folder 
            args = f'"/Users/gijshenstra/Documents/Thesis_project/durhens/durhens/demos/{scenario_folder}"'
            
            
            # manually overwrite config file variables
            args += ' SKIP_STEPS=False'  # skip to 9 pm on first day
            args += f' HW={hw}' 
            args += f' RES={res}' 
            args += f' DAYS={days}' 
            args += f' START_DATES={start_dates_str}'
            args += f' MAX_RAY_DIST=200'
            
            # overwrite material properties
            if 'custom' in scenario_folder.lower():
                args += f' albedo_sw={alb}'
                pass
            
            # start running the simulation script
            run_or_debug(
                '/Users/gijshenstra/Documents/Thesis_project/durhens/durhens/__main__.py', 
                wdir='/Users/gijshenstra/Documents/Thesis_project/durhens/durhens',  
                args=args)
