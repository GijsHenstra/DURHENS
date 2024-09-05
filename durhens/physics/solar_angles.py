#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:01:11 2020

@author: gijshenstra
"""

# from pyephem_sunpath.sunpath import sunpos
import datetime
import numpy as np
import pandas as pd

import math
import ephem
import itertools


def diff_utc_hours(dates=datetime.datetime.now()):
    seconds_diff = (dates - datetime.datetime.utcnow()).seconds
    return round(seconds_diff / 3600)


def date_today():
    return list(
        map(int, str(datetime.datetime.today()).split(" ")[0].split("-")))


def sun_pos(lat, lon, timeutc):
    someplace = ephem.Observer()
    someplace.lon, someplace.lat = str(lon), str(lat)
    someplace.date = timeutc
    sun = ephem.Sun()
    sun.compute(someplace)
    return (90 - math.degrees(sun.alt), math.degrees(sun.az))

    # return sun

def sunrise(df):
    sunrise = (df["zen_deg"] < 90).groupby(df.index.date).nlargest(1, keep='first').reset_index(1)["Datetime"]
    sunrise.index = pd.to_datetime(sunrise.index)
    return sunrise

def sunset(df):
    sunset = (df["zen_deg"] < 90).groupby(df.index.date).nlargest(1, keep='last').reset_index(1)["Datetime"]
    sunset.index = pd.to_datetime(sunset.index)
    return sunset

def sunhighpoint(df):
    return (df["zen_deg"]).groupby(df.index.date).idxmin()


def initialise_sun_pos_df(t_step, freq, start_date, end_date, tz, time_shift):
    
    
    start_date = start_date + datetime.timedelta(seconds=time_shift)
    
    df = pd.DataFrame()
    df['Datetime'] = pd.date_range(
        start_date, end_date, freq=freq, closed='left', tz=tz)
    
    df = df.set_index(['Datetime'])
    
    df['timeutc'] = df.index.tz_convert('UTC').strftime('%Y/%m/%d %H:%M:%S')
    
    return df

def location_sun_pos(lat, lon, t_step,
                     start_date=datetime.datetime.today(),
                     end_date=datetime.datetime.today(),
                     time_shift=0,
                     tz='Europe/Amsterdam',
                     interp_missing=True,
                     method_interpolate='linear'):
    freq = str(int(t_step)) + 's'
    
    end_date_midnight = end_date + datetime.timedelta(hours=23, minutes=59)
    
    df = initialise_sun_pos_df(t_step=t_step, freq=freq, start_date=start_date, end_date=end_date_midnight, tz=tz, time_shift=time_shift)

    df['zen_deg'] = df.apply(
        lambda row: sun_pos(lat, lon, timeutc=row.timeutc)[0], axis=1)
    df['azi_deg'] = df.apply(
        lambda row: sun_pos(lat, lon, timeutc=row.timeutc)[1], axis=1)

    del df['timeutc']
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')]

    df = df.asfreq(freq=freq).interpolate(method=method_interpolate)
    
    print(f'Created dataframe with solar angles for location {lat}, {lon}')
    return df

def custom_sun_pos(t_step,
                   start_date=datetime.datetime.today(),
                   end_date=datetime.datetime.today(),
                   # days=1,
                   time_shift=0,
                   tz='Europe/Amsterdam',
                   zenith=[0,10,20,30,40,50,60,70,80,90],
                   azimuth=[90],
                   interp_missing=True,
                   method_interpolate='linear'):
    freq = str(int(t_step)) + 's'

    end_date_midnight = end_date + datetime.timedelta(hours=23, minutes=59)

    df = initialise_sun_pos_df(t_step=t_step, freq=freq, start_date=start_date, end_date=end_date_midnight, tz=tz, time_shift=time_shift)
    df['zen_deg'] = 0
    df['azi_deg'] = 0

    # if (len(zenith) == 1 and len(azimuth) > 1):
    #     zenith = zenith * len(azimuth)
    # elif (len(zenith) == 1 and len(azimuth) == 1):
    #     pass
    # elif (len(zenith) > 1 and len(azimuth) == 1):
    #     azimuth = azimuth * len(zenith)
    
    if len(zenith) == len(azimuth):
        pass
    else:
        multiplied = len(zenith) * len(azimuth)
        zenith = zenith * int(multiplied / len(zenith))
        azimuth = [a for a in azimuth for i in range(int(multiplied / len(azimuth)))]
        # zenith = [zenith[i] for i in range(int(multiplied / len(zenith)))]
        
    for i, date in enumerate(np.unique(df.index.date)):
        try:
            # if len(zenith[i]) == 1:
            #     df.loc[(df.index.date == date), 'zen_deg'] = zenith[i]
            # else: 
            steps_per_day = len(df.loc[(df.index.date == date)])
                
            df.loc[(df.index.date == date), 'zen_deg'] = list(itertools.chain(*[[zen_i]* int(steps_per_day / len(zenith[i])) for zen_i in zenith[i]]))
            
            # if len(azimuth[i]) == 1:
            #     df.loc[(df.index.date == date), 'azi_deg'] = azimuth[i]
            # else:
            df.loc[(df.index.date == date), 'azi_deg'] = list(itertools.chain(*[[azi_i]* int(steps_per_day / len(azimuth[i])) for azi_i in azimuth[i]]))
        except IndexError:
            break
        
    # df['zen_deg'] = df['Zenith'].astype(float)
    # df['azi_deg'] = df['Azimuth'].astype(float)
        
    del df['timeutc']
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')]

    df = df.asfreq(freq=freq).interpolate(method=method_interpolate)

    print(f'Created dataframe with solar angles for \n zenith {zenith} \n azimuth{azimuth}')

    return df
    
    
