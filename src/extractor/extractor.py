# Import modules
import pandas as pd
import numpy as np

import pprint
import xarray as xr
import os, sys, logging
from pyproj import Transformer
from datetime import datetime, timedelta

from .helpers import *

from lxml import html
import requests

logging.basicConfig(stream=sys.stderr, 
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')

# Global vars
start_date = datetime(1950, 1, 1) # This reference date comes from the NetCDF convention used for encoding the TIME variable in the CTD measurements

def getDDS(url_info):
    # Get dds info, and assign max dimensions to TIME and DEPTH
    
    pc_dim_dict = {}
    time_stop_dict = {}
    depth_stop_dict = {}

    # get all content of the server_url, and then filter it with year and available platforms
    page = requests.get(url_info[0])
    webpage = html.fromstring(page.content)
    
    urls_filtered = [p for p in webpage.xpath('//a/@href') if p.endswith(f'{year}.nc{url_info[1]}.dds')]

    for u in urls_filtered:

        dds = f'{url_info[0]}/{u}'#; print(dds)

        # Find platform code
        if len(url_info[1]) == 0: pc = dds.split('_')[0][-2:] # nmdc case
        else: pc = dds.split('_')[1][-2:] # t2_hyrax case

        pc_dim_dict[pc] = retrieveDDSinfo(dds)

        time_stop_dict[pc] = pc_dim_dict[pc]['TIME']
        depth_stop_dict[pc] = pc_dim_dict[pc]['DEPTH']

    assert depth_stop_dict.keys() == time_stop_dict.keys(), 'TIME and DEPTH Keys error. Please check.'
    
    return pc_dim_dict


def getPositionDict(pc_dim_dict, url_info):
    # Extract data and create position_dict
    
    position_dict = {}
    
    for pc in pc_dim_dict.keys():

        coords_str = getQueryString(pc_dim_dict[pc], keylist = ['TIME', 'LATITUDE', 'LONGITUDE']) 

        fix_lab = f'58{pc}_CTD_{year}' # label to use for this campaign

        url = f'{url_info[0]}{fix_lab}.nc{url_info[1]}?{coords_str}'; print(f'Platform: {pc}. URL with Queries:', url)

        remote_data, data_attr = fetch_data(url, year)

        position_dict[pc] = {'data': remote_data, 
                             'data_attr': data_attr}
    # print(position_dict)    
    
    return position_dict
    

def makePositionDF(position_dict):
    # Load locations (LONG & LAT) and TIME of all measurements in a position_df_raw (includes duplicates)

    position_df_raw = pd.DataFrame() 

    for key in position_dict.keys():
        test = pd.DataFrame()

        test['Longitude_WGS84'] = position_dict[key]['data']['LONGITUDE'].data.astype(float)
        test['Latitude_WGS84'] = position_dict[key]['data']['LATITUDE'].data.astype(float)
        test['Time'] = position_dict[key]['data']['TIME'].data.astype(float)
        test['Platform'] = key

        # Convert TIME from float to datetime
        test['Time'] = [start_date + timedelta(t) for t in test.loc[:,'Time']]
        length = len(test[test['Platform']==key])
        print(f'Platform {key}: {length} measurement locations.')
        
        position_df_raw = position_df_raw.append(test) 
    
    position_df_raw['Index_ABS'] = np.arange(0,len(position_df_raw))
    position_df_raw = position_df_raw.rename_axis("Index_Relative")

    # Now remove duplicates
    duplicates = position_df_raw[position_df_raw.duplicated(subset='Time') == True]
    position_df = position_df_raw.drop_duplicates(subset=['Time'])

    print(f'\nMerged dataframe with all platforms. Total of {len(position_df_raw)} measurement positions')
    print(f'Duplicates: \t{len(duplicates)} / {len(position_df_raw)} \nRemaining: \t{len(position_df)} / {len(position_df_raw)}')
    print(position_df)

    return(position_df)
    

def filterBBOXandTIME(position_df, bbox, time1, time2):
    # Filter the position_df dataframe by BBOX
    position_df_bbox = position_df[(position_df.loc[:,'Longitude_WGS84'] >= bbox[0]) & 
                                   (position_df.loc[:,'Longitude_WGS84'] <= bbox[1]) & 
                                   (position_df.loc[:,'Latitude_WGS84'] >= bbox[2]) & 
                                   (position_df.loc[:,'Latitude_WGS84'] <= bbox[3])]

    # Print filtering results on original dataframe
    sel_outof_all = f'{len(position_df_bbox)} out of {len(position_df)}.'
#     print(f'Selected positions (out of available positions): {sel_outof_all}')
#     print(position_df_bbox)

    # Filter the position_df_bbox dataframe by TIME
    time_start = datetime(int(time1[:4]), int(time1[4:6]), int(time1[6:]))
    time_end = datetime(int(time2[:4]), int(time2[4:6]), int(time2[6:]))

    position_df_bbox_timerange = position_df_bbox.loc[(position_df_bbox['Time']>=time_start) & 
                                                      (position_df_bbox['Time']<=time_end)]

    # Print filtering results on original dataframe
#     time_filter_str = f'{time_start.strftime("%Y%m%d")}-{time_end.strftime("%Y%m%d")}'
    print(f'User-defined Time Range: {time1}-{time2}')
    sel_outof_all = f'{len(position_df_bbox_timerange)} out of {len(position_df)}.'
    print(f'Selected positions (out of available positions): {sel_outof_all}')

    # print(position_df_bbox_timerange)
    
    return position_df_bbox_timerange
    

def getIndices(df_filtered):
    index_dict = {}
    
    for pc in df_filtered['Platform'].unique():
        index_dict[pc] = df_filtered[df_filtered['Platform']==pc].index.tolist()
    
    return index_dict


def extract(depth, bbox, time1, time2, mesh, var):

    #============= Set-up ==============
    assert time1[:4] == time2[:4], 'ERROR: different year, please check.'
    
    global year # Define year as global variable
    year = int(time1[:4])
    
    # Print input parameters
    logging.info(f'Input Parameters:\nDepth: {depth}\nBBOX: {bbox}\nTime Range: {time1}-{time2}\nMesh: {mesh}\nVars: {var.split(",")}')
    
    # Define URL     
    nmdc_url = 'http://opendap1.nodc.no/opendap/physics/point/yearly/' # URL of Norwegian Marine Data Centre (NMDC) data server 
    url_info = [nmdc_url, '']

    #============= Extraction of NetCDF data =============
    # Retrieval of DDS info
    pc_dim_dict = getDDS(url_info)
    print(pc_dim_dict)
    
    # Extract all platform_codes
    platform_codes = [pc for pc in pc_dim_dict.keys()]
    print(f'Available platforms in given year {year}: {platform_codes}')
    
    # Create position_dict
    position_dict = getPositionDict(pc_dim_dict, url_info)

    # Match and merge LAT, LONG and TIME of positions in a position_df dataframe
    position_df = makePositionDF(position_dict)
    
    # Filter by BBOX and Time
    df_filtered = filterBBOXandTIME(position_df, bbox, time1, time2)
    print(df_filtered)

    # Dictionary of indices
    index_dict = getIndices(df_filtered)
    print(index_dict)
    #============= Data Processing =============
    
    
    
    stop
    return 'EXTRACT complete.'



