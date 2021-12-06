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
#     print(position_dict)    
    
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
    global sel_outof_all
    sel_outof_all = f'{len(position_df_bbox)} out of {len(position_df)}.'
#     print(f'Selected positions (out of available positions): {sel_outof_all}')
#     print(position_df_bbox)

    # Filter the position_df_bbox dataframe by TIME
    time_start = datetime(int(time1[:4]), int(time1[4:6]), int(time1[6:]))
    time_end = datetime(int(time2[:4]), int(time2[4:6]), int(time2[6:]))

    position_df_bbox_timerange = position_df_bbox.loc[(position_df_bbox['Time']>=time_start) & 
                                                      (position_df_bbox['Time']<=time_end)]

    # Print filtering results on original dataframe
    print(f'User-defined Time Range: {time_filter_str}')
    sel_outof_all = f'{len(position_df_bbox_timerange)} out of {len(position_df)}.'
    print(f'Selected positions (out of available positions): {sel_outof_all}')

    # print(position_df_bbox_timerange)
    
    return position_df_bbox_timerange
    

def getIndices(df_filtered):
    index_dict = {}
    
    for pc in df_filtered['Platform'].unique():
        index_dict[pc] = df_filtered[df_filtered['Platform']==pc].index.tolist()
    
    return index_dict


def extractVARsAndDepth(pc_sel, position_dict, pc_dim_dict, vars_sel, url_info):
    data_dict = {}
    metadata = {}

    for pc in pc_sel:

        metadata[pc] = {}

        v_min = int(float(position_dict[pc]['data_attr'][6]))
        metadata[pc]['vmin'] = v_min
        metadata[pc]['depth_abs_v1'] = 0 # this is fixed
        metadata[pc]['depth_abs_v2'] = pc_dim_dict[pc]['DEPTH'] # this is fixed

        # ==============================================================================
        """
        Define here the DEPTH range of your selection, in meters. Note that either:
        - 'depth_m_v1' is equal to the lower bound (ie index=0), or 
        - 'depth_m_v2' is equal to the upper bound (ie index=-1)
        """
        metadata[pc]['depth_m_v1'] = 0 
        metadata[pc]['depth_m_v2'] = depth_g
        # ==============================================================================

        # assert metadata[pc]['depth_m_v1'] < metadata[pc]['depth_m_v2'], 'ERROR: the lower bound must be lower than the higher bound' 
        # assert metadata[pc]['depth_m_v1'] == 0 or metadata[pc]['depth_m_v2'] == pc_dim_dict[pc]['DEPTH'], 'ERROR: one of the two values must be equal to one of the lower/upper bounds'

        #     print(f'DEPTH range of interest (meters): {metadata[pc]["depth_m_v1"]} - {metadata[pc]["depth_m_v2"]}')

        # the start and stop values are adjusted based on the vmin value
        if metadata[pc]['vmin'] == 1: 
            if metadata[pc]['depth_m_v1'] == 0: # 
                metadata[pc]['depth_newindex_v1'] = metadata[pc]['depth_m_v1'] # the same
                metadata[pc]['depth_newindex_v2'] = metadata[pc]['depth_m_v2'] # the same, so I have the right size. When I shift and add the nan, I get rid of further element on the right
                metadata[pc]['depth_newindex4xr_v2'] = metadata[pc]['depth_m_v2']# - 1

            elif metadata[pc]['depth_m_v1'] != 0: 
                metadata[pc]['depth_newindex_v1'] = metadata[pc]['depth_m_v1'] - 1 # start one element before
                metadata[pc]['depth_newindex_v2'] = metadata[pc]['depth_m_v2'] - 1 # last element is excluded, ie stop one element before. But then I'll have to remoove one element
                metadata[pc]['depth_newindex4xr_v2'] = metadata[pc]['depth_m_v2'] - metadata[pc]['depth_m_v1'] - 1 

        else:
            metadata[pc]['depth_newindex_v1'] = metadata[pc]['depth_m_v1']
            metadata[pc]['depth_newindex_v2'] = metadata[pc]['depth_m_v2']

            if metadata[pc]['depth_m_v1'] == 0: # 
                metadata[pc]['depth_newindex4xr_v2'] = metadata[pc]['depth_m_v2']

            elif metadata[pc]['depth_m_v1'] != 0: 
                metadata[pc]['depth_newindex4xr_v2'] = metadata[pc]['depth_m_v2'] - metadata[pc]['depth_m_v1']

        metadata[pc]['depth_newindex4xr_v1'] = 0

        pprint.pprint(metadata[pc])
        print(f'{pc} DEPTH range of interest (adjusted with vmin): {metadata[pc]["depth_newindex_v1"]} - {metadata[pc]["depth_newindex_v2"]}')

        fix_lab = f'58{pc}_CTD_{year}' # platform_codes and year are defined at the beginning of the notebook 

        # Get coordinates (needed for keeping the correct structure, and for plotting) 
        coords_str = getQueryString(pc_dim_dict[pc], keylist = ['TIME', 'LATITUDE', 'LONGITUDE']) # list the coordinates you want

        # Extract TIME and DEPTH dimension for queries 
        time_dims = getQuery(pc, start=0, stop=pc_dim_dict[pc]['TIME'])
        depth_dims = getQuery(pc, start=metadata[pc]['depth_newindex_v1'], stop=metadata[pc]['depth_newindex_v2'])#; print(depth_dims)

        # join TIME and DEPTH for Variables
        var_str_ALL = []
        for v in vars_sel: var_str_ALL = np.append(var_str_ALL, f'{v}{time_dims}{depth_dims}')
        queries_vars = ','.join(var_str_ALL)

        # Build url and url with queries (url_q)
        url = f'{url_info[0]}{fix_lab}.nc{url_info[1]}?{coords_str}' 
        url_q = f'{url},{queries_vars}'; print(f'Platform {pc} URL:', url_q)

        remote_data, data_attr = fetch_data(url_q, year)

        data_dict[pc] = {'data': remote_data, 
                         'data_attr': data_attr}

        print(f'{data_attr}\n')

    print(f'Checking the existing campaigns in the dictionary: {list(data_dict.keys())}')
    
    return data_dict, metadata


def getVminDict(overview_df, vars_sel):
    vmin_dict = {}

    # select only those platforms where vmin == 1
    vmin_pc = overview_df[overview_df['Vertical_min'] == 1.0].index

    for i in vmin_pc:
        vmin_dict[i] = {}

        for v in vars_sel:
            vmin_dict[i][v] = False
    
    return vmin_dict


def filterbyDepthAndIndices(data_dict, metadata, vmin_dict, vars_sel, df_filtered):
    filtered_xarr_dict = {}

    print(f'Selected DEPTH: {depth_g}m')
    for pc in data_dict.keys():

        # Generate a filtered xarray with all variables for selected Platform, for a certain DEPTH range
        if metadata[pc]['depth_m_v1']==0: align_and_nan = True
        else: align_and_nan = False

        for v in vars_sel: 
            check_alignment(data_dict, pc, v, align_and_nan, vmin_dict)

        filtered_xarr_dict[pc] = filter_xarr_DEPTH(df_filtered, 
                                                   data_dict,
                                                   platform=pc,
                                                   depth_range=[depth_g, depth_g])
        # display(filtered_xarr_dict[pc])

    return filtered_xarr_dict


def aggregatePlatformsAndMerge(data_dict, filtered_xarr_dict, vars_sel):
    # Dictionary of variables for each platform
    data_var_dict = {}
    depth_arr = []

    for pc in filtered_xarr_dict.keys():

        data_var_dict[pc] = {}
        data = filtered_xarr_dict[pc]

        depth_dim_pc = data.dims["DEPTH"]
        depth_arr.append(depth_dim_pc)

        print(f'PC {pc}\tFiltered Dims: TIME={data.dims["TIME"]}, DEPTH={data.dims["DEPTH"]}')

        for var in vars_sel:
            data_var_dict[pc][var] = filtered_xarr_dict[pc][var]

    assert all(x==depth_arr[0] for x in depth_arr), 'ERROR, the DEPTH dimensions must be equal.'
        
    # Now combine arrays across platforms, for each variable
    merged_arr = {}

    for var in vars_sel:

        merged_arr[var] = xr.merge([data_var_dict[pc][var] for pc in data_dict.keys()])  

        title = f'Var={var} (Merged Platforms)\nFilter: Time Range={time_filter_str};\nBBOX={bbox_name}; Depth Range={depth_g}m;\nSel/All={sel_outof_all}'

        plotVar_MergedPlatforms(merged_arr[var], var, title=title)
        # display(merged_arr[var])
        
    return data_var_dict, merged_arr


#=======================================================================================  
def extract(depth, bbox, time1, time2, mesh, vars_sel):
    #============= Set-up ==============
    assert time1[:4] == time2[:4], 'ERROR: different year, please check.'
    global time_filter_str
    time_filter_str = f'{time1}-{time2}'
    global year # Define year as global variable
    year = int(time1[:4])
    
    global bbox_name
    bbox_name = bbox

    global depth_g
    depth_g = depth
    
    # Print input parameters
    logging.info(f'Input Parameters:\nDepth: {depth_g}\nBBOX: {bbox}\nTime Range: {time1}-{time2}\nMesh: {mesh}\nVars: {vars_sel}')
    
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
    
    print('\n================================\n================================\n')
    #============= Data Processing =============
    pc_sel = df_filtered['Platform'].unique()
    print(pc_sel)
    data_dict, metadata = extractVARsAndDepth(pc_sel, position_dict, pc_dim_dict, vars_sel, url_info) # perhaps I can remove pc_dim_dict and related line
    print(data_dict)
    
    # Create overview dataframe
    overview_df = pd.DataFrame()
    overview_df = getAttributes(overview_df, data_dict)
    print(overview_df)
    
    # Generate vmin dictionary (needed to avoid doing the vmin adjustment more than once)
    vmin_dict = getVminDict(overview_df, vars_sel)
    print(vmin_dict)
    
    # Filter by Depth and Indices (generated by BBOX and Time indices)
    filtered_xarr_dict = filterbyDepthAndIndices(data_dict, metadata, vmin_dict, vars_sel, df_filtered)
    print(filtered_xarr_dict)
    
    # Aggregation of Available Platforms
    data_var_dict, merged_arr = aggregatePlatformsAndMerge(data_dict, filtered_xarr_dict, vars_sel)
    print(merged_arr)
    

    
    stop
    return 'EXTRACT complete.'



