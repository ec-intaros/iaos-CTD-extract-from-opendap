# Import modules
import pandas as pd
import numpy as np

import pprint
import xarray as xr
import os, sys, logging

from datetime import datetime, timedelta

from .helpers import *

from lxml import html
import requests

logging.basicConfig(stream=sys.stderr, 
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')

# Global var
global start_date
start_date = datetime(1950, 1, 1) # This reference date comes from the NetCDF convention used for encoding the TIME variable in the CTD measurements


def checkDates(time1,time2):
    time_start = datetime(int(time1[:4]), int(time1[4:6]), int(time1[6:]))
    time_end = datetime(int(time2[:4]), int(time2[4:6]), int(time2[6:]))
    assert time_start < time_end, 'ERROR: time1 cannot be after time2, please check.'
    
    year1 = int(time1[:4])
    year2 = int(time2[:4])
    time_str = f'{time1}-{time2}'
    
    years = list(np.arange(year1,year2+1))
    
    return time_start, time_end, year1, year2, time_str, years


def checkParams(depth, bbox, time1, time2, vars_sel, group, formats):
    # Define Global Variables
    global time_start, time_end, time_str, year1, year2, years, bbox_g, bbox_str, depth_g, vars_g, group_g, formats_g
     
    # Check dates
    time_start, time_end, year1, year2, time_str, years = checkDates(time1,time2)
    print('Time Range:', time_str)
    print('Years of data:', years)

    # Check BBOX
    bbox_g = [int(bb) for bb in bbox.split(',')]
    bbox_str = '.'.join([str(b) for b in bbox_g])
    print('Bounding Box:', bbox_g) #, bbox_str)

    # Check depth
    depth_g = depth
    print(f'Depth: {depth_g}m')

    # Check Vars
    vars_g = vars_sel
    print('Vars:', vars_g)

    # Check 'group' flag
    if len(vars_g) > 1:
        # the flag 'group' is required in this case
        group_g = group
        assert group_g is not None, 'The flag "--group" is required when multiple variables are selected.'
        print('Group files per Var:', group_g)

    else: group = False
    
    # Check Formats
    formats = [ff.lower() for ff in formats.split(',')]    
    formats_g = []
    if 'csv' in formats: formats_g.append('CSV')
    if 'netcdf4' in formats: formats_g.append('NetCDF4')
    print('Output files format(s):', formats_g)
    assert len(formats_g) > 0, 'ERROR: Output file format entered is wrong. It must be "csv", "netcdf4", or "csv,netcdf4".'


def getDDS(url_info, year):
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


def getPositionDict(pc_dim_dict, url_info, year):
    # Extract data and create position_dict
    
    position_dict = {}
    
    for pc in pc_dim_dict.keys():

        coords_str = getQueryString(pc_dim_dict[pc], keylist = ['TIME', 'LATITUDE', 'LONGITUDE']) 

        fix_lab = f'58{pc}_CTD_{year}' # label to use for this campaign

        url = f'{url_info[0]}{fix_lab}.nc{url_info[1]}?{coords_str}'; print(f'Platform: {pc}. URL with Queries:', url)

        remote_data, data_attr = fetch_data(url, year)

        position_dict[pc] = {'data': remote_data, 
                             'data_attr': data_attr}
    print(position_dict)    
    
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
    
    position_df_temp = position_df_raw.drop_duplicates(subset=['Time'])
    
    print(f'Merged dataframe with all platforms. Total of {len(position_df_raw)} measurement positions')
    print(f'Duplicates: \t{len(duplicates)} / {len(position_df_raw)} \nRemaining: \t{len(position_df_temp)} / {len(position_df_raw)}')

    return position_df_temp


def filterBBOXandTIME(position_df, time1, time2):
    # Filter the position_df dataframe by BBOX
    position_df_bbox = position_df[(position_df.loc[:,'Longitude_WGS84'] >= bbox_g[0]) & 
                                   (position_df.loc[:,'Longitude_WGS84'] <= bbox_g[1]) & 
                                   (position_df.loc[:,'Latitude_WGS84'] >= bbox_g[2]) & 
                                   (position_df.loc[:,'Latitude_WGS84'] <= bbox_g[3])]

    # Print filtering results on original dataframe
    global sel_outof_all
    sel_outof_all = f'{len(position_df_bbox)} out of {len(position_df)}.'
    
    # Filter the position_df_bbox dataframe by TIME
    
    position_df_bbox_timerange = position_df_bbox.loc[(position_df_bbox['Time']>=time_start) & 
                                                      (position_df_bbox['Time']<=time_end)]

    # Print filtering results on original dataframe
    print(f'\nUser-defined Time Range: {time_str}')
    sel_outof_all = f'{len(position_df_bbox_timerange)} out of {len(position_df)}.'
    print(f'Selected positions (out of available positions): {sel_outof_all}')

    # print(position_df_bbox_timerange)
    
    return position_df_bbox_timerange


def getIndices(df_filtered, year):
    
    index_dict[year] = {}
    
    # Select available platforms ONLY of the specific year
    df_year = df_filtered[pd.DatetimeIndex(df_filtered['Time']).year == year] # select only those rows of that year
    # print(df_year)    

    for pc in df_year['Platform'].unique():
        index_dict[year][pc] = df_year[df_year['Platform']==pc].index.tolist()
    
    return index_dict[year]


def extractVARsAndDepth(pc_sel, position_dict, pc_dim_dict, url_info, year):  
    data_dict = {}
    metadata = {}

    for pc in pc_sel:
        print(pc)
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
        for v in vars_g: var_str_ALL = np.append(var_str_ALL, f'{v}{time_dims}{depth_dims}')
        queries_vars = ','.join(var_str_ALL)

        # Build url and url with queries (url_q)
        url = f'{url_info[0]}{fix_lab}.nc{url_info[1]}?{coords_str}' 
        url_q = f'{url},{queries_vars}'; print(f'Platform {pc} URL:', url_q)

        remote_data, data_attr = fetch_data(url_q, year)

        data_dict[pc] = {'data': remote_data, 
                         'data_attr': data_attr}

        print(f'{data_attr}\n')
    
    assert list(pc_sel) == list(data_dict.keys()), 'ERROR: different platforms, please check.'
    
    return data_dict, metadata


def getVminDict(overview_df):
    vmin_dict = {}

    # select only those platforms where vmin == 1
    vmin_pc = overview_df[overview_df['Vertical_min'] == 1.0].index

    for i in vmin_pc:
        vmin_dict[i] = {}

        for v in vars_g:
            vmin_dict[i][v] = False
    
    return vmin_dict


def check_alignment(data_dict, pc, var, align_and_nan, vmin_dict):
    # Function to check whether data should be aligned if vmin = 1, and align if so if has not been done already

    xarr = data_dict[pc]['data']
    xarr_var = xarr[var].data
    
    vmin = float(xarr.attrs['geospatial_vertical_min'])

    if vmin == 0:
        print(f'Platform: {pc}; Vertical min: {vmin}; Var: {var}')
        
    elif vmin==1 and vmin_dict[pc][var]==False and align_and_nan: 
        # shift to the right and add nan in first position 
        print(f'Platform: {pc}; Vertical min: {vmin}; Var: {var} --> aligning and add nan')        
        data_dict[pc]['data'][var].data = adjust_with_vmin(xarr_var, value=np.nan)
        vmin_dict[pc][var] = True # to avoid doing hte vmin adjustment for this pc/var more than once  
        
    elif vmin==1 and vmin_dict[pc][var]==False and not align_and_nan: 
        # No need to shift, this occurred already in the data extraction
        print(f'Platform: {pc}; Vertical min: {vmin}; Var: {var} --> data has been aligned already')
        vmin_dict[pc][var] = True # to avoid doing hte vmin adjustment for this pc/var more than once


def filterbyDepthAndIndices(data_dict_yr, metadata_yr, vmin_dict_yr, df_filtered):
    
    print(f'Selected DEPTH: {depth_g}m')
    
    for year in data_dict_yr.keys():
        print(year)
        filtered_xarr_dict[year] = {}
        
        # Select available platforms ONLY of the specific year
        df_year = df_filtered[pd.DatetimeIndex(df_filtered['Time']).year == year] # select only those rows of that year
        pc_year = df_year['Platform'].unique()

        for pc in pc_year:
            print(pc)
            # Generate a filtered xarray with all variables for selected Platform, for a certain DEPTH range
            
            if metadata_yr[year][pc]['depth_m_v1']==0: align_and_nan = True
            else: align_and_nan = False

            for v in vars_g: 
                check_alignment(data_dict_yr[year], pc, v, align_and_nan, vmin_dict_yr[year]) 

            filtered_xarr_dict[year][pc] = filter_xarr_DEPTH(index_dict[year],
                                                             data_dict_yr[year],
                                                             platform=pc,
                                                             depth_range=[depth_g, depth_g])
            print(filtered_xarr_dict[year][pc])
        print('\n')
    
    return filtered_xarr_dict


def filter_xarr_DEPTH(index_yr, data_dict, platform, depth_range):
    # Function to filter XARRAY based on platform, Var and DEPTH

    # Filer data using the indexes of the elements filtered by year and platform 
    xarr_sel = data_dict[platform]['data'].isel(TIME=index_yr[platform],
                                                LATITUDE=index_yr[platform],
                                                LONGITUDE=index_yr[platform],
                                                DEPTH=slice(depth_range[0], depth_range[1]+1))
    return xarr_sel


def aggregatePlatforms(filtered_xarr_dict):
    # Dictionary of variables for each platform
    depth_arr = []

    for year in years:
        print(year)
        data_var_dict_yr[year] = {}

        for pc in filtered_xarr_dict[year].keys():

            data_var_dict_yr[year][pc] = {}
            data = filtered_xarr_dict[year][pc]

            depth_dim_pc = data.dims["DEPTH"]
            depth_arr.append(depth_dim_pc)

            print(f'PC {pc}\tFiltered Dims: TIME={data.dims["TIME"]}, DEPTH={data.dims["DEPTH"]}')

            for var in vars_g:
                data_var_dict_yr[year][pc][var] = filtered_xarr_dict[year][pc][var]

        assert all(x==depth_arr[0] for x in depth_arr), 'ERROR, the DEPTH dimensions must be equal.'
        #print(data_var_dict_yr[year])
        
    return data_var_dict_yr


def mergeArrays(data_var_dict_yr):
    # Combine arrays across platforms, for each variable
        
    for var in vars_g:
        merged_arr[var] = xr.merge([data_var_dict_yr[y][pc][var] for y in data_var_dict_yr.keys() for pc in data_var_dict_yr[y].keys()] )  
        
        # title = f'Var={var} (Merged Platforms)\nFilter: Time Range={time_str};\nBBOX={bbox_g}; Depth {depth}m;\nSel/All={sel_outof_all}'
        # plotVar_MergedPlatforms(merged_arr[var], var, title=title)
    
    return merged_arr


def mkdir():
    out_dir = os.path.join(os.getcwd(), 'exported_data')
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    return out_dir


def exportSeparate(out_dir, merged_arr):
    print('Creating separate files for each variable')

    for v in vars_g:

        fname = os.path.join(out_dir,f'export_BBOX={bbox_str}_timerange={time_str}_d={depth_g}m_var={v}')
        
        if 'NetCDF4' in formats_g:
            # Export each variable to NetCDF separately
            netcdfname = fname + '.nc.nc4'
            merged_arr[v].to_netcdf(path=netcdfname,mode='w')
            print(netcdfname, 'exported.')

        if 'CSV' in formats_g:
            # Export each variable to CSV separately
            csvname = fname + '.csv'
            m_temp = merged_arr[v].to_dataframe().reset_index()
            m_temp['DEPTH'] = depth_g
            m_temp[['TIME','DEPTH',v]].to_csv(csvname, sep=',', header=True)
            print(csvname, 'exported.')   
            
            
def exportGroup(out_dir, merged_arr_vars):
    print('Creating unique file with all variables')

    fname = os.path.join(out_dir,f'export_BBOX={bbox_str}_timerange={time_str}_d={depth_g}m_vars={"-".join(vars_g)}')

    if 'NetCDF4' in formats_g:
        # Export all variables to NetCDF
        netcdfname = fname + '.nc.nc4'
        merged_arr_vars.to_netcdf(path=netcdfname,mode='w')
        print(netcdfname, 'exported.')

    if 'CSV' in formats_g:
        # Export all variables to CSV
        csvname = fname + '.csv'
        m_temp = merged_arr_vars.to_dataframe().reset_index()
        m_temp['DEPTH'] = depth_g
        m_temp.to_csv(csvname, sep=',', header=True)
        print(csvname, 'exported.')
        
        


#=======================================================================================  
def extract(depth, bbox, time1, time2, vars_sel, group, formats):
    #============= Set-up ==============
    print('\n============\nSet-up\n============')
    # Check and print input parameters
    checkParams(depth, bbox, time1, time2, vars_sel, group, formats)
    
    # Define URL     
    nmdc_url = 'http://opendap1.nodc.no/opendap/physics/point/yearly/' # URL of Norwegian Marine Data Centre (NMDC) data server 
    url_info = [nmdc_url, '']; print('url:', nmdc_url)
    
    #============= Extraction of NetCDF data =============
    print('\n============\nData Extraction\n============')
    dds_year_dict = {}
    pos_year_dict = {}
    global position_df
    position_df = pd.DataFrame()
        
    for year in years: # need to do a for loop over the years as the data is saved in years on the server
        print('\nWorking on year:', year)
        
        # Retrieval of DDS info
        dds_year_dict[year] = getDDS(url_info, year) # dds_year_dict[year] replaced pc_dim_dict 
        pprint.pprint(dds_year_dict[year])
        
        # Extract all platform_codes for that year
        platform_codes = [pc for pc in dds_year_dict[year].keys()]
        print(f'Available platforms in given year {year}: {platform_codes}')

        # Create position_dict
        pos_year_dict[year] = getPositionDict(dds_year_dict[year], url_info, year) # pos_year_dict[year] replaced position_dict
    #     pprint.pprint(pos_year_dict[year])

        # Match and merge LAT, LONG and TIME of positions in a position_df dataframe
        print('Now makePositionDF')
        position_df_temp = makePositionDF(pos_year_dict[year])
        print(position_df_temp)
        
        position_df = position_df.append(position_df_temp)

    print('\nCOMBINED position_df'); print(position_df); print('')

    # Filter by BBOX and Time
    df_filtered = filterBBOXandTIME(position_df, time1, time2)
    print('\nFILTERED position_df'); print(df_filtered); print('')

    # Dictionary of indices
    global index_dict
    index_dict = {}
    for year in years: index_dict[year] = getIndices(df_filtered, year)
    print(index_dict)
    
    #============= Data Processing =============
    print('\n============\nData Processing\n============')
    
    global data_dict_yr, metadata_yr, vmin_dict_yr, filtered_xarr_dict
    data_dict_yr = {}
    metadata_yr = {}
    vmin_dict_yr = {}

    for year in years: # need to do a for loop over the years as the data is saved in years on the server
        
        # Extract all platform_codes for that year, given the BBOX and time constraints
        df_year = df_filtered[pd.DatetimeIndex(df_filtered['Time']).year == year] # select only those rows of that year
        pc_sel = df_year['Platform'].unique()
        print(f'Working on year: {year} - Available platforms: {pc_sel}')

        data_dict_yr[year], metadata_yr[year] = extractVARsAndDepth(pc_sel, pos_year_dict[year], dds_year_dict[year], url_info, year) 

        print(f'Attributes Year: {year}')
        # Create overview dataframe
        overview_df = pd.DataFrame()
        overview_df = getAttributes(overview_df, data_dict_yr[year])
        print('overview_df')
        print(overview_df)

        # Generate vmin dictionary (needed to avoid doing the vmin adjustment more than once)
        vmin_dict_yr[year] = getVminDict(overview_df)
    
    # Filter by Depth and Indices (generated by BBOX and Time indices) 
    filtered_xarr_dict = {}
    filtered_xarr_dict = filterbyDepthAndIndices(data_dict_yr, metadata_yr, vmin_dict_yr, df_filtered)
    print('filtered_xarr_dict', filtered_xarr_dict)    

    # Aggregation of Available Platforms in given Years
    print('\nAggregate')
    global data_var_dict_yr
    data_var_dict_yr = {}
    data_var_dict_yr = aggregatePlatforms(filtered_xarr_dict)
    print(data_var_dict_yr)

    # Merge Arrays
    print('\nMerge')
    global merged_arr
    merged_arr = {}
    merged_arr = mergeArrays(data_var_dict_yr)
    print(merged_arr['PSAL'])
    
    #============= Export to File =============
    print('\n============\nExport to File\n============')
    out_dir = mkdir() # make output directory

    if group: 
        # Combine arrays across platforms, with ALL variables
        merged_arr_vars = xr.merge([data_var_dict_yr[y][pc][var] for y in data_var_dict_yr.keys() for pc in data_var_dict_yr[y].keys() for var in vars_g])  
    #     merged_arr_vars = merged_arr_vars.fillna(0) # though this fixed the PSAL NAN issue, but it doesnt 
        exportGroup(out_dir, merged_arr_vars)        

    else: 
        exportSeparate(out_dir, merged_arr)
    
    print('EXTRACT complete.')

    return out_dir



