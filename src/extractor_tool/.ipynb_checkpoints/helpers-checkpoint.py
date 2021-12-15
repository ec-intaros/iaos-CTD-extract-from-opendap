# Import modules
import pandas as pd
import numpy as np
import requests
import re
import pprint
import xarray as xr
import os

from .extractor import *



def set_env():
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib64'
    os.environ['PREFIX'] = '/srv/conda/envs/env_intaros'
    os.environ["PROJ_LIB"] = os.path.join(os.environ["PREFIX"], "conda-otb/share/proj/")

    
# Function to retrieve DDS info (sizes of 'TIME', 'LATITUDE', 'LONGITUDE', 'DEPTH')
def retrieveDDSinfo(dds):
    
    # Function to extract the dimension of a specific keyword
    def findDim(txt, key):
        sel = [elem for elem in txt if key in elem]
        assert len(sel)==1

        dim = int(sel[0].split(' = ')[-1].strip(']'))-1
        return dim    
    
    r = requests.get(dds)
    
    unique_list = set(re.findall(r'\[.*?\]',r.text))
#     print('Unique elements:', unique_list)
    
    keys = ['TIME', 'LATITUDE', 'LONGITUDE', 'DEPTH']

    dim_dict = {}

    for k in keys:
        dim_dict[k] = findDim(txt=unique_list, key=k)

    return dim_dict


# Function to build a string for query on URL
def getQueryString(mydict, keylist):
#     print('Retriving Query string for keywords:', keylist)
    
    que_str = ','.join([f'{key}[0:1:{mydict[key]}]' for key in mydict.keys() if key in keylist])
#     print('Query string:', que_str)
    return que_str


# Function to fetch data using xarray.open_dataset and save attributes
def fetch_data(url, year):
    remote_data = xr.open_dataset(
        url,
        decode_times=False,
    )   
    
    lon_min = float(remote_data.attrs['geospatial_lon_min'])
    lon_max = float(remote_data.attrs['geospatial_lon_max'])
    lat_min = float(remote_data.attrs['geospatial_lat_min'])
    lat_max = float(remote_data.attrs['geospatial_lat_max'])

    plat_code = remote_data.attrs['platform_code']
    plat_name = remote_data.attrs['platform_name']
    year = year #url.split('subset_')[1][:4]
    dtype = remote_data.attrs['data_type']
    title = remote_data.attrs['title']
    inst = remote_data.attrs['instrument']
    v_min = remote_data.attrs['geospatial_vertical_min']
    v_max = remote_data.attrs['geospatial_vertical_max']

    data_attr = np.array([plat_code, plat_name, year, dtype, title, inst, 
                         v_min, v_max, lon_min, lon_max, lat_min, lat_max])   
    
    return remote_data, data_attr


# Function to save Attributes to a database
def getAttributes(my_df, my_dict):
    
    for key in my_dict.keys():

        my_df.loc[key,'Platform_code'] = [my_dict[key]['data_attr'][0].astype(str)]
        my_df.loc[key,'Platform_name'] = [my_dict[key]['data_attr'][1].astype(str)]
        my_df.loc[key,'Year'] = [my_dict[key]['data_attr'][2].astype(int)]
        my_df.loc[key,'Data_type'] = [my_dict[key]['data_attr'][3].astype(str)]
        my_df.loc[key,'Title'] = [my_dict[key]['data_attr'][4].astype(str)]
        my_df.loc[key,'Instrument'] = [my_dict[key]['data_attr'][5].astype(str)]
        my_df.loc[key,'Vertical_min'] = [my_dict[key]['data_attr'][6].astype(float)]
        my_df.loc[key,'Vertical_max'] = [my_dict[key]['data_attr'][7].astype(float)]
        my_df.loc[key,'Lon_min'] = [my_dict[key]['data_attr'][8].astype(float)]
        my_df.loc[key,'Lon_max'] = [my_dict[key]['data_attr'][9].astype(float)]
        my_df.loc[key,'Lat_min'] = [my_dict[key]['data_attr'][10].astype(float)]
        my_df.loc[key,'Lat_max'] = [my_dict[key]['data_attr'][11].astype(float)]

    return my_df    

    
# Function to adjust data array with Vertical Min
def adjust_with_vmin(xarr_var, value):
    
    xarr_var = np.insert(xarr_var, obj=0, values=value, axis=1)
    #print('Adjusted shape:', xarr_var.shape)
    #display(xarr_var)

    # Remove last element (ie same dimension as original)
    xarr_var_trimmed = xarr_var[:,:-1]
    #print('Trimmed shape:', xarr_var_trimmed.shape)
    #display(xarr_var_trimmed)
    
    # return the trimmed array, to re-assign it to the original element
    return xarr_var_trimmed


# Function to define queries
def getQuery(pc, start, stop):
    dims = f'[{start}:1:{stop}]' # in the format [start,step,stop]
    return dims