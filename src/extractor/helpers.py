# Import modules
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import requests
import re
from pyproj import Transformer
import os
import calendar 
from datetime import datetime


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
    
    
# Function to filter XARRAY based on platform, Var and DEPTH
def filter_xarr_DEPTH(df_toPlot, data_dict, platform, depth_range):
    
    # find indices for each platform for the selected data
    index = df_toPlot[df_toPlot['Platform']==platform].index.tolist()
    
    # Filer data using the indexes of the filtered elements
    xarr_sel = data_dict[platform]['data'].isel(TIME=index,
                                                LATITUDE=index,
                                                LONGITUDE=index,
                                                DEPTH=slice(depth_range[0], depth_range[1]+1))
    return xarr_sel


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


# Function to check whether data should be aligned if vmin = 1, and align if so if has not been done already
def check_alignment(data_dict, pc, var, align_and_nan, vmin_dict):
    
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
        
            
# Function to plot a specific variable across the merged platforms
def plotVar_MergedPlatforms(merged_arr_var, var, title):
    plt.figure()
    merged_arr_var[var].plot() 
    plt.title(title)
    

# Function to plot a specific variable
def plotFilteredVar(data_xarr_var, title):
    plt.figure()
    # display(data_xarr)
    data_xarr_var.plot()
    plt.title(title)
    

# Function to define queries
def getQuery(pc, start, stop):
    dims = f'[{start}:1:{stop}]' # in the format [start,step,stop]
    return dims


# Function to plot an interactive plot with Bokeh
def plotStatic(df2p, title, long, lat, xlim, ylim, bbox_dict=False):
    output_notebook() # necessary to show the plot 
    
    # Define Hover tool
    hover = HoverTool(tooltips=[
        ("Index_ABS", "@Index_ABS"),
        ("(Long, Lat)", f"(@{long}, @{lat})"),
        ("Platform", "@Platform")])
    
    p = figure(plot_width=500, 
               plot_height=500, 
               tools=[hover, WheelZoomTool(), BoxZoomTool(), ResetTool(), PanTool()],
               title=title,
               x_range=xlim, 
               y_range=ylim,
               x_axis_type="mercator", y_axis_type="mercator" # this allows using the WGS84 units but non-linearly spaced
              )
    
    # Add Country Boundaries
    tile_provider = get_provider(OSM)
    p.add_tile(tile_provider)
    
    # Add positions by platform in different colors
    colors = ['blue','red','green','orange','purple','gold','cyan','lime','magenta']
    
    for i, pc in enumerate(df2p['Platform'].unique()):
        p.circle(long, lat, 
                 size=4, color=colors[i], fill_color='white', 
                 source=df2p[df2p['Platform']==pc], 
                 legend_label=f'Platform {pc}')

    # Add BBOX area
    if bbox_dict: 
        bbox_val = list(bbox_dict.values())[0]
        p.quad(left=bbox_val[0], right=bbox_val[1], top=bbox_val[3], bottom=bbox_val[2], 
               legend_label=list(bbox_dict.keys())[0], fill_color='grey', fill_alpha=0.0, line_color="black")

    # Add legend
    p.legend.location = "bottom_right"
    p.legend.click_policy="hide"

    # Create an output html file for displaying, if needed
#     output_file("interactive_plot_html.html")

    show(p)
    
    
# Function to reproject coordinate systems
def reproject(crs_from, crs_to, lat, long):
    transformer = Transformer.from_crs(crs_from=crs_from, crs_to=crs_to)
    
    reprj_long = transformer.transform(lat, long)[0]
    reprj_lat = transformer.transform(lat, long)[1]
    
    return reprj_long, reprj_lat


# Function to apply margin for extensions for Data Producer Notebook
def applyMargin(value):
    margin = 20 # ie 20% of value to apply
    margin_rel = value * margin / 100
    return margin_rel


def getTimeMargin(time):
    year = int(str(time)[:4])
    month = int(str(time)[4:])
    if month == 1:
        month1 = 12
        month2 = month+1
        year1 = year-1
        year2 = year
    elif month == 12:
        month1 = month-1
        month2 = 1
        year1 = year
        year2 = year+1
    else:
        month1 = month-1
        month2 = month+1
        year1 = year2 = year

    day_start = 1
    day_end = calendar.monthrange(year2, month2)[1] # only used for the last day of month

    time_start = datetime(year1, month1, day_start)
    time_end = datetime(year2, month2, day_end)

    return time_start, time_end

    
    