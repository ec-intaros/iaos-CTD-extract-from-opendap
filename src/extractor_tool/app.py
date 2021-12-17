# Application - OPeNDAP Data Extractor

#==============================================
# Import modules
import os, sys, copy, logging
from shutil import move

import click, click2cwl
from click2cwl import dump

from .helpers import *
from .extractor import *

logging.basicConfig(stream=sys.stderr, 
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')

@click.command(
    short_help='OPeNDAP Data Extractor',
    help='This service performs: i) extraction of NetCDF data from a remote OPeNDAP server; ii) data filtering by bounding box, time and depth; and iii) output data export to CSV and/or NetCDF file.',
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option(
    '--depth', 
    '-d', 
    'depth', 
    help='The Depth variable, expressed as integer in meters.',
    required=True,
    type=click.INT,
)
@click.option(
    '--bbox',           
    '-b',           
    'bbox',           
    required=True,          
    help='The Bounding Box (BBOX) extent, expressed in degrees in the format "minLong, maxLong, minLat, maxLat", e.g. "-4, 7, 57, 62".',
    type=click.STRING,
)
@click.option(
    '--time1',           
    '-t1',           
    'time1',           
    required=True,  
    help='The date of the starting time, expressed in the format YYYYMMDD.',
    type=click.INT,
)
@click.option(
    '--time2',           
    '-t2',           
    'time2',           
    required=True,  
    help='The date of the ending time, expressed in the format YYYYMMDD.',
    type=click.INT,
)
@click.option(
    '--vars',           
    '-v',           
    'vars',           
    required=True,  
    help='The variable of interest (e.g. "TEMP"), or a list of them separated by a comma and without spaces (e.g. "TEMP,PRES,PSAL,CNDC").',
    type=click.STRING,
)
@click.option(
    '--group',           
    '-g',           
    'group',           
    required=False, 
    default='False',
    show_default=True, 
    help='if group==true (“true”, “1”, “t”, “yes”, “y”, and “on”), group all vars into a unique output file. If group==false (“false”, “0”, “f”, “no”, “n”, and “off”), create an output file for each var.',
    type=click.BOOL,
)
@click.option(
    '--format',           
    '-f',           
    'format',           
    required=False,  
    default='NetCDF4',
    show_default=True, 
    help='The format(s) desired for the generated output. Use: "csv" for CSV, "netcdf4" for NetCDF4, "csv,netcdf4" for both.',
    type=click.STRING,
)
@click.pass_context
def main(ctx, **kwargs):

    set_env()
    
    # dump the CWL and params (if requested)
    dump(ctx)

    # change directory to local 
    if 'TMPDIR' in os.environ:
        cwd = os.getcwd()
        os.chdir(os.environ['TMPDIR']) # change dir to the TMPDIR
    
    #==> Read Input Arguments
    depth = kwargs['depth']
    bbox = kwargs['bbox']
    time1 = str(kwargs['time1'])
    time2 = str(kwargs['time2'])
    vars_sel = kwargs['vars'].split(',')
    group = kwargs['group']
    formats = kwargs['format']
    
    #==> Apply Extractor Tool
    out_dir = extract(depth, bbox, time1, time2, vars_sel, group, formats)
    logging.info(f'File exported in the directory: {out_dir}')    
    
    # Move results / outputs from TMPDIR back to local
    if "TMPDIR" in os.environ:
        move(out_dir, cwd)

    logging.info('END.')

    
if __name__ == '__main__':
    main()
