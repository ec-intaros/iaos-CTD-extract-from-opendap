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
    help='This service performs extraction of NetCDF data from a remote OPeNDAP server.',
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
    help='The variable of interest (e.g. "TEMP"), or a list of them separated by a comma and without spaces (e.g. "TEMP,PRESS").',
    type=click.STRING,
)
@click.option(
    '--group',           
    '-g',           
    'group',           
    required=False,  
    help='if group==true (“true”, “1”, “t”, “yes”, “y”, and “on”), group all vars into a unique output file. If group==false (“false”, “0”, “f”, “no”, “n”, and “off”), create an output file for each var.',
    type=click.BOOL,
)
@click.option(
    '--format',           
    '-f',           
    'format',           
    required=True,  
    help='The format(s) desired for the generated output. Use: "csv" for CSV, "netcdf4" for NetCDF4, "csv,netcdf4" for both.',
    type=click.STRING,
)
@click.pass_context
def main(ctx, **kwargs):
    print(ctx)
    print('ciao')
    
    stop
    set_env()
    
    # dump the CWL and params (if requested)
    dump(ctx)

    # change directory to local 
    if 'TMPDIR' in os.environ:
        cwd = os.getcwd()
        os.chdir(os.environ['TMPDIR']) # change dir to the TMPDIR
    
    #==> Read Input Arguments
    depth = kwargs['depth']
    bbox = [int(bb) for bb in kwargs['bbox'].split(',')]
    time1 = str(kwargs['time1'])
    time2 = str(kwargs['time2'])
    print('ciao')
    
    
    
    
    
    
    
    stop
    #assert time1[:4] == time2[:4], 'ERROR: Years are different, please check.'
    vars_sel = kwargs['vars'].split(',')
    if len(vars_sel) > 1:
        # the flag 'group' is required in this case
        group = kwargs['group']
        assert group is not None, 'The flag "--group" is required when multiple variables are selected.'
    else: group = False
    formats = [ff.lower() for ff in kwargs['format'].split(',')]
    
    #==> Apply Extractor
    out_file = extract(depth, bbox, time1, time2, vars_sel, group, formats)
    logging.info(out_file)    
    stop
    # Move results / outputs from TMPDIR back to local
    if "TMPDIR" in os.environ:
        print('Here I have to move the output CSV/NetCDF file back to local')
#         move('catalog.json', os.path.join(cwd, 'catalog.json'))
#         move(item_out.id, os.path.join(cwd, item_out.id))

    logging.info('END.')

    
if __name__ == '__main__':
    main()
