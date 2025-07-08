#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import rasterio as rs
from cubbie.read_write_insar_utilities import isce_read_write
from Tectonic_Utils.read_write import netcdf_read_write
from cubbie.math_tools import grid_tools
import json

def date_string_to_datetime(dataframe, header, src_col, target_col, sorted=True):
    """
    returns dataframe with converted datetime [YYYY-MM-DD] of YYYYMMDD string.
    compatible with matplotlib plotting x-axis 
    YYYYMMDD is MT's preferred shorthand, easier to code here...

    Parameters:
    -------
    dataframe: pandas dataframe or path to .csv file
    header: line of header, usually 0 or None
    src_col: column that holds YYYYMMDD string
    target_col: column to update with datetime
    sorted: default True, returns sorted dataframe by date

    Returns:
    --------
    sorted dataframe by date
    
    """
    if '.csv' in dataframe: 
        df = pd.read_csv(dataframe, header=header)
    else:
        df = dataframe
    for i, date in enumerate(np.array(df[src_col])):
        df.loc[[i], [target_col]] = datetime.strptime(str(date), "%Y%m%d")
    if sorted:
        return df.sort_values(by=target_col)
    else:
        return df
    
def save_pygmt(template_filepath, append_name):
    head, tail = os.path.split(template_filepath)
    new_file = str("_".join(tail.split('.')[:-1]))
    outfile_path = head + '/' + new_file + append_name
    return outfile_path

def isce_to_grd(isce_file_path, units, param_file):
    """
    convert ISCE-generated .int file to .grd file. May work on other ISCE-generated files

    Parameters:
    --------
    isce_file_path: full path to file
    units: string to save what the pixel units are
    param_file: full path to parameters files. This code only uses param['extent']

    Returns:
    --------
    print statements confirming files were made
    
    """
    f = open(param_file)
    params = json.load(f)
    f.close()

    head, tail = os.path.split(isce_file_path)
    new_file = str("_".join(tail.split('.')[:-1]))
    outfile_path = head + '/' + new_file + '2disp.grd'

    xarray, yarray, data = isce_read_write.read_scalar_data(isce_file_path)
    x_arr, y_arr, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, params['extent'], verbose=False)
    netcdf_read_write.produce_output_netcdf(x_arr, y_arr, data_bbox*(5.8/(np.pi*4)), zunits=units,
                                            netcdfname=outfile_path)
    #print(isce_file_path, " ---> ", outfile_path)
    print("----------------------")
        
def get_tiff_info(file, verbose=True):
  """
  get geotiff info; hardcoded to extract specific info

  Parameters:
  -------
  file: filename string
  verbose: default true; will print more tiff info

  Returns:
  -------
  width
  height
  extent of raster

  """
  ds = rs.open(file)
  w = ds.width
  h = ds.height
  b = ds.bounds
  c = ds.crs
  t = ds.transform
  extent = list([b[0], b[2], b[1], b[3]])
  wm = b[2] - b[0]
  hm = b[3] - b[1]

  """
  minx = gt[0]
  miny = gt[3] + width*gt[4] + height*gt[5] 
  maxx = gt[0] + width*gt[1] + height*gt[2]
  maxy = gt[3]
  """

  if verbose:
    print("columns: ", w)
    print("rows: ", h)
    print("bounds: ", b)
    print("extent:", extent)
    print("width of area [crs]: ", wm)
    print("height of area [crs]: ", hm)
    print("crs: ", c)
    print("transform: ", t)
  ds.close
  return w, h, extent

def read_json(file):
    f = open(file)
    params = json.load(f)
    f.close()
    return params