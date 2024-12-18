#Import packages

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio as rs
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import os
from osgeo import gdal
from cubbie.read_write_insar_utilities import isce_read_write
from cubbie.math_tools import grid_tools
import geopandas as gp

#Set and change to working directory

work_dir = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/example_isce_int/20180105_20180117'
os.chdir(work_dir)

#Save subdirectories

subfolder_path = [f.path for f in os.scandir(work_dir) if f.is_dir()]


file = os.path.join(work_dir, 'geo_phase.int')

xarray, yarray, data = isce_read_write.read_scalar_data(file)

x_bbox, y_bbox, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, [-115.6, -115.2, 32.6, 33])

fig, ax = plt.subplots(figsize=[8,8], dpi=300)

shp_dir = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/shapefiles/IF_longterm/shapefiles'
qfaults_shp = os.path.join(shp_dir, 'sectionsALL.shp')
road_sec_shp = os.path.join(shp_dir, 'tl_2015_06_prisecroads.shp')
us_border_shp = os.path.join(shp_dir, 'cb_2018_us_nation_5m.shp')


qf = gp.read_file(qfaults_shp)
shdf = qf.to_crs("EPSG:4326") #explicit WGS to EPSG?
roads_sec = gp.read_file(road_sec_shp)
shdf_sec = roads_sec.to_crs("EPSG:4326")
us_border = gp.read_file(us_border_shp)


shdf.plot(ax=ax, edgecolor="red", linewidth=0.3)
shdf_sec.plot(ax=ax, edgecolor="black", linewidth=0.3)
us_border.plot(ax=ax, facecolor='None', edgecolor="black", linewidth=0.3, linestyle=(5, (10, 3))) #long dash with offset


ax.imshow(data_bbox, cmap='viridis', extent=[-115.6, -115.2, 32.6, 33])
plt.xlabel("longitude")
plt.ylabel("latitude")

plot_title = " ".join(file.split('/')[-2:]) 
plt.title(plot_title)

plt.show()

