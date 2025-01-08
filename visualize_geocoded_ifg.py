import matplotlib.pyplot as plt
import os
from cubbie.read_write_insar_utilities import isce_read_write
from cubbie.math_tools import grid_tools
import geopandas as gp
import glob
import json

plt.rcParams["font.family"] = "Arial"
plt.rcParams['axes.linewidth'] = 1

def do_fig1(params):

    xarray, yarray, data = isce_read_write.read_scalar_data(params['file'], verbose=False) #I suppress print functions

    _, _, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, params['extent'], verbose=False)

    fig, ax = plt.subplots(figsize=[8, 8], dpi=300)

    ax.imshow(data_bbox, cmap='viridis', extent=params['extent'])
    plt.xlabel("longitude", fontsize=10)
    plt.ylabel("latitude", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8, width=1)

    plot_title = " ".join(params["file"].split('/')[-2:])
    plt.title(plot_title, fontsize=14)

    outfile_from_orig = str("_".join(params["file"].replace('.', '/').split('/')[-2:])) + "_"
    outfile = outfile_from_orig + "_".join(params["work_dir"].split('/')[-1:])
    plt.savefig(os.path.join(params['work_dir'], outfile + '_A.png'), dpi=300) #json has experiment_dir also

    return

