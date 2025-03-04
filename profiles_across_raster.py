#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
from cubbie.read_write_insar_utilities import isce_read_write
from cubbie.math_tools import grid_tools
import geopandas as gp
import glob
import json
from cmcrameri import cm
from visualize_geocoded_ifg import do_fig1

def midpoint(x, y):
    return (x[0] + x[1])/2,  (y[0] + y[1])/2

def connect_points(ploc):
    lat0, lat1, lon0, lon1 = ploc[:, 3], ploc[:, 4], ploc[:, 5], ploc[:, 6]
    names = ploc[:, 2]
    lats = []
    lons = []
    for i in range(len(lat0)):
        lats.append([lat0[i], lat1[i]])
        lons.append([lon0[i], lon1[i]])
    return lats, lons, names

def do_fig1(params, ifg, xs, ys, labels):
    
    xarray, yarray, data = isce_read_write.read_scalar_data(ifg) #I suppress print functions

    _, _, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, params['extent'], verbose=False)

    fig, ax = plt.subplots(figsize=[8, 8], dpi=300)

    ax.imshow(data_bbox, cmap=cm.batlow, extent=params['extent'])

    for i in range(len(xs)):
        plt.plot(xs[i], ys[i], c='k', linewidth=1)
        text_x, text_y = midpoint(xs[i], ys[i])
        plt.text(text_x, text_y, str(labels[i]), size=4,
                ha="center", va="center",
                bbox=dict(boxstyle="circle", linewidth=0.5, facecolor='white'), color='k'
                )

    plt.xlabel("longitude", fontsize=10)
    plt.ylabel("latitude", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8, width=1)

    if ifg:
        plot_title = " ".join(ifg.split('/')[-2:])
        outfile_from_orig = str("_".join(ifg.replace('.', '/').split('/')[-2:]))
        outfile = "_".join(ifg.split('/')[-2:-1]) + '_' + outfile_from_orig
    else:
        plot_title = " ".join(params["file"].split('/')[-2:])
        outfile_from_orig = str("_".join(params["file"].replace('.', '/').split('/')[-2:])) + "_"
        outfile = outfile_from_orig + "_".join(params["work_dir"].split('/')[-1:])

    plt.title(plot_title, fontsize=14)
    plt.savefig(os.path.join('/media/mtan/rocket/mtan/IF_longterm/processing/merged/visualize_geocoded_ifg_profiles', outfile + '_prof.png'), dpi=300)
    plt.close()

    return

if __name__ == "__main__":

    param_file = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Data/IF_longterm/IF_longterm_ISCEinfo/params_local.json'
    f = open(param_file)
    params = json.load(f)
    f.close()