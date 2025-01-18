import matplotlib.pyplot as plt
import os
from cubbie.read_write_insar_utilities import isce_read_write
from cubbie.math_tools import grid_tools
import geopandas as gp
import glob
import json

plt.rcParams["font.family"] = "Arial"
plt.rcParams['axes.linewidth'] = 1

def do_fig1(params, ifg):

    xarray, yarray, data = isce_read_write.read_scalar_data(params['file'], verbose=False) #I suppress print functions

    _, _, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, params['extent'], verbose=False)

    fig, ax = plt.subplots(figsize=[8, 8], dpi=300)

    ax.imshow(data_bbox, cmap='viridis', extent=params['extent'])
    plt.xlabel("longitude", fontsize=10)
    plt.ylabel("latitude", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8, width=1)

    plot_title = " ".join(params["file"].split('/')[-2:])
    plt.title(plot_title, fontsize=14)

    if ifg:
        outfile_from_orig = str("_".join(ifg.replace('.', '/').split('/')[-2:]))
        outfile = "_".join(ifg.split('/')[-2:-1]) + '_' + outfile_from_orig
        plt.savefig(os.path.join(params['work_dir'], outfile + '_A.png'), dpi=300)
    else:
        outfile_from_orig = str("_".join(params["file"].replace('.', '/').split('/')[-2:])) + "_"
        outfile = outfile_from_orig + "_".join(params["work_dir"].split('/')[-1:])
        plt.savefig(os.path.join(params['work_dir'], outfile + '_A.png'), dpi=300)

    return

def do_fig2(params):

    xarray, yarray, data = isce_read_write.read_scalar_data(params['file'], verbose=False)

    _, _, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, params['extent'], verbose=False)

    qf_shp = gp.read_file(params["qfaults_shp"])
    #qf = qf_shp.to_crs("EPSG:4326")  # explicit WGS to EPSG?
    roads_shp = gp.read_file(params["road_shp"])
    us_border = gp.read_file(params["borders"])
    creepmeter = gp.read_file(params["creepmeters"])
    gps = gp.read_file(params["gps"])

    fig, ax = plt.subplots(figsize=[8, 8], dpi=300)

    qf_shp.plot(ax=ax, edgecolor="red", linewidth=0.6, zorder=20)
    roads_shp.plot(ax=ax, edgecolor="black", linewidth=0.6, zorder=10)
    us_border.plot(ax=ax, facecolor='None', edgecolor="black", linewidth=0.6, linestyle=(5, (10, 3)), zorder=5)  # long dash with offset
    creepmeter.plot(ax=ax, c='white', markersize=25, edgecolor='black', linewidth=0.5, marker="^", zorder=15)
    gps.plot(ax=ax, c='white', markersize=30, marker=".", edgecolor='black', linewidth=0.5, zorder=15)

    ax.imshow(data_bbox, cmap='viridis', extent=params['extent'])
    plt.xlabel("longitude", fontsize=10)
    plt.ylabel("latitude", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8, width=1)

    plot_title = " ".join(params["file"].split('/')[-2:])
    plt.title(plot_title, fontsize=14)

    outfile_from_orig = str("_".join(params["file"].replace('.', '/').split('/')[-2:])) + "_"
    outfile = outfile_from_orig + "_".join(params["work_dir"].split('/')[-1:])
    plt.savefig(os.path.join(params['work_dir'], outfile + '_B.png'), dpi=300)

    return


if __name__ == "__main__":

    param_file = '/media/mtan/rocket/mtan/IF_longterm/processing/merged/params.json'
    with open(param_file) as f:
        params_raw = f.read()

    params = json.load(params_raw)

    for ifg in list(glob.glob('./*/geo_phase.int', root_dir=os.getcwd(), recursive=True)):
        do_fig1(params)
        do_fig2(params)