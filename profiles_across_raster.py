#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
from cubbie.read_write_insar_utilities import isce_read_write
from cubbie.math_tools import grid_tools
import pandas as pd
import numpy as np
import glob
import json
from cmcrameri import cm
import utils
import geopandas as gp
import pygmt

def connect_points(ploc):
    lat0, lat1, lon0, lon1 = ploc[:, 3], ploc[:, 4], ploc[:, 5], ploc[:, 6]
    names = ploc[:, 2]
    lats = []
    lons = []
    for i in range(len(lat0)):
        lats.append([lat0[i], lat1[i]])
        lons.append([lon0[i], lon1[i]])
    return lons, lats, names

def midpoint(x, y):
    return (x[0] + x[1])/2,  (y[0] + y[1])/2

def simple_plot_profiles(lons, lats):
    for i in range(len(lons)):
        plt.plot(lons[i], lats[i], c='k', linewidth=1)
        plt.text(lons[i][0], lats[i][0], i, size=6,
                ha="center", va="center",
                bbox=dict(boxstyle="circle", linewidth=0.5, facecolor='white'), color='k'
                )
    plt.show()

def plot_profiles(params, ifg, xs, ys, labels):
    
    xarray, yarray, data = isce_read_write.read_scalar_data(ifg) #I suppress print functions

    _, _, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, params['extent'], verbose=False)

    fig, ax = plt.subplots(figsize=[8, 8], dpi=300)

    ax.imshow(data_bbox, cmap=cm.batlow, extent=params['extent'])

    for i in range(len(xs)):
        plt.plot(xs[i], ys[i], c='k', linewidth=1, zorder=25)
        xloc, yloc = midpoint(xs[i], ys[i])
        plt.text(xloc, yloc, str(labels[i]), size=4,
                ha="center", va="center", zorder=30,
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

def sort(subset):
    """
    sort profile lon/lat coordinates and add a pad to plot region around profile

    Parameters:
    --------
    subset: profiles array

    Returns:
    --------
    lons: longitude coordinates [lonA, lonB]
    lats: latitude coordinates [latA, latB]
    name: name of profile
    """
    lons = []
    lats = []
    names = subset[:, 2]

    if subset.shape[0] > 1:
        for i in range(subset.shape[0]):
            if subset[i][5] < subset[i][6]:
                lats.append([subset[i][3], subset[i][4]])
                lons.append([subset[i][5], subset[i][6]])
            else:
                lats.append([subset[i][4], subset[i][3]])
                lons.append([subset[i][6], subset[i][5]])
    else:
        if subset[5] < subset[6]:
            lats.append([subset[3], subset[4]])
            lons.append([subset[5], subset[6]])
        else:
            lats.append([subset[4], subset[3]])
            lons.append([subset[6], subset[5]])
        
    return lons, lats, names

def sort_and_pad(subset):
    sorted = np.sort(subset)
    pad = np.sqrt( (sorted[0] - sorted[1])**2 + (sorted[3] - sorted[2])**2 )
    pad = 0.05
    if subset[0] < subset[1]:
        xs = [subset[0], subset[1]]
        ys = [subset[2], subset[3]]
    else:
        xs = [subset[1], subset[0]]
        ys = [subset[3], subset[2]]
    return xs, ys, [sorted[0] - pad, sorted[1] + pad, sorted[2] - pad, sorted[3] + pad]

def plot_profiles_gmt(grid_file, isce_file, ploc, ploc_idx, title, outfile=True):
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP='ddd.xx') # decimal degrees

    xarray, yarray, data = isce_read_write.read_scalar_data(isce_file)
    _, _, data_bbox = grid_tools.clip_array_by_bbox(xarray, yarray, data, params['extent'], verbose=False)
    
    subset = np.sort([ploc[ploc_idx][5], ploc[ploc_idx][6], ploc[ploc_idx][4], ploc[ploc_idx][3]])
    xs, ys, sorted = sort_and_pad(subset)

    df = pygmt.project(
    x=xarray, 
    y=yarray, 
    z=data_bbox,
    unit=True,
    #center=[ploc_arr[img][5], ploc_arr[img][3]],  # Start point of survey line (longitude, latitude)
    #endpoint=[ploc_arr[img][6], ploc_arr[img][4]],  # End point of survey line (longitude, latitude)
    center=[xs[0], ys[0]],
    endpoint=[xs[1], ys[1]],
    generate=0.005,  # Output data in steps of 0.005 km [unit=True]
    width=[-0.1, 0.1],
    output_type='pandas'
    #outfile='/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Data/IF_longterm/creep_event_files/profiles/test_profile.dat'
    )

    df = pygmt.grdtrack(grid=grid_file, points=df, newcolname="disp")

    pady = np.mean(df.disp)
    padx = np.mean(df.p)/4
    region_padded = [0-padx, max(df.p)+padx, min(df.disp)-pady, max(df.disp)+pady]

    fig.grdimage(grid=grid_file, cmap="batlow", frame="ag", region=sorted) #"-115.420286/-115.414122/32.739045/32.744088"
    fig.coast(
        region=sorted,
        #projection="M10c",
        #land="tan",
        #water="steelblue",
        borders="1/0.5p,black,-",
        #frame=["WSne", "a"],
        # Set the label alignment (+a) to right (r)
        map_scale="jBL+o1c/1c+c-7+w500e+f+lm+ar",
        # Fill the box in white with a transparency of 30 percent, add a solid
        # outline in darkgray (gray30) with a thickness of 0.5 points, and use
        # rounded edges with a radius of 3 points
        box="+gwhite@30+p0.5p,gray40,solid+r3p",
    )

    with fig.inset(
        position="jBR+o0.1c",
        box="+pblack",
        region=params['extent'],
        projection="U11N/4c"
    ):
        # Highlight the Japan area in "lightbrown"
        # and draw its outline with a pen of "0.2p".
        fig.coast(
        # Set the projection to Mercator, and the plot width to 10 centimeters
        projection="M4c",
        # Set the region of the plot
        region=params['extent'],
        # Set the frame of the plot, here annotations and major ticks
        frame=["WSne", "a"],
        # Set the color of the land to "darkgreen"
        land="whitesmoke",
        # Set the color of the water to "lightblue"
        water="lightblue",
        # Draw national borders with a 1-point black line
        borders="1/0.5p,black,-",

    )
        # Plot a rectangle ("r") in the inset map to show the area of the main
        # figure. "+s" means that the first two columns are the longitude and
        # latitude of the bottom left corner of the rectangle, and the last two
        # columns the longitude and latitude of the upper right corner.
        fig.plot(data=qf_shp)
        fig.plot(
            #x=[ploc_arr[img][5], ploc_arr[img][6]],
            #y=[ploc_arr[img][3], ploc_arr[img][4]],  # Latitude in degrees North
            x=xs,
            y=ys,
            # Draw a 2-points thick, red, dashed line for the survey line
            pen="2p,red,solid")
        rectangle = [region_padded]
        fig.plot(data=rectangle, style="r+s", pen="2p,blue")
        #fig.text(text="US", x=-115.62, y=32.66, angle=5)
        #fig.text(text="MEX", x=-115.62, y=32.648, angle=5)

    fig.plot(
                #x=[ploc_arr[img][5], ploc_arr[img][6]],
                #y=[ploc_arr[img][3], ploc_arr[img][4]],  # Latitude in degrees North
        x=xs,
        y=ys,
                # Draw a 2-points thick, red, dashed line for the survey line
        pen="2p,red,solid")

            # Add labels "A" and "B" for the start and end points of the survey line
    fig.text(
                #x=[ploc_arr[img][5], ploc_arr[img][6]],
                #y=[ploc_arr[img][3], ploc_arr[img][4]],
        x=xs,
        y=ys,
        text=["A", "B"],
        offset="0c/0.5c",  # Move text 0.2 centimeters up (y direction)
        font="16p",  # Use a font size of 15 points
        fill="white"
        )
    
    #---------------

    fig.shift_origin(yshift="h+1.5c")

    fig.basemap(
        projection="X10/6",
        region=region_padded,
        frame=[f'wsne+t{title}']
    )

    fig.plot(
        x=df.p,
        y=df.disp,
        fill="gray",  # Fill the polygon in "gray"
        # Draw a 1-point thick, black, solid outline
        frame=["xafg+ldistance along profile [m]",
            "yafg+loffset [cm]"],
        style='c0.1c',
        pen=1,  # Force closed polygon
    )

    if outfile:
       fig.savefig(utils.save_pygmt(grid_file, append_name="_creepProfile.png"))

    fig.show() 

if __name__ == "__main__":

    param_file = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Data/IF_longterm/IF_longterm_ISCEinfo/params_local.json'
    f = open(param_file)
    params = json.load(f)
    f.close()

    qf_shp = gp.read_file(params["qfaults_shp"])
    roads_shp = gp.read_file(params["road_shp"])

    profile_loc = pd.read_csv(params['profiles'], header=0)
    ploc_arr = np.array(profile_loc)

    for grd in list(glob.glob('/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Data/IF_longterm/creep_event_files/*.grd')):
        print("")
        print("")
