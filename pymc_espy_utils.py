#!/usr/bin/env python

import numpy as np
import json

from elastic_stresses_py import PyCoulomb
from elastic_stresses_py.PyCoulomb.inputs_object import io_intxt
from elastic_stresses_py.PyCoulomb import coulomb_collections as cc
from tectonic_utils.geodesy import insar_vector_functions

import arviz as az
from arviz import from_netcdf
#import pymc as pm

def get_dU(disp_points):
    """for testing"""
    dU = []
    #dE = []
    #dN = []
    for point in disp_points:
        #dE = np.append(dE, point.dE_obs)
        #dN = np.append(dN, point.dN_obs)
        dU = np.append(dU, point.dU_obs)
    return dU

def get_los(disp_points, platform):
    """
    Function takes in elastic_stresses_py displacement object and returns line of sight displacement

    Parameters:
    -------
    disp_points: elastic_stresses_py displacement object

    platform: s1d (sentinel-1 descending), s1a (sentinel-1 ascending), uav (uavsar)

    Returns:
    --------
    los data vector * [-1]
    
    """
    if platform == 's1d':
        azi = 190
        inc = 37
    elif platform == 's1a':
        azi = 12
        inc = 37
    elif platform == 'uav':
        azi = -95
        inc = 60
    else:
        print("defaulting to s1d geometry...")
    los = []
    for point in disp_points:
        los = np.append(los, insar_vector_functions.def3D_into_LOS(point.dE_obs, point.dN_obs, point.dU_obs, azi, inc))
    return los*-1

def read_intxt(input_file, mu=30e9, _lame1=30e9):
    """
    Reads input file specified in elastic_stresses_py documentation and makes an input object

    Parameters:
    -------
    input_file: textfile formatted from elastic_stresses_py documentation

    mu (optional): default 30e9

    _lame1 (optional): default 30e9

    Returns:
    --------
    elastic_stresses_py input object
    
    """
    sources, receivers = [], []
    receiver_horiz_profile = None
    [PR1, FRIC, minlon, maxlon, zerolon, minlat, maxlat, zerolat] = io_intxt.get_general_compute_params(input_file)
    [start_x, end_x, start_y, end_y, xinc, yinc] = io_intxt.compute_grid_params_general(minlon, maxlon, minlat, maxlat,
                                                                               zerolon, zerolat)
    ifile = open(input_file, 'r')
    for line in ifile:
        temp = line.split()
        if len(temp) > 0:
            if temp[0] == 'Source_WC:':  # wells and coppersmith convenient format
                one_source_object = io_intxt.get_source_wc(line, zerolon, zerolat)
                sources.append(one_source_object)
            if temp[0] == 'Source_Patch:':  # source-slip convenient format
                one_source_object = io_intxt.get_source_patch(line, zerolon, zerolat)
                sources.append(one_source_object)
    ifile.close()

    # Wrapping up the inputs.
    input_obj = cc.Input_object(PR1=PR1, FRIC=FRIC, depth=0, start_gridx=start_x, finish_gridx=end_x,
                             start_gridy=start_y, finish_gridy=end_y, xinc=xinc, yinc=yinc, minlon=minlon,
                             maxlon=maxlon, zerolon=zerolon, minlat=minlat, maxlat=maxlat, zerolat=zerolat,
                             receiver_object=receivers, source_object=sources,
                             receiver_horiz_profile=receiver_horiz_profile)
    return input_obj

def do_update(default_inputs, slip, width, dip):
    """
    Update fault slip object from new slip, width, dip values

    Parameters:
    -------
    default_inputs: input object

    slip: slip [m]

    width: width of rectangular slip patch (i.e. top - bottom depth) [km]

    dip: dip of rectangular slip patch [degrees]

    Returns:
    --------
    input object with modified source of slip, width, dip
    
    """
    internal_source = PyCoulomb.fault_slip_object.fault_slip_object.coulomb_fault_to_fault_object(default_inputs.source_object)
    internal_source[0].slip = slip
    internal_source[0].width = width
    internal_source[0].dip = dip

    #lons, lats = internal_source[0].get_four_corners_lon_lat()
    #print(lons, lats)

    modified_source = PyCoulomb.fault_slip_object.fault_slip_object.fault_object_to_coulomb_fault(internal_source, 
                                                                                zerolon_system=default_inputs.zerolon, 
                                                                                zerolat_system=default_inputs.zerolat)
    return default_inputs.modify_inputs_object(source_object=modified_source)

def read_json(file):
    f = open(file)
    params = json.load(f)
    f.close()
    return params

def save_pymc_model(model, location):
    model.to_netcdf(location)
    print("Model saved: \n{} \nlocation: {}".format(str(model), str(location)))

def calc_stress_drop(pymc_models, mu):
    dTaus = []
    for model in pymc_models:
        loaded_model = from_netcdf(model)
        w_mean = np.mean(az.convert_to_dataset(loaded_model)['width'])*1000 #km-->m
        s_mean = np.mean(az.convert_to_dataset(loaded_model)['slip'])
        dTau = s_mean * mu / (2 * w_mean) #[m*kPa / m]
        dTaus = np.append(dTaus, dTau)
    return dTaus