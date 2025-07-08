#!/usr/bin/env python

from elastic_stresses_py import PyCoulomb
from elastic_stresses_py.PyCoulomb import coulomb_collections as cc
from elastic_stresses_py.PyCoulomb.fault_slip_triangle import triangle_okada
from elastic_stresses_py.PyCoulomb.point_source_object import point_sources
from elastic_stresses_py.PyCoulomb.disp_points_object.disp_points_object import Displacement_points
from elastic_stresses_py.PyCoulomb import utilities, io_additionals, run_dc3d, run_mogi, conversion_math
from elastic_stresses_py.PyCoulomb.inputs_object import io_intxt

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os

import pytensor
import pytensor.tensor as pt
import pymc as pm
from pymc import HalfCauchy, Model, Normal, sample
from pytensor.graph import Apply, Op

from pymc_espy_utils import get_los, read_intxt, do_update

def do_okada(params, slip, width, dip):
    """
    Perform Okada dislocation of a rectangular slip patch with slip, width, dip

    Parameters:
    -------
    params: textfile of parameters 

    slip: slip [m]

    width: width of rectangular slip patch (i.e. top - bottom depth) [km]

    dip: dip of rectangular slip patch [degrees]

    Returns:
    --------
    line of sight data array
    
    """
    inputs = do_update(inputs_orig, slip[0], width[0], dip[0])

    model_disp_points = run_dc3d.compute_ll_def(inputs, params, disp_points)

    MyOutObject = cc.Out_object(x=[], y=[], x2d=[], y2d=[], u_disp=[], v_disp=[], w_disp=[],
                                strains=[], model_disp_points=model_disp_points,
                                zerolon=inputs.zerolon, zerolat=inputs.zerolat,
                                source_object=inputs.source_object, receiver_object=inputs.receiver_object,
                                receiver_normal=[], receiver_shear=[],
                                receiver_coulomb=[], receiver_profile=[])

    los = get_los(MyOutObject.model_disp_points)

    return los

def my_loglike(slip, width, dip, sigma, data):
    """
    Takes parameter values, sigma, and data and evaluates loglike of Gaussian distribution

    Parameters:
    -------
    slip: slip [m]

    width: width of rectangular slip patch (i.e. top - bottom depth) [km]

    dip: dip of rectangular slip patch [degrees]

    sigma: uncertainty in input data
    
    data: array of data

    Returns:
    --------
    loglike of Gaussian distribution
    
    """
    for param in (slip, width, dip, sigma, data):
        if not isinstance(param, (float, np.ndarray)):
            raise TypeError(f"Invalid input type to loglike: {type(param)}") 
    model = do_okada(params, slip, width, dip)
    return -0.5 * ((data - model / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma))

class LogLike(Op):

    def make_node(self, slip, width, dip, sigma, data):
        slip = pt.as_tensor(slip)
        width = pt.as_tensor(width)
        dip = pt.as_tensor(dip)
        sigma = pt.as_tensor(sigma)
        data = pt.as_tensor(data)

        inputs = [slip, width, dip, sigma, data]
        outputs = [data.type()]

        return Apply(self, inputs, outputs)
    
    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        slip, width, dip, sigma, data = inputs

        loglike_eval = my_loglike(slip, width, dip, sigma, data)

        outputs[0][0] = np.asarray(loglike_eval)

def custom_dist_loglike(data, slip, width, dip, sigma, x):
    """
    Evaluates loglike of Gaussian distribution with data and model

    Parameters:
    -------
    data: array of input data

    slip: slip [m]

    width: width of rectangular slip patch (i.e. top - bottom depth) [km]

    dip: dip of rectangular slip patch [degrees]

    sigma: uncertainty in input data
    
    x: model values

    Returns:
    --------
    loglike of Gaussian distribution
    
    """
    # data, or observed is always passed as the first input of CustomDist
    return LogLike(slip, width, dip, sigma, x, data)

if __name__ == "__main__":

    #########################################################
    # really important, set globals once before running !!  #
    #########################################################
    wd = "/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/codes/experiment2/pymc_tests/test3"
    os.chdir(wd)

    inputs_orig = read_intxt("normal_fault_in.txt")
    params = PyCoulomb.configure_calc.configure_stress_calculation('my_config_normal.txt')
    disp_points = io_additionals.read_disp_points("normal_fault_disp.txt")
    data = np.loadtxt('los_data_2000.txt')

    sigma = 1.0
    #########################################################
    #########################################################

    with pm.Model() as no_grad_model:
        slip = pm.Normal("slip", mu=0.047, sigma=0.005) #mu=mean, sigma=st dev.
        #depth = pm.Uniform("depth", lower=0.0, upper=1.0, initval=0.75)
        width = pm.TruncatedNormal("width", mu=0.75, sigma=0.05, lower=0.65, upper=0.85)
        #dip = pm.Normal("dip", mu=80, sigma=5)
        dip = pm.TruncatedNormal("dip", mu=45, sigma=1, lower=30, upper=60, initval=45)
        #dip = pm.Uniform("dip", lower=75.0, upper=90.0, initval=80)

        # use a CustomDist with a custom logp function
        likelihood = pm.CustomDist(
            "likelihood", slip, width, dip, sigma, observed=data, logp=custom_dist_loglike
        )

    """
    with no_grad_model:
        # Use custom number of draws to replace the HMC based defaults
        idata_no_grad = pm.sample(3000, tune=1000)
    """