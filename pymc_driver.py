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
import os
from typing import Optional, Tuple

import pytensor
import pytensor.tensor as pt
import pymc as pm
from pytensor.graph import Apply, Op

from pymc_espy_utils import get_los, read_intxt, do_update, read_json
from pymc_visualize import plot_stats, plot_corner

def do_okada(slip, width, dip):
    """
    Perform Okada dislocation of a rectangular slip patch with slip, width, dip

    Parameters:
    -------
    slip: slip [m]

    width: width of rectangular slip patch (i.e. top - bottom depth) [km]

    dip: dip of rectangular slip patch [degrees]

    Returns:
    --------
    line of sight data array
    
    """
    inputs = do_update(inputs_orig, slip[0], width[0], dip[0])

    model_disp_points = run_dc3d.compute_ll_def(inputs, params_in, disp_points)

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
    model = do_okada(slip, width, dip)
    #return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)
    return -(0.5/sigma**2)*np.sum((data - model)**2) #eq. 5.1 Menke textbook

class LogLike(Op):

    def make_node(self, slip, width, dip, sigma, data) -> Apply:
        slip = pt.as_tensor(slip)
        width = pt.as_tensor(width)
        dip = pt.as_tensor(dip)

        inputs = [slip, width, dip, sigma, data]
        outputs = [data.type()]

        return Apply(self, inputs, outputs)
    
    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        slip, width, dip, sigma, data = inputs

        loglike_eval = my_loglike(slip, width, dip, sigma, data)

        outputs[0][0] = np.asarray(loglike_eval)

def gauss_draw(slip, width, dip, size=None):
        return np.random.Generator.random(slip, width, dip, size = size)

def custom_dist_loglike(data, slip, width, dip, sigma):
    """
    Evaluates loglike of Gaussian distribution with data and model

    Parameters:
    -------
    data: array of input data

    slip: slip [m]

    width: width of rectangular slip patch (i.e. top - bottom depth) [km]

    dip: dip of rectangular slip patch [degrees]

    sigma: uncertainty in input data

    Returns:
    --------
    loglike of Gaussian distribution
    
    """
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(slip, width, dip, sigma, data)

if __name__ == "__main__":

    #########################################################
    # really important, set globals once before running !!  #
    #########################################################
    params = read_json('/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/codes/experiment2/pymc_tests/test3/pymc_params_synth_50disp.json')
    #params = read_json('/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/codes/experiment2/pymc_tests/20180105_20180117/pymc_params.json')
    os.chdir(params['experiment_dir'])

    inputs_orig = read_intxt(params['inputs_orig'])
    params_in = PyCoulomb.configure_calc.configure_stress_calculation(params['params'])
    disp_points = io_additionals.read_disp_points(params['disp_points'])
    data = np.loadtxt(params['data'])

    sigma = params['sigma']
    #########################################################
    #########################################################
    
    loglike_op = LogLike()
    RANDOM_SEED = 58
    rng = np.random.default_rng(RANDOM_SEED)

    # initialize model
    with pm.Model() as no_grad_model:
        # priors
        # slip = pm.Normal("slip", mu=params['slip_mu'], sigma=params['slip_sigma']) #mu=mean, sigma=st dev.
        # width = pm.TruncatedNormal("width", mu=params['width_mu'], sigma=params['width_sigma'], lower=params['width_lower'], upper=params['width_upper'])
        # dip = pm.TruncatedNormal("dip", mu=params['dip_mu'], sigma=params['dip_sigma'], lower=params['dip_lower'], upper=params['dip_upper'], initval=params['dip_init'])
        
        # uniform priors
        slip = pm.Uniform("slip", lower=0.01, upper=0.06, initval=0.045)
        width = pm.Uniform("width", lower=0.05, upper=1.5, initval=0.75)
        dip = pm.Uniform("dip", lower=5.0, upper=90.0, initval=80)

        # use a CustomDist with a custom logp function
        likelihood = pm.CustomDist(
            "likelihood", slip, width, dip, sigma, size=len(data), observed=data, logp=custom_dist_loglike, random=gauss_draw
        )

    # run model
    with no_grad_model:
        step = pm.Metropolis() # specify step
        # Use custom number of draws to replace the HMC based defaults
        idata_no_grad = pm.sample(draws=1000, tune=500, step=step, return_inferencedata=True)
        #idata_no_grad.extend(pm.sample_posterior_predictive(idata_no_grad, random_seed=RANDOM_SEED))

    
    # visualize model results
    plot_stats(idata_no_grad) # plots trace, posterior, and summary table

    plot_corner(idata_no_grad, burn_in=True) # plots corner plots
