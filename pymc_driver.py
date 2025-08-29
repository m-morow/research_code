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
import pprint

import pytensor
import pytensor.tensor as pt
import pymc as pm
from pytensor.graph import Apply, Op

from pymc_espy_utils import get_los, read_intxt, do_update, read_json
import pymc_visualize

def do_okada(slip_ok, width_ok, dip_ok, m_ok, x_ok, b_ok, inputs=None):
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
    inputs = do_update(inputs_orig, slip_ok[0], width_ok[0], dip_ok[0])

    #x = [pt.lon for pt in disp_points]

    model_disp_points_ok = run_dc3d.compute_ll_def(inputs, params_in, disp_points)

    los_ok = get_los(model_disp_points_ok)
    #pprint.pprint(los_ok)

    return los_ok + (m_ok*x_ok + b_ok)

def my_loglike(slip_ll, width_ll, dip_ll, m_ll, x_ll, b_ll, sigma_ll, data_ll):
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
    for param in (slip_ll, width_ll, dip_ll, m_ll, x_ll, b_ll, sigma_ll, data_ll):
        if not isinstance(param, (float, np.ndarray)):
            raise TypeError(f"Invalid input type to loglike: {type(param)}") 
    model = do_okada(slip_ll, width_ll, dip_ll, m_ll, x_ll, b_ll)
    #return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)
    return -(0.5/sigma_ll**2)*np.sum((data_ll - model)**2) #eq. 5.1 Menke textbook

class LogLike(Op):

    def make_node(self, slip, width, dip, m, x, b, sigma, data) -> Apply:
        slip = pt.as_tensor(slip)
        width = pt.as_tensor(width)
        dip = pt.as_tensor(dip)
        m = pt.as_tensor(m)
        x = pt.as_tensor(x)
        b = pt.as_tensor(b)
        data = pt.as_tensor(data)
        sigma = pt.as_tensor(sigma)

        inputs = [slip, width, dip, m, x, b, sigma, data]
        outputs = [data.type()]

        return Apply(self, inputs, outputs)
    
    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        slip, width, dip, m, x, b, sigma, data = inputs

        loglike_eval = my_loglike(slip, width, dip, m, x, b, sigma, data)

        outputs[0][0] = np.asarray(loglike_eval)

def gauss_draw(slip, width, dip, size=None):
        return np.random.Generator.random(slip, width, dip, size = size)

def custom_dist_loglike(data_dl, slip_dl, width_dl, dip_dl, m_dl, x_dl, b_dl, sigma_dl):
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
    return loglike_op(slip_dl, width_dl, dip_dl, m_dl, x_dl, b_dl, sigma_dl, data_dl)

if __name__ == "__main__":

    #########################################################
    # really important, set globals once before running !!  #
    #########################################################
    #js = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/codes/experiment2/pymc_tests/20180105_20180117/pymc_params.json'
    js = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/codes/experiment2/pymc_tests/test3/pymc_params_synth_50disp.json'
    params = read_json(js)
    os.chdir(params['experiment_dir'])

    inputs_orig = read_intxt(params['inputs_orig'])
    params_in = PyCoulomb.configure_calc.configure_stress_calculation(params['params'])
    disp_points = io_additionals.read_disp_points(params['disp_points'])
    data = np.loadtxt(params['data'])
    x = np.loadtxt(params['disp_points'], usecols=0)

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
        #dip = pm.TruncatedNormal("dip", mu=params['dip_mu'], sigma=params['dip_sigma'], lower=params['dip_lower'], upper=params['dip_upper'], initval=params['dip_init'])
        
        # uniform priors
        slip = pm.Uniform("slip", lower=0.015, upper=0.05, initval=0.03)
        width = pm.Uniform("width", lower=0.05, upper=2.5, initval=1)
        dip = pm.Uniform("dip", lower=30, upper=90.0, initval=60)

        m = pm.Normal("slope", 0, sigma=20)
        b = pm.Normal("intercept", 0, sigma=20)

        # use a CustomDist with a custom logp function
        likelihood = pm.CustomDist(
            "likelihood", slip, width, dip, m, x, b, sigma, size=len(data), observed=data, logp=custom_dist_loglike
        )

    # run model
    with no_grad_model:
        step = pm.Metropolis() # specify step
        # Use custom number of draws to replace the HMC based defaults
        idata_no_grad = pm.sample(draws=200, tune=200, step=step, return_inferencedata=True)

    d, w, s, slope_m, b_const = pymc_visualize.set_up_okada(js, idata_no_grad)
    print("====== mean ======")
    print("dip = ", d, "width = ", w, "slip = ", s, "m = ", slope_m, "b = ", b_const)

    slope = np.zeros(50) + float(slope_m.mean())
    b_linear = np.zeros(50) + float(b_const.mean())
    los = do_okada(np.array([s]), np.array([w]), np.array([d]), slope, x, b_linear, inputs_orig)
    print("====== LOS array ======")
    pprint.pprint(los)

    deg2m = 40075*1000 * np.cos(np.deg2rad(32)) / 360 #quick conversion
    x_meters = []
    for x_prof in x:
        x_meters = np.append(x_meters, (x_prof-x[0])*deg2m)
    
    plt.plot(x_meters, los, label='pymc fit')
    plt.scatter(x_meters, data, label='data')
    plt.savefig('/Users/mata7085/Desktop/test_los.png')

    # visualize model results
    #pymc_visualize.plot_stats(idata_no_grad) # plots trace, posterior, and summary table

    #pymc_visualize.plot_corner(idata_no_grad, burn_in=False) # plots corner plots

    #pymc_visualize.set_up_okada(js, idata_no_grad)

    # save model results
    #save_model = os.path.join(params["experiment_dir"], "results")
    #save_pymc_model(idata_no_grad, save_model + "/model_109.nc")