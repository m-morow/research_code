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
import corner
import numpy as np
import os

import pytensor
import pytensor.tensor as pt
import pymc as pm
from pymc import HalfCauchy, Model, Normal, sample
from pytensor.graph import Apply, Op

from pymc_espy_utils import get_los, read_intxt, do_update, read_json
from pymc_driver import do_okada

def plot_stats(pymc_model, round=3):
    fig, ax = plt.subplots(3)
    fig.suptitle('model overview')
    ax[0] = az.summary(pymc_model, round_to=round)
    ax[1] = az.plot_trace(pymc_model)
    ax[2] = az.plot_posterior(pymc_model)
    return fig

def plot_corner(pymc_model, burn_in=False):
    chains = int(pymc_model.posterior.dims['chain'])
    draws = int(pymc_model.posterior.dims['draw'])
    tune = int(pymc_model.posterior.attrs['tuning_steps'])

    if burn_in:
        idata = pymc_model.sel(draw=slice(tune, None))
        means = idata.mean()
        cnr = corner.corner(idata, divergences=True)
        cnr.suptitle("{} draws, {} chains, {} samples per chain removed".format(draws, chains, tune))
        cnr = corner.overplot_lines(cnr, [means.posterior["slip"], means.posterior["width"], means.posterior["dip"]], color="#71A8C4")
    else:
        means = pymc_model.mean()
        cnr = corner.corner(pymc_model, divergences=True)
        cnr.suptitle("{} draws, {} chains, 0 samples per chain removed".format(draws, chains))
        cnr = corner.overplot_lines(cnr, [means.posterior["slip"], means.posterior["width"], means.posterior["dip"]], color="#71A8C4")
    return cnr    

def plot_okada(json_params, pymc_model, stdev_hi, stdev_lo):
    params = read_json(json_params)
    os.chdir(params['experiment_dir'])

    inputs_orig = read_intxt(params['inputs_orig'])
    params_in = PyCoulomb.configure_calc.configure_stress_calculation(params['params'])
    disp_points = io_additionals.read_disp_points(params['disp_points'])

    sigma = params['sigma']
    dip_mean = np.mean(az.convert_to_dataset(pymc_model)['dip'])
    width_mean = np.mean(az.convert_to_dataset(pymc_model)['width'])
    slip_mean = np.mean(az.convert_to_dataset(pymc_model)['slip'])

    los = do_okada(np.array([slip_mean]), np.array([width_mean]), np.array([dip_mean]), m=1, x=disp_points, b=0)

    lonpt = np.loadtxt('/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/codes/experiment2/pymc_tests/20180105_20180117/disp_pt_109.txt', usecols=0)
    data = np.loadtxt('/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/codes/experiment2/pymc_tests/20180105_20180117/los_data_109.txt', usecols=0)
    plt.plot(lonpt, los, label='pymc fit', c='red')
    plt.scatter(lonpt, data, c='k')
    plt.legend(loc='best')
    plt.show()