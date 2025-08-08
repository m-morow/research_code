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

import pytensor
import pytensor.tensor as pt
import pymc as pm
from pymc import HalfCauchy, Model, Normal, sample
from pytensor.graph import Apply, Op

from pymc_espy_utils import get_los, read_intxt, do_update

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
        cnr = corner.corner(idata, divergences=True)
        cnr.suptitle("{} draws, {} chains, {} samples per chain removed".format(draws, chains, tune))
        plt.close()
    else:
        cnr = corner.corner(pymc_model, divergences=True)
        cnr.suptitle("{} draws, {} chains, 0 samples per chain removed".format(draws, chains))
        plt.close()
    return cnr

#def plot_profile(params, extent_small=False):
    