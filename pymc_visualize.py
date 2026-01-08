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

from pymc_espy_utils import get_los, read_intxt, do_update, read_json, uncertainties
import pymc_driver

def plot_posterior(pymc_model, outfile=None):
    fig, ax = plt.subplots()
    ax = az.plot_posterior(pymc_model)
    if outfile:
        plt.savefig(outfile)
        print('saved posterior plot to {}'.format(outfile))

def plot_corner(pymc_model, burn_in=False, outfile=None):
    chains = int(pymc_model.posterior.dims['chain'])
    draws = int(pymc_model.posterior.dims['draw'])
    tune = int(pymc_model.posterior.attrs['tuning_steps'])

    if burn_in:
        idata = pymc_model.sel(draw=slice(tune, None))
        #means = idata.mean()
        cnr = corner.corner(idata, divergences=True)
        cnr.suptitle("{} draws, {} chains, {} samples per chain removed".format(draws, chains, tune))
        #cnr = corner.overplot_lines(cnr, [means.posterior["slip"], means.posterior["width"], means.posterior["dip"]], color="#71A8C4")
    else:
        #means = pymc_model.mean()
        cnr = corner.corner(pymc_model, divergences=True)
        cnr.suptitle("{} draws, {} chains, 0 samples per chain removed".format(draws, chains))
        #cnr = corner.overplot_lines(cnr, [means.posterior["slip"], means.posterior["width"], means.posterior["dip"]], color="#71A8C4")
    if outfile:
        plt.savefig(outfile)
        print('saved corner plot to {}'.format(outfile))   

def set_up_okada(pymc_model, data):
    print(az.summary(pymc_model))

    d_mean = np.mean(az.convert_to_dataset(pymc_model)['dip'])
    w_mean = np.mean(az.convert_to_dataset(pymc_model)['width'])
    s_mean = np.mean(az.convert_to_dataset(pymc_model)['slip'])

    slope_mean = np.mean(az.convert_to_dataset(pymc_model)['slope'])
    b_mean = np.mean(az.convert_to_dataset(pymc_model)['intercept'])

    slope = np.zeros(len(data)) + float(slope_m.mean())
    b_linear = np.zeros(len(data)) + float(b_const.mean())

    #slope, int, slip, width, dip
    #means = np.array(az.summary(pymc_model)['mean'])[2:]
    sds = np.array(az.summary(pymc_model)['mean'])[2:]

    text1 = r'{:<6}'.format('$slip$') + \
        r'$=\, {} \pm {}$'.format(means[0], sds[0]) + r'{}'.format(' cm')
    text2 = r'{:<6}'.format('$width$') + \
        r'$=\, {} \pm {}$'.format(means[1], sds[1]) + r'{}'.format(' m')
    text3 = r'{:<6}'.format('$dip$') + \
        r'$=\, {} \pm {}$'.format(means[2], sds[2]) + r'{}'.format(' deg')
    text = text1 + '\n' + text2 + '\n' + text3

    print("====== mean ======")
    print("dip = ", d_mean, "width = ", w_mean, "slip = ", s_mean, "m = ", slope_mean, "b = ", b_mean)

    return d_mean, w_mean, s_mean, slope, b_linear, sds, text

def plot_los_model(los, data, x, outfile):
    deg2m = 40075*1000 * np.cos(np.deg2rad(32)) / 360 #quick conversion
    x_meters = []
    for x_prof in x:
        x_meters = np.append(x_meters, (x_prof-x[0])*deg2m)
    
    plt.xlabel('distance along profile [m]')
    plt.ylabel('LOS displacement [cm]')
    plt.plot(x_meters, los*100, label='pymc fit')
    plt.scatter(x_meters, data*100, label='data')
    plt.savefig(outfile)
    print('saved LOS model to {}'.format(outfile))
    return x_meters

def test_plot_los_model(los, data, x, textbox, outfile):
    deg2m = 40075*1000* np.cos(np.deg2rad(32)) / 360 #quick conversion
    x_m = []
    for x_prof in x:
        x_m = np.append(x_m, (x_prof-x[0])*deg2m)
    
    fig, ax = plt.subplots()
    plt.xlabel('distance along profile [km]')
    plt.ylabel('LOS displacement [cm]')
    plt.scatter(x_m, data*100, label='data')
    plt.plot(x_m, los*100, label='model')
    #plt.legend(loc='best')
    if textbox:
        #_, _, text = uncertainties(stats)
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    if outfile:
        plt.savefig(outfile)
    print('saved LOS model to {}'.format(outfile))
    return x_km