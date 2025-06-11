#!/usr/bin/env python

from elastic_stresses_py import PyCoulomb
from elastic_stresses_py.PyCoulomb import coulomb_collections as cc
from elastic_stresses_py.PyCoulomb.fault_slip_triangle import triangle_okada
from elastic_stresses_py.PyCoulomb.point_source_object import point_sources
from elastic_stresses_py.PyCoulomb.disp_points_object.disp_points_object import Displacement_points
from Tectonic_Utils.geodesy import fault_vector_functions
from elastic_stresses_py.PyCoulomb import utilities, io_additionals, run_dc3d, run_mogi, conversion_math
from elastic_stresses_py.PyCoulomb.inputs_object import io_intxt

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import pytensor
import pytensor.tensor as pt

import pymc as pm

from pymc import HalfCauchy, Model, Normal, sample
from pytensor.graph import Apply, Op

import pymc_espy_utils

def do_okada(params, slip, depth, dip):
    inputs = pymc_espy_utils.do_update(inputs_orig, slip[0], depth[0], dip[0])

    model_disp_points = run_dc3d.compute_ll_def(inputs, params, disp_points)

    MyOutObject = cc.Out_object(x=[], y=[], x2d=[], y2d=[], u_disp=[], v_disp=[], w_disp=[],
                                strains=[], model_disp_points=model_disp_points,
                                zerolon=inputs.zerolon, zerolat=inputs.zerolat,
                                source_object=inputs.source_object, receiver_object=inputs.receiver_object,
                                receiver_normal=[], receiver_shear=[],
                                receiver_coulomb=[], receiver_profile=[])
    
    dU = pymc_espy_utils.get_dU(MyOutObject.model_disp_points)

    return dU

def my_loglike(slip, depth, dip, sigma, x, data):
    for param in (slip, depth, dip, sigma, x, data):
        if not isinstance(param, (float, np.ndarray)):
            raise TypeError(f"Invalid input type to loglike: {type(param)}") 
    model = do_okada(params, slip, depth, dip)
    return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)

class LogLike(Op):

    def make_node(self, slip, depth, dip, sigma, x, data):
        slip = pt.as_tensor(slip)
        depth = pt.as_tensor(depth)
        dip = pt.as_tensor(dip)
        sigma = pt.as_tensor(sigma)
        x = pt.as_tensor(x)
        data = pt.as_tensor(data)

        inputs = [slip, depth, dip, sigma, x, data]
        #outputs = [pt.vector()]
        outputs = [data.type()]

        return Apply(self, inputs, outputs)
    
    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        slip, depth, dip, sigma, x, data = inputs

        loglike_eval = my_loglike(slip, depth, dip, sigma, x, data)

        outputs[0][0] = np.asarray(loglike_eval)