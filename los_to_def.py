#!/usr/bin/env python

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cubbie.read_write_insar_utilities import isce_read_write
from Tectonic_Utils.read_write import netcdf_read_write
from Tectonic_Utils.geodesy import insar_vector_functions
from cmcrameri import cm

def enu2los(grd_e, grd_n, grd_u, flight_angle, incidence_angle, save=False, outpath=None):
    """
    Convert E, N, U .grd files to LOS using flight angle and incidence angle.

    Parameters:
    --------
    grd_e
    grd_n
    grd_u
    flight_angle: flight angle of satellite, accepts degree or radians, assuming angle in
                  degree is always larger than 1
    incidence_angle: average incidence angle of satellite, accepts degree or radians, assuming
                     angle in degree is always larger than 1
    save: default False, set if you want to save LOS xarray to .grd file
    outpath: default None, set if save is True

    Returns:
    --------
    los: xarray object
    """
    if flight_angle > 1:
        flight_angle = np.deg2rad(flight_angle)
        incidence_angle = np.deg2rad(incidence_angle)
    
    n = xr.open_dataset(grd_n)
    e = xr.open_dataset(grd_e)
    u = xr.open_dataset(grd_u)

    los = insar_vector_functions.def3D_into_LOS(e, n, u, flight_angle, incidence_angle)

    e.close()
    n.close()
    u.close()

    if save:
        netcdf_read_write.produce_output_netcdf(los['x'], los['y'], los['z'], 
                                                zunits='los_m',                                 
                                                netcdfname=outpath)

    return los