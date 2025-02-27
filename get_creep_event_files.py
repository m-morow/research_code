#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
from cubbie.read_write_insar_utilities import isce_read_write
from cubbie.math_tools import grid_tools
import geopandas as gp
import glob
import json
from cmcrameri import cm

plt.rcParams['axes.linewidth'] = 1

if __name__ == "__main__":