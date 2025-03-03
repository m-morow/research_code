#!/usr/bin/env python

import os
import glob
import json
import shutil
import numpy as np
from cubbie.read_write_insar_utilities import isce_read_write

los = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Data/IF_longterm/los_files/geo_los.rdr'
xarray, yarray, data = isce_read_write.read_scalar_data(los)