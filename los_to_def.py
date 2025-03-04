#!/usr/bin/env python

import os
import glob
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from cubbie.read_write_insar_utilities import isce_read_write

los = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Data/IF_longterm/los_files/geo_los.rdr'
los_binary = np.fromfile(los)

los_binary_matrix = np.reshape(los_binary, (9000, 10000))

fig, ax = plt.subplots()
ax.imshow(los_binary_matrix)
plt.show()