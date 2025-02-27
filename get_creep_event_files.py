#!/usr/bin/env python

import os
import glob
import json
import shutil
import numpy as np

if __name__ == "__main__":
    #creep_file = '/media/mtan/rocket/mtan/IF_longterm/processing/merged/params_largerswath.json'
    #f = open(creep_file)
    #creep_events = json.load(f)
    #f.close()
    creep_dates = ['20150708_20150801', '20180105_20180117', '20190629_20190711', '20190711_20190723',
                  '20190723_20190804', '20210302_20210314', '20210425_20210501','20220520_20220601',
                  '20230115_20230127', '20230208_20230220', '20230328_20230409', '20230409_20230421',
                  '20231217_20231229', '20240203_20240215', '20241105_20241117']
    
    workdir = '/media/mtan/rocket/mtan/IF_longterm/processing/merged/interferograms/'

    creep_events_path = []

    for i in creep_dates:
        creep_events_path = np.append(creep_events_path, workdir + i)
        
    #creep_events = ['/media/mtan/rocket/mtan/IF_longterm/processing/merged/params_largerswath.json']

    for file in creep_events_path:
        base, extension = os.path.splitext(file)
        base_split = base.split('/')[-2:]
        new_filename = base_split[0] + '_' + base_split[1] + extension
        full_path = str("/".join(base.split('/')[:-2]))+ "/" + new_filename
        #shutil.copy(file, full_path)