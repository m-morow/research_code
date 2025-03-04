#!/usr/bin/env python

import os
import glob
import json
import shutil
import numpy as np

def rename_and_copy(src_path, orig_filename, dst_path=None):
    """
    rename and copy files to same directory (default) or new directory

    params:
    -------
    src_path: path to original file

    orig_filename: original filename

    dst_path: default=None; user can also set destination of renamed file

    returns:
    -------
    target_file_path: the full path of the original file

    new_full_path: the full path of where the renamed file will go (default is the file stays in the original dir)
    """
    target_file_path = os.path.join(src_path, orig_filename)
    base, extension = os.path.splitext(target_file_path)
    base_split = base.split('/')[-2:]
    new_filename = base_split[0] + '_' + base_split[1] + extension
    if dst_path is None:
        new_full_path = str("/".join(base.split('/')[:-1]))+ "/" + new_filename
    else:
        new_full_path = os.path.join(dst_path, new_filename)
    return target_file_path, new_full_path

if __name__ == "__main__":
    creep_dates = ['20150708_20150801', '20180105_20180117', '20190629_20190711', '20190711_20190723',
                  '20190723_20190804', '20210302_20210314', '20210425_20210501','20220520_20220601',
                  '20230115_20230127', '20230208_20230220', '20230328_20230409', '20230409_20230421',
                  '20231217_20231229', '20240203_20240215', '20241105_20241117']
    
    target_files = ['geo_filt_fine.unw', 'geo_filt_fine.unw.xml']
    #target_files = ['geo_filt_fine.cor', 'geo_filt_fine.cor.xml',
    #                'geo_phase.int', 'geo_phase.int.xml',
    #                'geo_filt_phase.int', 'geo_filt_phase.int.xml']
    
    workdir = '/media/mtan/rocket/mtan/IF_longterm/processing/merged/interferograms/'

    dstdir = '/media/mtan/rocket/mtan/IF_longterm/processing/merged/creep/'

    print("""
          source directory = {}
          target files = {}
          destination directory = {}""".format(workdir, target_files, dstdir))
    user_input = input("\n Do you want to continue? (yes/no): ")

    if user_input.lower() in ["yes", "y", "YES"]:
        print("\n continuing ... ")
        for date in creep_dates:
            path = os.path.join(workdir, date)
            for file in target_files:
                target_file_path, new_full_path = rename_and_copy(path, file, dst_path=dstdir)
                #print(target_file_path, new_full_path) #test before running
                #print("-----") 
                shutil.copy(target_file_path, new_full_path)
    else:
        print('\n exiting ...')