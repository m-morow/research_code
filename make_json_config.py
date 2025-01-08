import os
from datetime import datetime
import json

experiment_dir = '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/example_isce_int/'
experiment_data_dir = experiment_dir + 'processing'
json_outfile_name = 'params'
file_key = '/geo_phase.int'

#Save subdirectories
experiment_data_folders = [f.path for f in os.scandir(experiment_data_dir) if f.is_dir()]

json_count = 0

for subfolder in experiment_data_folders:
    param_dict = {"experiment_dir": '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/example_isce_int/',
                  "work_dir": '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/IF_longterm/example_isce_int/20180105_20180117',
                  "file": 'none',
                  "extent": [-115.6, -115.2, 32.6, 33],
                  "road_shp": '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/shapefiles/IF_longterm/shapefiles/tl_2015_06_prisecroads.shp',
                  "qfaults_shp": '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/shapefiles/IF_longterm/shapefiles/sectionsALL.shp',
                  "borders": '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/shapefiles/IF_longterm/shapefiles/cb_2018_us_nation_5m.shp',
                  "creepmeters": '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/shapefiles/IF_longterm/shapefiles/creepmeters.shp',
                  "gps": '/Users/mata7085/Library/CloudStorage/OneDrive-UCB-O365/Documents/shapefiles/IF_longterm/shapefiles/unr_gps_stations_roi.shp'
                  }
    
    filepath = subfolder + file_key
    working_dir = subfolder

    if os.path.exists(filepath):
        param_dict.update({'file': filepath})
        param_dict.update({'experiment_dir': experiment_dir})
        param_dict.update({'work_dir': working_dir})
        with open(os.path.join(subfolder, json_outfile_name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(param_dict, f, ensure_ascii=False, indent=4)
        json_count += 1
    else:
        param_dict.update({'experiment_dir': experiment_dir})
        param_dict.update({'work_dir': working_dir})
        with open(os.path.join(subfolder, json_outfile_name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(param_dict, f, ensure_ascii=False, indent=4)
        json_count += 1

with open(os.path.join(experiment_dir, 'json_config_logfile.txt'), "a") as log:
    log.write("%s -- %s folders, %s json files created" % (datetime.now().strftime("%Y-%m-%d %H:%M"), len(experiment_data_folders), json_count) + "\n")