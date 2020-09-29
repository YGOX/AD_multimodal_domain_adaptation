'''
reference to PET_CSV, matching RID and find corresponding MRI, and generate MRI.csv
'''

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from configurations.paths import paths, file_names

path = paths['data']['ADNI_study_data_original_labels']
file_name = file_names['data']['PET_to_curr_label_mapping']

img_loc = paths['data']['Raw_MRI_location']
all_imgs = glob.glob(img_loc)

columns = ['FileName', 'Phase', 'RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'DXCURREN', 'DXCHANGE', 'DIAGNOSIS_LABEL']
df_new = pd.DataFrame(columns=columns)

files = []

for indx in range(len(all_imgs)):
    fn = all_imgs[indx]
    fn = re.split(r'[/.]', fn)
    files.append(fn[-2])

# print files

file = os.path.join(path, file_name)
df = pd.read_csv(file, engine='python')

for idx, row in df.iterrows():

    pet_file_id = row['FileName']
    site_id = re.split(r'[_]', pet_file_id)[1]
    rid = re.split(r'[_]', pet_file_id)[3]
    selected_files = [s for s in files if '_' + str(rid) + '_' in s]

    if selected_files:
            df_temp = pd.DataFrame({
                'FileName' 	: selected_files[0],
                'Phase' 			: [row['Phase']],
                'RID'				: [row['RID']],
                'VISCODE'			: [row['VISCODE']],
                'VISCODE2'			: [row['VISCODE2']],
                'EXAMDATE' 			: [row['EXAMDATE']],
                'DXCURREN'			: [row['DXCURREN']],
                'DXCHANGE'			: [row['DXCHANGE'] ],
                'DIAGNOSIS_LABEL' 	: [row['DIAGNOSIS_LABEL']]
            })
            df_new = pd.concat([df_new, df_temp], ignore_index=True)

# Store data in new csv file
df_new = df_new[[columns[0], columns[1], columns[2], columns[3], columns[4],
                 columns[5], columns[6], columns[7], columns[8]]]

df_new.to_csv(os.path.join(paths['data']['Input_to_Training_Model'],
                           file_names['data']['MRI_to_curr_label_mapping']))

