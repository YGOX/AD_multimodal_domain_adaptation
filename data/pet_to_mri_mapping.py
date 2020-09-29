'''
to find corresponding MRI according to PET, generate a list of MRI file names, then to select the MRI from the parent folder
'''

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from configurations.paths import paths, file_names

path = paths['data']['ADNI_study_data_original_labels']
file_name = file_names['data']['MRI_to_curr_label_mapping']
out_name = file_names['data']['MRI_list']

img_loc = paths['data']['Raw_MRI_location']
all_imgs = glob.glob(img_loc)

files = []

for indx in range(len(all_imgs)):
    fn = all_imgs[indx]
    fn = re.split(r'[/.]', fn)
    files.append(fn[-2])
# print files

file = os.path.join(path, file_name)
df = pd.read_csv(file, engine='python')
mri_files= []

for idx, row in df.iterrows():

    pet_files = row['FileName']
    pet_dates_str = re.split(r'[_]', pet_files)[-4]
    pet_file_dates = datetime.strptime(pet_dates_str[:8], '%Y%m%d')

    rid = row['RID']
    rid = format(rid, '04d')
    site_id= re.split(r'[_]', pet_files)[1]
    selected_files = [s for s in files if site_id+ '_S_' + str(rid) in s]

    if selected_files:
        mri_dates_str = [re.split(r'[_]', fn)[-3] for fn in selected_files]
        mri_file_dates = [datetime.strptime(d[:8], '%Y%m%d') for d in mri_dates_str]
        diff = np.array([abs(d - pet_file_dates).days for d in mri_file_dates])
        mri_files.append(selected_files[np.argmin(diff)]+'.nii')
        print(selected_files[np.argmin(diff)])
        '''
        print idx, rid, examdate
        print 'diff :', np.argmin(diff), diff[np.argmin(diff)]
        print selected_files
        #print matched_date
        #pdb.set_trace()
        '''
with open(os.path.join(path, out_name), 'w') as f:
    for item in mri_files:
        f.write("%s\n" % item)


