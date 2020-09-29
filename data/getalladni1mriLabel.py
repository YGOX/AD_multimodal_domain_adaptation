'''
This code finds labels for each of the 3D MRI and stores the MRI filenames and corresponding labels in a csv file
'''

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from configurations.paths import paths, file_names

path = paths['data']['ADNI_study_data_original_labels']
file_name = file_names['data']['ADNI1_MRI_data_original_labels']

img_loc = paths['data']['Raw_MRI_location']
all_imgs = glob.glob(img_loc)

columns = ['FileName', 'ImageID', 'SubjectID', 'RID', 'DIAGNOSIS_LABEL']
df_new = pd.DataFrame(columns=columns)

files = []

for indx in range(len(all_imgs)):
    fn = all_imgs[indx]
    fn = re.split(r'[/.]', fn)
    if fn[-1]=='gz':
        files.append(fn[-3])
    else:
        files.append(fn[-2])
    print(files[-1])
# print files

file = os.path.join(path, file_name)
df = pd.read_csv(file, engine='python')
list_img= df['Image Data ID'].tolist()

for idx, row in df.iterrows():

    id = row['Subject']
    selected_files = [s for s in files if id in s]

    if selected_files:
        img_id = [re.split(r'[_]', fn)[-1] for fn in selected_files]
        '''
        print idx, rid, examdate
        print 'diff :', np.argmin(diff), diff[np.argmin(diff)]
        print selected_files
        #print matched_date
        #pdb.set_trace()
        '''
        for i, iid in zip(range(len(selected_files)), img_id):
            row_id= list_img.index(int(iid[1:]))
            sid= df.iloc[[row_id]]['Subject'].item()
            dl= df.iloc[[row_id]]['Group'].item()
            rid= int(re.split(r'[_]', sid)[-1])
            # Create new label field
            df_temp = pd.DataFrame({
                'FileName': selected_files[i],
                'ImageID': [iid],
                'SubjectID': [sid],
                'RID': [rid],
                'DIAGNOSIS_LABEL': [dl]})
            df_new = pd.concat([df_new, df_temp])

            # drop the filename from list
            files.remove(selected_files[i])

# Store data in new csv file
df_new = df_new[[columns[0], columns[1], columns[2], columns[3], columns[4]]]

df_new.to_csv(os.path.join(paths['data']['Input_to_Training_Model'],
                           file_names['data']['MRI_to_curr_label_mapping']))

