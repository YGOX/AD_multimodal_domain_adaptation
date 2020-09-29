import h5py
import numpy as np
import nibabel
import glob
import pandas as pd
from configurations.paths import paths, file_names
import os

images_folder = paths['data']['Raw_MRI_location']
# next_label_file = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data'][
#	'MRI_to_next_label_mapping'])
current_label_file = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data'][
    'MRI_to_curr_label_mapping'])

files = glob.glob(images_folder)
# next_labels_df = pd.read_csv(next_label_file)
curr_labels_df = pd.read_csv(current_label_file)

# data_shape4 = (next_labels_df.shape[0], 1, 189, 233, 197)
# data_shape4 = (curr_labels_df.shape[0], 1, 189, 233, 197)
# print data_shape, next_labels_df.columns.values
hdf5_path = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['ad_nc_mri_hdf5_file'])
columns = ['FileName', 'Phase', 'RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'DXCURREN', 'DXCHANGE', 'DIAGNOSIS_LABEL']
df_new = pd.DataFrame(columns=columns)
tags, cls_count= np.unique(np.array(curr_labels_df['DIAGNOSIS_LABEL']), return_counts=True)
label_count= {tags[0]:cls_count[0], tags[1]:cls_count[1], tags[2]:cls_count[2]}
data_shape4 = (label_count['AD']+label_count['NL'], 1, 79, 95, 79)
hdf5_file = h5py.File(hdf5_path, mode='w')

dt = h5py.special_dtype(vlen=bytes)

hdf5_file.create_dataset("RID", (label_count['AD']+label_count['NL'],), np.int16)
hdf5_file.create_dataset("FileName", (label_count['AD']+label_count['NL'],), dtype=dt)
hdf5_file.create_dataset("Image4D", data_shape4, np.float32)
hdf5_file.create_dataset("CurrLabel", (label_count['AD']+label_count['NL'],), dtype=dt)
# hdf5_file.create_dataset("NextLabel", (next_labels_df.shape[0],), dtype = dt)
id=0
for idx, row in curr_labels_df.iterrows():
    selected_idx = [indx for indx, f in enumerate(files) if row['FileName'] in files[indx]]
    # print(idx, row['FileName'], files[selected_idx[0]])
    label = curr_labels_df.loc[curr_labels_df['FileName'] == row['FileName'], 'DIAGNOSIS_LABEL'].iloc[0]
    # print(label)
    if label != 'MCI':
        img_nib = nibabel.load(files[selected_idx[0]])
        img = np.nan_to_num(img_nib.get_fdata())

        hdf5_file["RID"][id] = row['RID']
        hdf5_file["FileName"][id] = row['FileName']
        hdf5_file["Image4D"][id] = img
        hdf5_file["CurrLabel"][id] = label
        # hdf5_file["NextLabel"][idx] = row['DIAGNOSIS_LABEL']
        print(idx, hdf5_file["RID"][id], hdf5_file["FileName"][id], hdf5_file["CurrLabel"][id],
              hdf5_file["Image4D"][id].shape)
        df_temp = pd.DataFrame({
            'FileName' : [row['FileName']],
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

        #df_new = df_new[[columns[0], columns[1], columns[2], columns[3], columns[4],
        #                 columns[5], columns[6], columns[7], columns[8]]]
        df_new.to_csv(os.path.join(paths['data']['Input_to_Training_Model'],
                                   file_names['data']['mri_ad_nc_mapping']))
        id += 1

hdf5_file.close()