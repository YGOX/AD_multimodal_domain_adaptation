import h5py
import numpy as np
import nibabel
import glob
import pandas as pd
from configurations.paths import paths, file_names
import os
from numpy import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

images_folder = paths['data']['Raw_MRI_location']
# next_label_file = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data'][
#	'MRI_to_next_label_mapping'])
current_label_file = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data'][
    'MRI_to_curr_label_mapping'])

files = glob.glob(images_folder)
# next_labels_df = pd.read_csv(next_label_file)
curr_labels_df = pd.read_csv(current_label_file)

# data_shape4 = (next_labels_df.shape[0], 1, 189, 233, 197)
data_shape4 = (curr_labels_df.shape[0], 1, 120, 132, 96)#182, 218, 182)
#data_shape4 = (curr_labels_df.shape[0], 1, 79, 95, 79)
# print data_shape, next_labels_df.columns.values
hdf5_path = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['mri_hdf5_file'])

hdf5_file = h5py.File(hdf5_path, mode='w')

dt = h5py.special_dtype(vlen=bytes)

hdf5_file.create_dataset("RID", (curr_labels_df.shape[0],), np.int16)
hdf5_file.create_dataset("FileName", (curr_labels_df.shape[0],), dtype=dt)
hdf5_file.create_dataset("Image4D", data_shape4, np.float32)
hdf5_file.create_dataset("CurrLabel", (curr_labels_df.shape[0],), dtype=dt)
# hdf5_file.create_dataset("NextLabel", (next_labels_df.shape[0],), dtype = dt)


for idx, row in curr_labels_df.iterrows():
    selected_idx = [indx for indx,f in enumerate(files) if row['FileName'] in files[indx]]
    # print(idx, row['FileName'], files[selected_idx[0]])
    label = curr_labels_df.loc[curr_labels_df['FileName'] == row['FileName'], 'DIAGNOSIS_LABEL'].iloc[0]
    # print(label)
    img_nib = nibabel.load(files[selected_idx[0]])
    
    img = np.nan_to_num(img_nib.get_fdata())
    print(img.shape)
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    #img= np.clip(img, 0.1, 0.9)
    hdf5_file["RID"][idx] = row['RID']
    hdf5_file["FileName"][idx] = row['FileName']
    hdf5_file["Image4D"][idx] = img
    hdf5_file["CurrLabel"][idx] = label
    # hdf5_file["NextLabel"][idx] = row['DIAGNOSIS_LABEL']
    print(idx, hdf5_file["RID"][idx], hdf5_file["FileName"][idx], hdf5_file["CurrLabel"][idx],
          hdf5_file["Image4D"][idx].shape)

hdf5_file.close()