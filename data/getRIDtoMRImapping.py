'''
returns dictionary of patient to MRIs mapping
'''

import h5py
import numpy as np
import pickle
import os
from configurations.paths import paths, file_names

fname = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['mri_hdf5_file'])
pkl_file = open(os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['RIDtoMRI']), 'wb')

f = h5py.File(fname, 'r')
print (f.keys())
#print type(f['RID'][0])

RIDtoMRI = dict()

rid_set = set(f['RID'])
#print(len(rid_set))
#print rid_set

for rid in rid_set:
    indx = np.where(f['RID'][:] == rid)[0]
    #print rid, indx
    RIDtoMRI[rid] = indx

pickle.dump(RIDtoMRI, pkl_file)
pkl_file.close()

f.close()