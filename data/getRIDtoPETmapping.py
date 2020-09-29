'''
returns dictionary of patient to PETs mapping
'''

import h5py
import numpy as np
import pickle
import os
from configurations.paths import paths, file_names

fname = os.path.join(paths['data']['Input_to_Training_Model'],
					 file_names['data']['pet_hdf5_file'])

pkl_file = open(os.path.join(paths['data']['Input_to_Training_Model'],file_names['data']['RIDtoPET']), 'wb')

f = h5py.File(fname, 'r')
print (f.keys())
#print type(f['RID'][0])

RIDtoPET = dict()

rid_set = set(f['RID'])
#print(len(rid_set))
#print rid_set

for rid in rid_set:

    indx = np.where(f['RID'][:] == rid)[0]
    #print rid, indx
    RIDtoPET[rid] = indx

pickle.dump(RIDtoPET, pkl_file)
pkl_file.close()

f.close()