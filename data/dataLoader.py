'''
pyTorch custom dataloader
'''
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from configurations.paths import paths, file_names
import os
import pickle
from random import shuffle
from configurations.modelConfig import params, num_classes, name_classes

from data.dataAugmentation import random_transform
import pandas as pd
from data.splitDataset import getIndicesTrainValidTest
#import torchvision.transforms as transforms

from scipy.ndimage import zoom


def intersection(a,b):
    return list(set(a) & set(b))


class HDF5loader():
    def __init__(self, filename, trans=None, train_indices=None):
        f = h5py.File(filename, 'r',  libver='latest', swmr=True)
        self.img_f = f['Image4D']
        self.trans = trans
        
        if 'mri' in filename:
            self.modality = 'mri'
        elif 'pet' in filename:
            self.modality = 'pet'
        
        
        self.train_indices = train_indices
        
        
        
        if len(name_classes) == 2 and name_classes[1] in ['AD','MCI']:
            ignore = np.array([0 if x.decode('utf-8') in name_classes else 1 for x in f[params['train']['timestamp']]],dtype = 'bool')
            self.label = [0 if x.decode('utf-8')  == name_classes[0] else (1 if x.decode('utf-8')  == name_classes[1] else 2) for x in   f[params['train']['timestamp']]]
            self.ind_to_use = np.arange(ignore.size,dtype='int')[~ignore]
            _, cls_count= np.unique(np.array(self.label)[np.array(intersection(train_indices,self.ind_to_use))], return_counts=True)
        
        elif len(name_classes) == 2:
            self.label = [0 if x.decode('utf-8')  == name_classes[0] else 1 for x in f[params['train']['timestamp']]]
            self.ind_to_use = np.arange(len(self.label))
        elif len(name_classes) == 3:
            self.label = [0 if x.decode('utf-8')  == 'CN' else (2 if x.decode('utf-8')  == 'AD' else 1) for x in f[params['train']['timestamp']]]
            self.ind_to_use = np.arange(len(self.label))
        
        print('Using',self.ind_to_use.size,'point of classes',name_classes,'out of total',len(self.label))
        
        _, cls_count= np.unique(np.array(self.label)[np.array(intersection(train_indices,self.ind_to_use))], return_counts=True)

        cls_weights= 1/cls_count
        self.cls_weights= (cls_weights/cls_weights.sum()).tolist()
        print('Weights per class:',self.cls_weights)
        #self.label = [0 if x == 'NL' else (2 if x == 'AD' else 1) for x in f['NextLabel']]
        #self.label = [0 if x == 'NL' else 1 for x in f['CurrLabel']] #for current label    #for binary
        # classification on current label
        #self.label = [0 if x == 'NL' else 1 for x in f['NextLabel']]    #for binary classification on next label

    def __getitem__(self, index):
        img = self.img_f[index]
        label = self.label[index]
        #print('original', img.shape) #(1, 189, 233, 197)
        
        # for coronal view (channels, depth, 0, 1)
        # img = np.moveaxis(img, 1, 3)
        #print('1. ', img.shape)    #(1, 233, 197, 189)
        
        # drop 10 slices on either side since they are mostly black
        # reshape to (depth, 0, 1, channels) for data augmentation
        
        img = np.moveaxis(img, 0, 3)
        
        
        
        # random transformation
        if self.trans is not None and index in self.train_indices:
            img = random_transform(img, **self.trans)
        # reshape back to (channels, depth, 0, 1)
        img = np.moveaxis(img, 3, 0)
        #print('3. ', img.shape)    #(1, 213, 197, 189)
        #normalizing image - Gaussian normalization per volume
        if np.std(img) != 0:  # to account for black images
            mean = np.mean(img)
            std = np.std(img)
            img = 1.0 * (img - mean) / std
        
        img = img.astype(float)
        img = torch.from_numpy(img).float()
        
        label = torch.LongTensor([label])
        return (img ,label, index)
        
    def __len__(self):
        return self.img_f.shape[0]

    def get_cls_weights(self):
        return self.cls_weights

    
def dataLoader(hdf5_file, train_indices, valid_indices, test_indices, trans=None, batch_size = None):
    if batch_size is None:
        batch_size = params['train']['batch_size']
    if params['train']['modality'] == 'multi':
        batch_size = int(batch_size/2)
    
    print('Batch size (per modality):',batch_size)
    
    num_workers = 0
    pin_memory = False
    
    '''

    train_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
                                                   file_names['data']['Train_set_indices']), 'r'))
    shuffle(train_indices)

    valid_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
                                                   file_names['data']['Valid_set_indices']), 'r'))

    test_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
                                                   file_names['data']['Test_set_indices']), 'r'))
    '''
    #train_indices, valid_indices, test_indices = getIndicesTrainValidTest()
    #shuffle(train_indices)

    data = HDF5loader(hdf5_file, trans, train_indices=train_indices)
    sizes = (len(intersection(train_indices,data.ind_to_use)), len(intersection(valid_indices,data.ind_to_use)), len(intersection(test_indices,data.ind_to_use)))
    train_sampler = SubsetRandomSampler(intersection(train_indices,data.ind_to_use))
    valid_sampler = SubsetRandomSampler(intersection(valid_indices,data.ind_to_use))
    test_sampler =  SubsetRandomSampler(intersection(test_indices,data.ind_to_use))

    cls_weights = data.get_cls_weights()
    
    # idea: simply take all indices from PET, and divide MRI over same RID's
    train_loader = DataLoader(data, batch_size, sampler=train_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(data, batch_size, sampler=valid_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(data, batch_size=1, sampler=test_sampler,
                         num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader, test_loader, cls_weights, sizes)
    


def run_test_():

    from configurations.modelConfig import data_aug
    max_epochs = 10
    
    datafile = os.path.join(paths['data']['hdf5_path'], file_names['data']['hdf5_file'])
    train_loader, valid_loader, test_loader = dataLoader(datafile, trans=data_aug)
    
    from tqdm import tqdm
    
    for ep in range(max_epochs):
        print('Epoch ' + str(ep) + ' out of ' + str(max_epochs))
        
        pbt = tqdm(total=len(train_loader))
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            #print('batch ' + str(batch_idx) + ' out of ' + str(len(train_loader)))
            pbt.update(1)
        pbt.close()
    
    
def run_tests():
    n_gpus = 1
    
    max_epochs = 10
    
    data = HDF5loader(os.path.join(paths['data']['hdf5_path'], file_names['data']['hdf5_file']))

    train_sampler = SubsetRandomSampler([0, 1, 2])
    valid_sampler = SubsetRandomSampler([3, 4, 5])

    train_iter = DataLoader(data, batch_size=1*n_gpus, sampler=train_sampler, num_workers=8)
    valid_iter = DataLoader(data, batch_size=1*n_gpus, sampler=valid_sampler, num_workers=8)

    for ep in range(max_epochs):
        print('Epoch ' + str(ep) + ' out of ' + str(max_epochs))

        print('TRAIN:')
        for batch_idx, data_ in enumerate(train_iter):
            batch_x, batch_y = data_
            print('batch ' + str(batch_idx) + str(batch_y) + ' out of ' + str(len(train_iter)))

        
        
        print('VALID:')
        for batch_idx, data_ in enumerate(valid_iter):
            batch_x, batch_y = data_
            print('batch ' + str(batch_idx) + str(batch_y) + ' out of ' + str(len(valid_iter)))
        
        
#run_tests()

#run_test_()
