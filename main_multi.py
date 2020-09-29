import os
import argparse

from configurations.paths import paths, file_names
from configurations.modelConfig import layer_config, params, data_aug, name_classes

from data.dataLoader import dataLoader, run_test_, dataLoader
from data.splitDataset import getIndicesTrainValidTest

from models.vae import VAE, Adversary
from train_ad import Trainer

import time
import torch
import numpy as np

from random import shuffle
#torch.backends.cudnn.enabled = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type = str, default = None )
    parser.add_argument('--modality', type = str, default = None )
    parser.add_argument('--advers', type = bool, default = None )
    parser.add_argument('--batch_sz', type = int, default = None )
    args = parser.parse_args()


    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = params['train']['GPU']
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    
        
    if args.modality is None:
        modality = params['train']['modality']
    else:
        modality = args.modality

    if args.advers is None:
        use_adversary = params['model']['advers_use'] and modality == 'multi'
    else:
        use_adversary = args.advers and modality=='multi'

    print('gpu:', args.gpu, 'modality', modality, 'adversary', use_adversary)    
    
    num_classes = len(name_classes)

    torch.multiprocessing.set_sharing_strategy('file_system')
    # create the experiment dirs
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_folder = paths['output']['vae_base_folder']
    expt_folder = base_folder + params['train']['experiment_id'] + timestr
    print('=========== Folder',expt_folder,'=============')
    
    train_indices_mri, valid_indices_mri, test_indices_mri = getIndicesTrainValidTest(modality = 'mri')
    train_indices_pet, valid_indices_pet, test_indices_pet = getIndicesTrainValidTest(modality = 'pet')
    shuffle(train_indices_mri)
    shuffle(train_indices_pet)
    datafile_mri = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['mri_hdf5_file'])
    datafile_pet = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['pet_hdf5_file'])
    
    
    
    print('Classes:', name_classes)
    print('Augmentation settings:', data_aug)
    print('Params', params)
    if not os.path.exists(expt_folder):
        os.mkdir(expt_folder)
        
    print('Run : {}\n'.format(timestr))

    # create an instance of the model\
    model = VAE()
    if use_adversary and modality == 'multi':
        adversary = Adversary()
    else:
        adversary = None

    # count model parameters
    print('Paramater Count :', sum(p.numel() for p in model.parameters()))
    if use_adversary: 
        print('Adversary paramater Count :', sum(p.numel() for p in adversary.parameters()))
    
    
    # create data generator
        
    train_loader_mri, valid_loader_mri, test_loader_mri, cls_weights_mri, setsizes_mri = \
            dataLoader(datafile_mri, train_indices_mri, valid_indices_mri, test_indices_mri, trans=data_aug, batch_size = args.batch_sz)
    train_loader_pet, valid_loader_pet, test_loader_pet, cls_weights_pet, setsizes_pet = \
            dataLoader(datafile_pet, train_indices_pet, valid_indices_pet, test_indices_pet, trans=data_aug, batch_size = args.batch_sz)
    
    if modality == 'multi' or modality == 'pet':
        train_loader_s, valid_loader_s, test_loader_s, cls_weights_s, setsizes_s = \
            train_loader_mri, valid_loader_mri, test_loader_mri, cls_weights_mri, setsizes_mri
        train_loader_t, valid_loader_t, test_loader_t, cls_weights_t, setsizes_t = \
            train_loader_pet, valid_loader_pet, test_loader_pet, cls_weights_pet, setsizes_pet
    elif modality == 'mri':
        train_loader_t, valid_loader_t, test_loader_t, cls_weights_t, setsizes_t = \
            train_loader_mri, valid_loader_mri, test_loader_mri, cls_weights_mri, setsizes_mri
        train_loader_s, valid_loader_s, test_loader_s, cls_weights_s, setsizes_s = \
            train_loader_pet, valid_loader_pet, test_loader_pet, cls_weights_pet, setsizes_pet
    else:
        raise ValueError('not a valid modality chosen')
    
    setsizes = np.array([setsizes_s, setsizes_t])
    cls_weights = np.array([cls_weights_s, cls_weights_t])
    # create trainer and pass all required components
    if modality == 'multi':
        trainer = Trainer(model, train_loader_s, valid_loader_s, train_loader_t, valid_loader_t, expt_folder, setsizes = setsizes, weights = cls_weights, adversary=adversary)
    else:
        trainer = Trainer(model, None, None, train_loader_t, valid_loader_t, expt_folder, setsizes = setsizes, weights = cls_weights, adversary=adversary)
    # train model
    try:
        trainer.train()
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)
        print('######## Exciting gracefully ########\n Quit training at step ',trainer.curr_epoch,'. Results so far:')

    if modality =='multi':
        modalities = ['MRI','PET']
    elif modality =='mri':
        modalities = ['_','MRI']
    elif modality == 'pet':
        modalities = ['_','PET']
    else:
        modalities = ['MRI','PET']
    
    for m_index, modname in enumerate(modalities):        
        max_index = np.argmax(np.array(trainer.valid_bal_accuracy)[:,m_index])
        print(max_index)
        print('Max',modname,'validation bal acc:',trainer.valid_bal_accuracy[max_index,m_index],' at step', max_index )
        print('Corresponding train:   ',trainer.train_bal_accuracy[max_index][m_index])


        if len(name_classes)==3:
            classifications = ['MCI vs AD','CN vs AD', 'CN vs MCI']
            for c in range(3):
                print('Max validation for',classifications[c])
                maxindex = np.argmax(np.array(trainer.valid_bal_acc_per_class)[:,m_index,c])
                print(trainer.cm_per_class[maxindex,m_index, c])
                print('Balanced acc:',trainer.valid_bal_acc_per_class[maxindex,m_index, c])
        
    
    
    # test model
    print('TESTING TARGET')
    trainer.test(test_loader_t, modality = 1)
    if trainer.multi: 
        print('TESTING SOURCE')
        trainer.test(test_loader_s, modality = 0)
    
    
    
    print('BEST PERFORMANCE TESTER')
    pretrained_dict = torch.load(os.path.join(trainer.expt_folder, 'latest_model.pkl'))
    
    if trainer.multi: 
        print('TESTING SOURCE')
        trainer.test(test_loader_s, modality = 0)
    
    print('TESTING TARGET')
    # load the new state dict
    trainer.model.load_state_dict(pretrained_dict)
    trainer.test(test_loader_t, modality = 1)
    
    
    
    
