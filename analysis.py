import os
import argparse

from configurations.paths import paths, file_names
from configurations.modelConfig import layer_config, params, data_aug, name_classes

from data.dataLoader import dataLoader, run_test_, dataLoader
from data.splitDataset import getIndicesTrainValidTest
from utils.visualizations import visualizeFilters
from models.vae import VAE, Adversary
from train_ad import Trainer

import time
import torch
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.transform import resize, rotate

folder_name = 'nohup209'
use_adversary = False
modality = 'multi'
tasks=[False, False, True]
perspect = 0
#torch.manual_seed(0)


model = VAE()
#torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = params['train']['GPU']

expt_name = '../data/vae_outputs'

prefixed = [filename for filename in os.listdir(expt_name) if filename.startswith(folder_name)]

if len(prefixed)>1:
    print(prefixed)
    raise ValueError
else:
    expt_folder = os.path.join(expt_name, prefixed[0])


if use_adversary and modality == 'multi':
    adversary = Adversary()
else:
    adversary = None


#torch.manual_seed(0)    
train_indices_mri, valid_indices_mri, test_indices_mri = getIndicesTrainValidTest(modality = 'mri')
#torch.manual_seed(0)

train_indices_pet, valid_indices_pet, test_indices_pet = getIndicesTrainValidTest(modality = 'pet')
datafile_mri = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['mri_hdf5_file'])
datafile_pet = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['pet_hdf5_file'])


#torch.manual_seed(0)

train_loader_mri, valid_loader_mri, test_loader_mri, cls_weights_mri, setsizes_mri = \
            dataLoader(datafile_mri, train_indices_mri, valid_indices_mri, test_indices_mri, trans=data_aug, batch_size = None)

#torch.manual_seed(0)

train_loader_pet, valid_loader_pet, test_loader_pet, cls_weights_pet, setsizes_pet = \
            dataLoader(datafile_pet, train_indices_pet, valid_indices_pet, test_indices_pet, trans=data_aug, batch_size = None)
    
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

trainer = Trainer(model, train_loader_s, valid_loader_s, train_loader_t, valid_loader_t, expt_folder, setsizes = setsizes, weights = cls_weights, adversary=adversary)

pretrained_dict = torch.load(os.path.join(expt_folder, 'latest_model.pkl'))
trainer.model.load_state_dict(pretrained_dict)

#visualizeFilters(pretrained_dict, expt_folder)

import nibabel as nib
import matplotlib.pyplot as plt
petimage=192
mriimage=1063
def neuro_imshow(img, slice_ind=None, name = None):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        ax[i].set_axis_off()
        
    if slice_ind is None:
        a = np.random.randint(img.shape[2]*1/3,img.shape[2]*2/3)
        ax[0].imshow(img[:,:,a], cmap='gray')
        ax[0].set_title('S')
        a = np.random.randint(img.shape[0]*1/3,img.shape[0]*2/3)
        ax[1].imshow(img[round(img.shape[0]/2),:,:], cmap='gray')
        ax[1].set_title('A')
        a = np.random.randint(img.shape[1]*1/3,img.shape[1]*2/3)
        ax[2].imshow(img[:,a,:], cmap='gray')
        ax[2].set_title('C');
        
    elif slice_ind ==-1:
        
        ax[0].imshow(rotate(img[:,:,round(img.shape[2]/2)].T,180), cmap='gray')
        #ax[0].set_title('Axial')
        ax[1].imshow(rotate(img[round(img.shape[0]/2),:,:].T,180), cmap='gray')
        #ax[1].set_title('Sagittal')
        ax[2].imshow(rotate(img[:,round(img.shape[1]/2),:].T,180), cmap='gray')
        #ax[2].set_title('Coronal');
    else:
        
        ax[0].imshow(img[:,:,slice_ind], cmap='gray')
        ax[0].set_title('S')
        ax[1].imshow(img[slice_ind,:,:], cmap='gray')
        ax[1].set_title('A')
        ax[2].imshow(img[:,slice_ind,:], cmap='gray')
        ax[2].set_title('C');
    
    if name is None:
        plt.show()
    else:
        plt.savefig(name)

        


if modality == 'multi': 
    
    if tasks[0]:
        print('TESTING SOURCE')
        trainer.test(test_loader_s, modality = 0)
    if tasks[1]:
        for i, (images, labels, index) in enumerate(test_loader_s):
            print(index,labels)
            if index!=mriimage:
                continue
            else:
                print(labels)
                print('SUCCES source')
            with torch.no_grad():
                img = Variable(images).cuda()

            _,_ , _, _, x_hat, _ = trainer.model(img)
            img = img.detach().cpu().numpy()[0,0]
            x_hat = x_hat.detach().cpu().numpy()[0,0]
            print(img.shape,x_hat.shape)
            neuro_imshow(img, slice_ind=-1, name = 'source_original'+folder_name+'.png')
            neuro_imshow(x_hat, slice_ind=-1, name = 'source_reconstruction'+folder_name+'.png')
            break
            
if tasks[0]:
    print('TESTING TARGET')
    trainer.test(test_loader_t, modality = 1)
    
    

if tasks[1]:
    for i, (images, labels, index) in enumerate(test_loader_t):
        print(index,labels)
        if index!=petimage and not (index==mriimage and modality=='mri'):
            continue
        else:
            print(labels)
            print('Succes target')
        with torch.no_grad():
            img = Variable(images).cuda()

        _,_ , _, _, x_hat, _ = trainer.model(img)
        img = img.detach().cpu().numpy()[0,0]
        x_hat = x_hat.detach().cpu().numpy()[0,0]
        print(img.shape,x_hat.shape)
        neuro_imshow(img, slice_ind=-1, name = 'target_original'+folder_name+'.png')
        neuro_imshow(x_hat, slice_ind=-1, name = 'target_reconstruction'+folder_name+'.png')
        break


if tasks[2]:

    import h5py
    import os
    from configurations.paths import paths, file_names
    from configurations.modelConfig import params, layer_config
    from data.splitDataset import getIndicesTrainValidTest
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader
    import numpy as np
    import torch
    from torch.autograd import Variable

    import matplotlib as mpl
    # %matplotlib inline
    mpl.use('agg')
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    

    from models.vae import VAE
    import random

    seed = 42
    random.seed(seed)
    #torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_classes = 3
    
    model = VAE()


    # load the new state dict
    model.load_state_dict(pretrained_dict)
    model.cuda()
    #model.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv4.register_forward_hook(get_activation('conv4'))

    print('================ perspective', perspect, ' ====================')
    def save_frames(clean_data, activation, data_index,folder):
        """
        :param folder:
        :param clean_data: one sample of B x 1 x H x W x D
        :param activation: one sample of B x channel x H_k x W_k x D_k
        :slice_n : slice number
        """
        #     mpl.use('Agg')
        #     import matplotlib.pyplot as plt
        print('saving in', folder)
        if not os.path.exists(folder):
            os.mkdir(folder)
        single_data = clean_data[data_index].squeeze(0).cpu().data.numpy()
        one_act = activation[data_index].float().cpu().data.numpy()
        # upsampling
        resized_act = resize(one_act, output_shape=(single_data.shape))
        single_data = np.moveaxis(single_data,perspect,0)
        resized_act = np.moveaxis(resized_act,perspect,0)
        minimum = np.min(resized_act)
        maximum = np.max(resized_act)
        for slice_n in range(clean_data[0,0].shape[perspect]):
            if slice_n<10: continue
            print(slice_n)
            # for slc in list([53, 91, 103, 124, 152]):
            plt.close()
            plt.axis('off')

            brain = rotate(single_data[slice_n], 90)
            # cam = rotate(resized_act[slice_n], 90)
            plt.imshow(brain, cmap='gray')
            cmap = mpl.cm.bwr(np.linspace(0, 1.0, 2))
            cmap = mpl.colors.ListedColormap(cmap[0:, :-1])
            resized_act[slice_n] = (resized_act[slice_n] - minimum) / (maximum-minimum)
            resized_act[slice_n]*= resized_act[slice_n]>0
            saliency = cmap(resized_act[slice_n])
            saliency[..., -1] = np.abs(resized_act[slice_n]-0.5)*2#np.logical_or(resized_act[slice_n]<0.3,resized_act[slice_n]>0.7)*0.6
            saliency = rotate(saliency, 90)
            plt.imshow(saliency)
            plt.savefig(os.path.join(folder, 'file%03d.png' % slice_n))

    minimum_p = [0.6,0.6,0.6]
    data_X = None
    data_Y = None
    done = {'nl_ad': 0}

    for j, [images, nextlabels, indices] in enumerate(test_loader_pet):
        print('test loop',j)
        data_X = Variable(images).cuda()
        data_Y = Variable(nextlabels).cuda()
        fn = data_Y
        print(len(images))
        for i in range(len(images)):
            nextlabel = int(nextlabels[i].cpu().numpy())
            currlabel = nextlabel
            # if currlabel == 0 and nextlabel == 0:
            #     label = 'nl_nl'
            #     lab = 0

            label = 'nl_ad'
            lab = currlabel
            data = data_X[i]
            data = data.unsqueeze(0)
            _, _, _, _, _, p_hat = model(data)
            p_hat = torch.exp(p_hat).cpu().data.numpy()[0]
            print(p_hat)
            if p_hat[lab]<minimum_p[lab]:
                print(' check 2' , fn[i])
                continue
            else:
                print(p_hat)
                minimum_p[lab] = p_hat[lab]
            #    done[label] += 1
            
            act = activation['conv4'].squeeze(0)
            print('data shape:' ,data.shape)
            save_frames(data_X, act, 0, 'small_brains/divergent_' + folder_name+str(lab) + '_pet'+str(perspect)+'_' + str(p_hat[lab]))
            print(j, label, fn[i], currlabel, nextlabel, p_hat)
            break
        # if done['nl_nl'] >= 1 and done['nl_ad'] >= 1 and done['ad_ad'] >= 1:
        if done['nl_ad'] >= 1:
            print('check 4')
            break


