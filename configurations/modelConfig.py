import numpy as np

type_of_learning = 3




if type_of_learning == 3:
    name_classes = np.asarray(['CN', 'MCI', 'AD'])
    #class_weight = [1, 1, 1]
elif type_of_learning == 2:
    name_classes = np.asarray(['CN', 'Abnormal'])
    #class_weight = [1, 1]
else:
    name_classes = np.asarray(['MCI', 'AD'])

    
params = {
    'model': {
        'conv_drop_prob': 0.0,
        'fcc_drop_prob' : 0.3,
        'fcc_init_drop_prob' : 0.0,
        'advers_feats'  : 0,
        'advers_use'    : False
    },
    
    'train'    :    {
        'experiment_id'     : 'NO-OUTPUT',
        'model'             : 'VAE',
        'timestamp'         : 'CurrLabel',    #'CurrLabel'
        'seed'              : 4242,
        'learning_rate'     : 0.0001,
        'num_epochs'        : 40,
        'batch_size'        : 16,              
        'label_weights'     : 0,
        'lambda'            : 1,
        'nu'                : 2,
        'kappa'             : 1,
        'lr_schedule'       : [15, 25, 35],
        'glav'              : False, #use global averaging instead of fully connected classifier
        'variational'       : False,  #use variational auto encoder. Requires glav=False
        'set_to_use'        : 'both_2', #options: big or small
        'GPU'               : '1',
        'modality'          : 'multi',
        'alternative'       : False,
        'zoom'              : False,
        'hard_encoding'     : True,
        'debug'             : False
    }
}

if params['train']['debug']==True:
    params['train']['batch_size']=1

extra_big = params['train']['alternative']
num_classes = len(name_classes)
latent_dim = 256
middle_layer_discrim = 128
layer_1 = 24
layer_2 = 24
layer_3 = 24
layer_4 = 24

layer_5 = 12
layer_6 = 48
layer_7 = 64
layer_8 = 11



if params['train']['glav']:
    if extra_big:
        input_classifier = layer_8 #latent_dim #alternatively, use layer_4
    else:
        input_classifier = layer_4
else:
    input_classifier = latent_dim    

if extra_big:
    num_conv = 5
else:
    num_conv = 4



if params['train']['set_to_use'] == 'both_2':
    img_shape = np.array([120,132,96])

elif params['train']['modality']=='pet':
    img_shape = np.array([160,96,160])
elif params['train']['modality']=='mri' and params['train']['zoom']==False:
    img_shape = np.array([182,218,182])#np.array([213, 197, 189])
    

for _ in range(num_conv):
    img_shape = np.ceil(img_shape / 2)
    
#img_shape.astype(np.int32)

layer_config = {
    'conv1': {
        'in_channels': 1,
        'out_channels': layer_1,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv2': {
        'in_channels': layer_1,
        'out_channels': layer_2,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv3': {
        'in_channels': layer_2,
        'out_channels': layer_3,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv4': {
        'in_channels': layer_3,
        'out_channels': layer_4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv5': {
        'in_channels': layer_4,
        'out_channels': layer_5,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv6': {
        'in_channels': layer_5,
        'out_channels': layer_6,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv7': {
        'in_channels': layer_6,
        'out_channels': layer_7,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv8': {
        'in_channels': layer_7,
        'out_channels': layer_8,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'gaussian': layer_4 * int(np.prod(img_shape[:])),  # 14 * 13 * 12,,
    'z_dim': latent_dim,
    
    'fc1': {
        'in': input_classifier,  # 14 * 13 * 12,
        'out': middle_layer_discrim
    },
    'fc2': {
        'in': middle_layer_discrim,
        'out': num_classes
    },
    'tconv5': {
        'in_channels': layer_5,
        'out_channels': layer_4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1    #(0, 0, 1)
    },
    'tconv6': {
        'in_channels': layer_7,
        'out_channels': layer_6,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1
    },
    'tconv7': {
        'in_channels': layer_6,
        'out_channels': layer_5,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1    #(0, 0, 0)
    },
    'tconv8': {
        'in_channels': layer_5,
        'out_channels': layer_4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1    #(0, 0, 0)
    },
    'tconv1': {
        'in_channels': layer_4,
        'out_channels': layer_3,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1    #(0, 0, 1)
    },
    'tconv2': {
        'in_channels': layer_3,
        'out_channels': layer_2,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1
    },
    'tconv3': {
        'in_channels': layer_2,
        'out_channels': layer_1,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1    #(0, 0, 0)
    },
    'tconv4': {
        'in_channels': layer_1,
        'out_channels': 1,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
        # 'output_padding' : 1    #(0, 0, 0)
    },
    'maxpool3d': {
        'ln': {
            'kernel': 2,
            'stride': 2
        },
        
        'adaptive': 1
    }
}

if extra_big:
    layer_config['gaussian'] = layer_5 * int(np.prod(img_shape[:]))  # 14 * 13 * 12,,
    


# data augmentation
data_aug = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.5,
    'rotation_range': 10,
    #'spline_warp': True,
    #'warp_sigma': 0.1,
    #'warp_grid_size': 3,
    ## 'crop_size': (100, 100),
    'channel_shift_range': 0.2
}