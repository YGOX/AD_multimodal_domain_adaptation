'''
Convolutional Neural Network with reconstrcution loss of euto-encoder as regularization
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from configurations.modelConfig import layer_config, params, num_classes
from configurations.modelConfig import img_shape


def crop(layer, target_size):
    #print('inside crop :', layer.size(), target_size)
    dif = [(layer.size()[2] - target_size[0]) // 2, (layer.size()[3] - target_size[1]) // 2, (layer.size()[4] -
                                                                                            target_size[2]) // 2]
    cs = target_size
    #print(cs, dif)
    
    layer = layer[:, :, dif[0]:dif[0] + cs[0], dif[1]:dif[1] + cs[1], dif[2]:dif[2] + cs[2]]
    #print(layer.size())
    
    return layer

class Adversary(nn.Module):
    """Adversary used for domain adaptation """
    
    def __init__(self):
        super(Adversary, self).__init__()
        self.input_size = round(layer_config['fc1']['in']*4/5)
        self.fc1 = nn.Linear(self.input_size, layer_config['fc1']['out'])
        self.fc2 = nn.Linear(layer_config['fc2']['in'], layer_config['fc2']['out'])
        # self.fc3 = nn.Linear(layer_config['fc3']['in'], layer_config['fc3']['out'])

        self.dropout = nn.Dropout(params['model']['fcc_drop_prob'])
        
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in')
                # nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                nn.init.constant_(m.bias.data, 0.01)
            
    def forward(self, z):
        z = z[:, :self.input_size]
        z = (self.relu(self.fc1(z)))
        z = self.dropout(self.fc2(z))
        prob = self.logsoftmax(z)
        return prob

    
    
class VAE(nn.Module):
    """Cnn with simple architecture of 6 stacked convolution layers followed by a fully connected layer"""
    
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=layer_config['conv1']['in_channels'],
                               out_channels=layer_config['conv1']['out_channels'],
                               kernel_size=layer_config['conv1']['kernel_size'], stride=layer_config['conv1']['stride'],
                               padding=layer_config['conv1']['padding'])
        self.conv2 = nn.Conv3d(in_channels=layer_config['conv2']['in_channels'],
                               out_channels=layer_config['conv2']['out_channels'],
                               kernel_size=layer_config['conv2']['kernel_size'], stride=layer_config['conv2']['stride'],
                               padding=layer_config['conv2']['padding'])
        self.conv3 = nn.Conv3d(in_channels=layer_config['conv3']['in_channels'],
                               out_channels=layer_config['conv3']['out_channels'],
                               kernel_size=layer_config['conv3']['kernel_size'], stride=layer_config['conv3']['stride'],
                               padding=layer_config['conv3']['padding'])
        self.conv4 = nn.Conv3d(in_channels=layer_config['conv4']['in_channels'],
                               out_channels=layer_config['conv4']['out_channels'],
                               kernel_size=layer_config['conv4']['kernel_size'], stride=layer_config['conv4']['stride'],
                               padding=layer_config['conv4']['padding'])
        
        
        
        self.tconv1 = nn.Conv3d(in_channels=layer_config['tconv1']['in_channels'],
                                         out_channels=layer_config['tconv1']['out_channels'],
                                         kernel_size=layer_config['tconv1']['kernel_size'],
                                         stride=layer_config['tconv1']['stride'],
                                         padding=layer_config['tconv1']['padding'])
        self.tconv2 = nn.Conv3d(in_channels=layer_config['tconv2']['in_channels'],
                                         out_channels=layer_config['tconv2']['out_channels'],
                                         kernel_size=layer_config['tconv2']['kernel_size'],
                                         stride=layer_config['tconv2']['stride'],
                                         padding=layer_config['tconv2']['padding'])
        self.tconv3 = nn.Conv3d(in_channels=layer_config['tconv3']['in_channels'],
                                         out_channels=layer_config['tconv3']['out_channels'],
                                         kernel_size=layer_config['tconv3']['kernel_size'],
                                         stride=layer_config['tconv3']['stride'],
                                         padding=layer_config['tconv3']['padding'])
        self.tconv4 = nn.Conv3d(in_channels=layer_config['tconv4']['in_channels'],
                                         out_channels=layer_config['tconv4']['out_channels'],
                                         kernel_size=layer_config['tconv4']['kernel_size'],
                                         stride=layer_config['tconv4']['stride'],
                                         padding=layer_config['tconv4']['padding'])
        
        self.fc_mean = nn.Linear(layer_config['gaussian'], layer_config['z_dim'])
        self.fc_logvar = nn.Linear(layer_config['gaussian'], layer_config['z_dim'])
        self.lineard = nn.Linear(layer_config['z_dim'], layer_config['gaussian'])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
        self.fc1 = nn.Linear(layer_config['fc1']['in'], layer_config['fc1']['out'])
        self.fc2 = nn.Linear(layer_config['fc2']['in'], layer_config['fc2']['out'])
        
        self.global_avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.bn1 = nn.BatchNorm3d(layer_config['conv1']['out_channels'])
        self.bn2 = nn.BatchNorm3d(layer_config['conv2']['out_channels'])
        self.bn3 = nn.BatchNorm3d(layer_config['conv3']['out_channels'])
        self.bn4 = nn.BatchNorm3d(layer_config['conv4']['out_channels'])
        
        self.tbn1 = nn.BatchNorm3d(layer_config['tconv1']['out_channels'])
        self.tbn2 = nn.BatchNorm3d(layer_config['tconv2']['out_channels'])
        self.tbn3 = nn.BatchNorm3d(layer_config['tconv3']['out_channels'])
        self.tbn4 = nn.BatchNorm3d(layer_config['tconv4']['out_channels'])
        
        self.dropout3d = nn.Dropout3d(p=params['model']['conv_drop_prob'])
        self.dropout = nn.Dropout(params['model']['fcc_drop_prob'])
        self.dropout2 = nn.Dropout(params['model']['fcc_init_drop_prob'])
        
        self.maxpool = nn.MaxPool3d(layer_config['maxpool3d']['ln']['kernel'], layer_config['maxpool3d'][
            'ln']['stride'], ceil_mode=True)
        
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        if params['train']['alternative']:
            self.bn5 = nn.BatchNorm3d(layer_config['conv5']['out_channels'])
            #self.bn6 = nn.BatchNorm3d(layer_config['conv6']['out_channels'])
            #self.bn7 = nn.BatchNorm3d(layer_config['conv7']['out_channels'])
            #self.bn8 = nn.BatchNorm3d(layer_config['conv8']['out_channels'])

            self.tbn5 = nn.BatchNorm3d(layer_config['tconv5']['out_channels'])
            #self.tbn6 = nn.BatchNorm3d(layer_config['tconv6']['out_channels'])
            #self.tbn7 = nn.BatchNorm3d(layer_config['tconv7']['out_channels'])
            #self.tbn8 = nn.BatchNorm3d(layer_config['tconv8']['out_channels'])
            
            self.conv5 = nn.Conv3d(in_channels=layer_config['conv5']['in_channels'],
                               out_channels=layer_config['conv5']['out_channels'],
                               kernel_size=layer_config['conv5']['kernel_size'], stride=layer_config['conv4']['stride'],
                               padding=layer_config['conv5']['padding'])
            #self.conv6 = nn.Conv3d(in_channels=layer_config['conv6']['in_channels'],
            #                   out_channels=layer_config['conv6']['out_channels'],
            #                   kernel_size=layer_config['conv6']['kernel_size'], stride=layer_config['conv6']['stride'],
            #                   padding=layer_config['conv6']['padding'])
            #self.conv7 = nn.Conv3d(in_channels=layer_config['conv7']['in_channels'],
            #                   out_channels=layer_config['conv7']['out_channels'],
            #                   kernel_size=layer_config['conv7']['kernel_size'], stride=layer_config['conv7']['stride'],
            #                   padding=layer_config['conv7']['padding'])
            #self.conv8 = nn.Conv3d(in_channels=layer_config['conv8']['in_channels'],
            #                   out_channels=layer_config['conv8']['out_channels'],
            #                   kernel_size=layer_config['conv8']['kernel_size'], stride=layer_config['conv8']['stride'],
            #                   padding=layer_config['conv8']['padding'])
            self.tconv5 = nn.Conv3d(in_channels=layer_config['tconv5']['in_channels'],
                                             out_channels=layer_config['tconv5']['out_channels'],
                                             kernel_size=layer_config['tconv5']['kernel_size'],
                                             stride=layer_config['tconv5']['stride'],
                                             padding=layer_config['tconv5']['padding'])
            #self.tconv6 = nn.Conv3d(in_channels=layer_config['tconv6']['in_channels'],
            #                                 out_channels=layer_config['tconv6']['out_channels'],
            #                                 kernel_size=layer_config['tconv6']['kernel_size'],
            #                                 stride=layer_config['tconv6']['stride'],
            #                                 padding=layer_config['tconv6']['padding'])
            #self.tconv7 = nn.Conv3d(in_channels=layer_config['tconv7']['in_channels'],
            #                                 out_channels=layer_config['tconv7']['out_channels'],
            #                                 kernel_size=layer_config['tconv7']['kernel_size'],
            #                                 stride=layer_config['tconv7']['stride'],
            #                                 padding=layer_config['tconv7']['padding'])
            #self.tconv8 = nn.Conv3d(in_channels=layer_config['tconv8']['in_channels'],
            #                                 out_channels=layer_config['tconv8']['out_channels'],
            #                                 kernel_size=layer_config['tconv8']['kernel_size'],
            #                                 stride=layer_config['tconv8']['stride'],
            #                                 padding=layer_config['tconv8']['padding'])
        
        
        self.shapes = []
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in')
                # nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                nn.init.constant_(m.bias.data, 0.01)
            
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in')
                # nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                nn.init.constant_(m.bias.data, 0.01)
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0.01)
                
    def encoder(self, x):
        
        #print('encoder : ', x.size())
        self.shapes.append(x.size()[-3:])
        
        x = self.dropout3d(self.relu(self.bn1(self.maxpool(self.conv1(x)))))
        self.shapes.append(x.size()[-3:])
        # print(x.size())
        
        x = self.dropout3d(self.relu(self.bn2(self.maxpool(self.conv2(x)))))
        self.shapes.append(x.size()[-3:])
        # print(x.size())
        
        x = self.dropout3d(self.relu(self.bn3(self.maxpool(self.conv3(x)))))
        self.shapes.append(x.size()[-3:])
        # print(x.size())
        
        x = self.dropout3d(self.relu(self.bn4(self.maxpool(self.conv4(x)))))
       
        
        if params['train']['alternative']:
            self.shapes.append(x.size()[-3:])
            x = self.dropout3d(self.relu(self.bn5(self.maxpool(self.conv5(x)))))
            #self.shapes.append(x.size()[-3:])
            # print(x.size())

            #x = self.dropout3d(self.relu(self.bn6(self.maxpool(self.conv6(x)))))
            #self.shapes.append(x.size()[-3:])
            # print(x.size())

            #x = self.dropout3d(self.relu(self.bn7(self.maxpool(self.conv7(x)))))
            #self.shapes.append(x.size()[-3:])
            # print(x.size())

            #x = self.dropout3d(self.relu(self.bn8(self.maxpool(self.conv8(x)))))
            
            
        #print('encoder : ', self.shapes)
        #print(x.size())
        
        # flatten
        h = x.view(x.size(0), -1)
        #print(h.size())
        
        mu = self.dropout2(self.fc_mean(h))
        logvar = self.fc_logvar(h)
        
        return mu, logvar, x
    
    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(mu.size())).cuda()
        z = mu + sigma * eps
        return z
    
    def decoder(self, z):
        #print('decoder : ', x.size())
        if params['train']['glav']:
            x_hat = z
        else:
            x_hat = self.dropout2((self.relu(self.lineard(z))))
            #print(x_hat.size())
            x_hat = x_hat.view(x_hat.size(0), -1, int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
            #print(x_hat.size())
        
        if params['train']['alternative']:
            x_hat = self.dropout3d(self.relu(self.tbn5(self.tconv5(self.upsample(x_hat)))))
            x_hat = crop(x_hat, self.shapes[-1])
            self.shapes.pop()
            #print(x_hat.size())

            #x_hat = self.dropout3d(self.relu(self.tbn6(self.tconv6(self.upsample(x_hat)))))
            #x_hat = crop(x_hat, self.shapes[-1])
            #self.shapes.pop()
            #print(x_hat.size())

            #x_hat = self.dropout3d(self.relu(self.tbn7(self.tconv7(self.upsample(x_hat)))))
            #x_hat = crop(x_hat, self.shapes[-1])
            #self.shapes.pop()
            #print(x_hat.size())

            #x_hat = self.dropout3d(self.relu(self.tbn8(self.tconv8(self.upsample(x_hat)))))
            #x_hat = crop(x_hat, self.shapes[-1])
            #self.shapes.pop()
        
        x_hat = self.dropout3d(self.relu(self.tbn1(self.tconv1(self.upsample(x_hat)))))
        x_hat = crop(x_hat, self.shapes[-1])
        self.shapes.pop()
        #print(x_hat.size())
        
        x_hat = self.dropout3d(self.relu(self.tbn2(self.tconv2(self.upsample(x_hat)))))
        x_hat = crop(x_hat, self.shapes[-1])
        self.shapes.pop()
        #print(x_hat.size())
        
        x_hat = self.dropout3d(self.relu(self.tbn3(self.tconv3(self.upsample(x_hat)))))
        x_hat = crop(x_hat, self.shapes[-1])
        self.shapes.pop()
        #print(x_hat.size())
        
        x_hat = self.dropout3d(self.relu(self.tbn4(self.tconv4(self.upsample(x_hat)))))
        x_hat = crop(x_hat, self.shapes[-1])
        self.shapes.pop()
       

        
        #print(x_hat.size())
        
        return x_hat
    
    
    def classifier(self, z):
        
        # print(x.size())
        if params['train']['glav']:
            z = self.global_avg_pooling(z)
            z = z.view(z.size(0), -1)
        z = self.dropout((self.relu(self.fc1(z))))
        # print(x.size())
        z = self.fc2(z) #self.fc2(z) changed this, so last z is returned and not middle layer
        prob = self.logsoftmax(z)
        
        return prob, z
    
    # print(output.size())
    

    def forward(self, x):
        # encoder
        
        mu, logvar, x = self.encoder(x)
        #print(mu.size(), logvar.size())
        
        #reparameterize
        if params['train']['variational']:
            z = self.reparametrize(mu, logvar)
        #print(z.size())
        elif params['train']['glav']:
            z = x
        else:
            z = mu
            x = mu
        # decoder
        x_hat = self.decoder(z)
        
        # classifier
        class_prob, classifier_embedding = self.classifier(x)
        
        
        
        
        
        return z, classifier_embedding, mu, logvar, x_hat, class_prob
    

