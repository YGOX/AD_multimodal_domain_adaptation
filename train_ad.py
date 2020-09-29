''''
train model
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from configurations.modelConfig import params, num_classes, name_classes
from tqdm import tqdm
import numpy as np
from utils.visualizations import plot_confusion_matrix, plot_embedding
from utils.metrics import updateConfusionMatrix, calculateF1Score, calculate_balanced_acc, updatePairwiseCM
from tensorboardX import SummaryWriter
from utils.save import saveModelandMetrics
from torch.optim.lr_scheduler import MultiStepLR
from data.splitDataset import getIndicesTrainValidTest
from utils.visualizations import plotROC





class Trainer(object):
    def __init__(self, model, train_loader_s, valid_loader_s, train_loader, valid_loader, expt_folder, setsizes, weights = None, adversary=None ):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda:0")
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count() 
            if num_gpus>1:
                print("Using GPUs:",torch.cuda.device_count())
                self.model = nn.DataParallel(model)
            else:
                self.model = model
                print("Single GPU is used")
            self.model.to(self.device)
        else:
            self.model = model
        
        self.multi = train_loader_s is not None
        
        if adversary is not None:
            self.ad_net = adversary
            if num_gpus>1:
                print("Using GPUs:",torch.cuda.device_count())
                self.ad_net = nn.DataParallel(self.ad_net)
            self.ad_net.to(self.device)
            self.adversary = self.multi
            self.adversarial_loss = 0 # changeeee
            self.ad_criterion = nn.NLLLoss()
            self.ad_optimizer = torch.optim.AdamW(adversary.parameters(), lr=params['train']['learning_rate'])
            self.ad_weight = params['train']['kappa']
        else:
            self.ad_weight = 0
            self.adversary = False
            
        if self.multi:
            self.train_loader_s = train_loader_s
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            self.valid_loader_s = valid_loader_s
            
        else:    
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            
        self.trainset_size, self.validset_size, self.testset_size = setsizes[:,0], setsizes[:,1], setsizes[:,2]
        print('set sizes', setsizes)
            
        
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=params['train']['learning_rate'])
        
        if weights is None:
            weigths = params['train']['label_weights']
            print('Using default weights:', weigths)
        
        
        if params['train']['hard_encoding']:
            if self.multi: self.classification_criterion_s = nn.NLLLoss(weight=torch.FloatTensor(weights[0,:]).cuda())
        
            self.classification_criterion = nn.NLLLoss(weight=torch.FloatTensor(weights[1,:]).cuda())
        else:
            self.classification_criterion = nn.NLLLoss(weight=torch.FloatTensor(weights[1,:]).cuda())
            self.weights_s = torch.FloatTensor(weights[0,:]).cuda()
            self.weights = torch.FloatTensor(weights[1,:]).cuda()
            self.encod_mat = torch.tensor([[5/6, 1/6, 0],[1/6,2/3,1/6],[0,1/6,5/6]]).cuda()
            
        self.reconstruction_loss = nn.MSELoss()           
        
        
        self.curr_epoch = 0
        self.best_epoch = 0
        self.batchstep = 0
        
        self.expt_folder = expt_folder
        self.writer = SummaryWriter(log_dir=expt_folder)
        num_epochs = params['train']['num_epochs']
        self.train_losses_class, self.train_losses_vae, self.train_mse, self.train_kld, \
        self.train_bal_accuracy, self.train_f1_Score, self.train_accuracy = ([] for i in range(7))
        if num_classes == 3: 
            self.cm_per_class = np.zeros((num_epochs,2,3,2,2))
            self.valid_bal_acc_per_class = np.zeros((num_epochs,2,3))
        self.valid_mse = np.zeros((num_epochs,2))
        self.valid_kld = np.zeros((num_epochs,2))
        self.valid_f1_Score = np.zeros((num_epochs,2))
        self.valid_bal_accuracy = np.zeros((num_epochs,2))
        self.valid_accuracy = np.zeros((num_epochs,2))
        self.valid_losses = np.zeros((num_epochs,2))
        
        
        
    def xentropy_soft(self, p_hat, labels, weight):
        labels_onehot = F.one_hot(labels,3).float()
        labels_new    = torch.matmul(labels_onehot, self.encod_mat)
        assert labels_new.size() == p_hat.size(), "size mismatch ! "+str(labels_new.size()) + " " + str(p_hat.size())
        weights = torch.matmul(labels_onehot, weight)
        cost_value = -torch.mean(torch.matmul(weights,labels_new * p_hat))
        return cost_value
    
    def klDivergence(self, mu, logvar):
        # D_KL(Q(z|X) || P(z|X))
        # P(z|X) is the real distribution, Q(z|X) is the distribution we are trying to approximate P(z|X) with
        # calculate in closed form
        return (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    
    # self.lambda_ = 1    #hyper-parameter to control regularizer by reconstruction loss
    def vae_loss(self, recon_x, x, mu, logvar):
        MSE = self.reconstruction_loss(recon_x, x)
        # MSE = nn.CrossEntropyLoss(recon_x, x, size_average=False)
        # BCE = F.mse_loss(recon_x, x, size_average=False)
        
        if params['train']['variational']:
            KLD = self.klDivergence(mu, logvar)
        else:
            KLD = torch.zeros(MSE.size())
            
        return MSE + KLD, MSE, KLD
    
    def train(self):
        scheduler = MultiStepLR(self.optimizer, milestones=params['train']['lr_schedule'], gamma=0.1)  # [40,
        # 60] earlier
        best_val_acc = 0.0
        best_val_loss = 0.0
        best_val_f1 = 0.0
        no_improvement=0
        breakpoint = 5
        
        for _ in range(params['train']['num_epochs']):
            print('Training...\nEpoch : ' + str(self.curr_epoch))
            self.iter_train = iter(self.train_loader)
            if self.multi:
                self.iter_train_s = iter(self.train_loader_s)
                self.num_batches = max([len(self.iter_train_s),len(self.iter_train)])
            else:
                self.num_batches = len(self.iter_train)
            
            # Train Model
            accuracy, classification_loss, vae_loss, mse, kld, f1_score, bal_acc = self.trainEpoch()
            scheduler.step()
            
            
            self.train_losses_class.append(classification_loss)
            self.train_losses_vae.append(vae_loss)
            self.train_mse.append(mse)
            self.train_kld.append(kld)
            self.train_accuracy.append(accuracy)
            self.train_f1_Score.append(f1_score)
            self.train_bal_accuracy.append(bal_acc)
            
            # Validate Model
            print ('Validation...')
            with torch.no_grad():
                if self.multi: _, _, _= self.validate(modality = 0)
                val_acc, val_loss, val_f1 = self.validate(modality = 1)
            # Save model
            if val_loss <= best_val_loss or val_acc >= best_val_acc or val_f1 >= best_val_f1:
                best_val_loss= val_loss
                best_val_acc= val_acc
                best_val_f1 = val_f1
                saveModelandMetrics(self, best_val_acc, best_val_loss, best_val_f1)
                no_improvement = 0
                if val_loss <= best_val_loss:
                    self.best_epoch = self.curr_epoch
            else:
                no_improvement+=1
                if no_improvement==breakpoint: 
                    print('@@@@@@@@@@@@@@@@@@@@@@@ Epoch',self.curr_epoch,'No improvement in', breakpoint, 'iterations @@@@@@@@@@@@@@@@@@@@@@@@')
                    break #early stopping when stuff doesn't get better
            self.curr_epoch += 1
    
    def trainEpoch(self):
        self.model.train(True)
        
        pbt = tqdm(total=self.num_batches)
        
        cm = np.zeros((2, num_classes, num_classes), int)
        
        minibatch_losses_class = np.zeros(2)
        minibatch_losses_vae = np.zeros(2)
        minibatch_accuracy = np.zeros(2)
        minibatch_kld = np.zeros(2)
        minibatch_mse = np.zeros(2)
        
        for batch_idx in range(self.num_batches):
            if batch_idx>0 and params['train']['debug']:
                print('Debug mode is on, breaking after first batch')
                break
            try:
                batch = self.iter_train.next()
            except StopIteration:
                self.iter_train = iter(self.train_loader)
                batch = self.iter_train.next()
            
            (images, labels, _) = batch
            
            if self.multi:
                try:
                    batch = self.iter_train_s.next()
                except StopIteration:
                    self.iter_train_s = iter(self.train_loader_s)
                    batch = self.iter_train_s.next()
                (images_s, labels_s, _) = batch
            else:
                images_s, labels_s = None, None
            
            torch.cuda.empty_cache()
            
            accuracy, class_loss, conf_mat, vae_loss, kld, mse = self.trainBatch(batch_idx, images_s, labels_s, images, labels)
            
            minibatch_losses_class += class_loss
            minibatch_losses_vae += vae_loss
            minibatch_kld += kld
            minibatch_mse += mse
            
            minibatch_accuracy = minibatch_accuracy + accuracy
            cm = cm + conf_mat
            
            pbt.update(1)
        
        pbt.close()
        
        minibatch_losses_class /= (np.max(self.trainset_size[0]))
        minibatch_losses_vae /= (np.max(self.trainset_size[0]))
        minibatch_mse /= (np.max(self.trainset_size[0]))
        minibatch_kld /= (np.max(self.trainset_size[0]))
        minibatch_accuracy /= (np.max(self.trainset_size[0]))
        
        bal_accuracy = calculate_balanced_acc(cm[1,:,:])
        if self.multi: 
            bal_accuracy_s = calculate_balanced_acc(cm[0,:,:])
        else:
            bal_accuracy_s = 0
        
        minibatch_bal_accuracy = np.array([bal_accuracy_s,bal_accuracy])
        
        # Plot losses
        self.writer.add_scalar('train_classification_loss', minibatch_losses_class[1], self.curr_epoch)
        self.writer.add_scalar('train_vae_loss', minibatch_losses_vae[1], self.curr_epoch)
        self.writer.add_scalar('train_reconstruction_loss', minibatch_mse[1], self.curr_epoch)
        self.writer.add_scalar('train_KL_divergence', minibatch_kld[1], self.curr_epoch)
        self.writer.add_scalar('train_accuracy', minibatch_accuracy[1], self.curr_epoch)
        self.writer.add_scalar('train_accuracy', minibatch_bal_accuracy[1], self.curr_epoch)
        
        self.writer.add_scalar('train_classification_loss_s', minibatch_losses_class[0], self.curr_epoch)
        self.writer.add_scalar('train_vae_loss_s', minibatch_losses_vae[0], self.curr_epoch)
        self.writer.add_scalar('train_reconstruction_loss_s', minibatch_mse[0], self.curr_epoch)
        self.writer.add_scalar('train_KL_divergence_s', minibatch_kld[0], self.curr_epoch)
        self.writer.add_scalar('train_accuracy_s', minibatch_accuracy[0], self.curr_epoch)
        self.writer.add_scalar('train_accuracy_s', minibatch_bal_accuracy[0], self.curr_epoch)
        
        
        # Plot confusion matrices
        plot_confusion_matrix(cm[1,:,:], location=self.expt_folder, title='VAE on Train Set')
        
        # F1 Score
        f1_score = calculateF1Score(cm[1,:,:])
        self.writer.add_scalar('train_f1_score', f1_score, self.curr_epoch)
        print('F1 Score : ', f1_score)
        
        # plot ROC curve
        # plotROC(cm, location=self.expt_folder, title='ROC Curve(Train)')
        
        return minibatch_accuracy, minibatch_losses_class, minibatch_losses_vae, minibatch_mse, minibatch_kld, f1_score, minibatch_bal_accuracy
    
    
    def trainBatch(self, batch_idx, images_s, labels_s, images, labels):
        if self.multi: 
            batch_size = len(labels_s)
            batch_size_t = len(labels)
        else:
            batch_size = len(labels)
        
        # Forward
        # x_hat is reconstructed image, p_hat is predicted classification probability
        if self.multi:
            _, _, mu, logvar, x_hat, p_hat = self.model(Variable(torch.cat((images_s,images),0)).cuda())
            if self.adversary:
                for param in self.ad_net.parameters():
                    param.requires_grad = False
                ad_preds = self.ad_net(mu.cuda())
                domain_labels = Variable(torch.LongTensor([[1]*batch_size+[0]*batch_size_t])).squeeze()
                ad_loss = self.ad_criterion(ad_preds.cuda(), domain_labels.cuda())
            vae_loss_s, mse_s, kld_s = self.vae_loss(x_hat[:batch_size].cuda(), Variable(images_s).cuda(), mu[:batch_size].cuda(), logvar[:batch_size].cuda())
            vae_loss, mse, kld = self.vae_loss(x_hat[batch_size:].cuda(), Variable(images).cuda(), mu[batch_size:].cuda(), logvar[batch_size:].cuda())
        else:
            _, _, mu, logvar, x_hat, p_hat = self.model(Variable(images).cuda())
            vae_loss, mse, kld = self.vae_loss(x_hat.cuda(), Variable(images).cuda(), mu.cuda(), logvar.cuda())
        
        #print('hello')
        

        #mu_s, logvar_s, x_hat_s, p_hat_s = None, None, None, None #mu[:batch_size], logvar[:batch_size], x_hat[:batch_size], p_hat[:batch_size]
        
        #mu_t, logvar_t, x_hat_t, p_hat_t = None, None, None, None #mu[batch_size:], logvar[batch_size:], x_hat[batch_size:], p_hat[batch_size:]
        
        #vae_loss_s, mse_s, kld_s = self.vae_loss(x_hat_s.cuda(), images_s.cuda(), mu_s.cuda(), logvar_s.cuda())
        #vae_loss, mse, kld = self.vae_loss(x_hat.cuda(), images.cuda(), mu_t.cuda(), logvar.cuda())
        
        
        labels = Variable(labels).cuda()
        labels = labels.view(-1, )
        
        if self.multi:
            labels_s = Variable(labels_s).cuda()
            labels_s = labels_s.view(-1, )
        
            if params['train']['hard_encoding']: 
                classification_loss_s = self.classification_criterion(p_hat[:batch_size].cuda(), labels_s)
                classification_loss = self.classification_criterion(p_hat[batch_size:].cuda(), labels)
            else:
                classification_loss_s = self.xentropy_soft(p_hat[:batch_size].cuda(), labels_s, self.weights_s)
                classification_loss = self.xentropy_soft(p_hat[batch_size:].cuda(), labels, self.weights)
            
            loss = classification_loss + params['train']['lambda'] * vae_loss + params['train']['nu'] *(classification_loss_s + params['train']['lambda'] * vae_loss_s)
            
            if self.adversary:
                loss = loss + self.ad_weight * ad_loss
        
        else:
            if params['train']['hard_encoding']: 
                classification_loss = self.classification_criterion(p_hat.cuda(), labels)
            else:
                classification_loss = self.xentropy_soft(p_hat.cuda(), labels, self.weights)
            
            loss = classification_loss + params['train']['lambda'] * vae_loss
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.multi: 
            vae_loss_s, mse_s, kld_s = vae_loss_s.data.item(), mse_s.data.item(), kld_s.data.item()
        vae_loss, mse, kld = vae_loss.data.item(), mse.data.item(), kld.data.item()
        
        if self.multi:
            if self.adversary:
                # Train domain discriminator
                for param in self.ad_net.parameters():
                    param.requires_grad = True
                ad_preds = self.ad_net(mu.detach().cuda())
                domain_labels = Variable(torch.LongTensor([[0]*batch_size+[1]*batch_size_t])).squeeze()
                ad_loss_2 = self.ad_criterion(ad_preds.cuda(), domain_labels.cuda())
                self.ad_optimizer.zero_grad()
                # classification_loss.backward(retain_graph=True)
                ad_loss_2.backward()
                self.ad_optimizer.step()

                del ad_loss, ad_preds, ad_loss_2, domain_labels
        
        
            # Compute accuracy
            _, pred_labels_s = torch.max(p_hat[:batch_size], 1)
            accuracy_s = (labels_s == pred_labels_s).float().sum()

            _, pred_labels = torch.max(p_hat[batch_size:], 1)
            accuracy = (labels == pred_labels).float().sum()
            cm_s = updateConfusionMatrix(labels_s.data.cpu().numpy(), pred_labels_s.data.cpu().numpy())
            cm = updateConfusionMatrix(labels.data.cpu().numpy(), pred_labels.data.cpu().numpy())
            classification_loss_s = classification_loss_s.data.item()
            classification_loss = classification_loss.data.item()
        
        else:
            _, pred_labels = torch.max(p_hat, 1)
            accuracy = (labels == pred_labels).float().sum()
            accuracy_s,vae_loss_s, mse_s, kld_s  = 0,0,0,0
            cm = updateConfusionMatrix(labels.data.cpu().numpy(), pred_labels.data.cpu().numpy())
            cm_s = np.zeros(cm.shape, dtype = cm.dtype)
            classification_loss = classification_loss.data.item()
            classification_loss_s = 0
        
        
        
        
       # Print metrics
        if batch_idx % 100 == 0:
            print('Epoch [%d/%d], Batch [%d/%d] Classification Loss: %.4f VAE Loss: %.4f Accuracy: %0.2f '
                  % (self.curr_epoch, params['train']['num_epochs'],
                     batch_idx, self.trainset_size[1],
                     classification_loss,#.item(),
                     vae_loss,#.item(),
                     accuracy * 1.0 / params['train']['batch_size']))
        
        
        
        # clean GPU
        del images, labels, x_hat, p_hat,  pred_labels, mu, logvar, _ #, mu_s, logvar_s, p_hat_s, x_hat_s
        
        if self.multi: 
            del images_s, labels_s, pred_labels_s
            
        self.writer.add_scalar('minibatch_classification_loss', np.mean(classification_loss), self.batchstep)
        self.writer.add_scalar('minibatch_vae_loss', np.mean(vae_loss), self.batchstep)
        self.batchstep += 1
        
        accuracy = np.array([accuracy_s, accuracy])
        class_loss = np.array([classification_loss_s, classification_loss])
        vae_loss = np.array([vae_loss_s, vae_loss])
        kld = np.array([kld_s, kld])
        mse = np.array([mse_s, mse])
        cm = [cm_s, cm]
        
        del classification_loss_s, classification_loss, kld_s, mse_s, cm_s, accuracy_s
        
        return accuracy, class_loss, cm, vae_loss, kld, mse
    
    
    
    def validate(self,modality = 1):
        self.model.eval()
        correct = 0
        mse = 0
        kld = 0
        cm = np.zeros((num_classes, num_classes), int)
        loss = 0
        domain_correct = 0
        if modality==1:
            modname = 'target'
            loader = self.valid_loader
        else:
            modname = 'source'
            loader = self.valid_loader_s
        
        print(modname)
        pb = tqdm(total=len(loader))
        
        for i, (images, labels,_) in enumerate(loader):
            with torch.no_grad():
                img = Variable(images).cuda()
                
                _, _, mu, logvar, x_hat, p_hat = self.model(img)
                _, predicted = torch.max(p_hat.data, 1)
                labels = labels.view(-1, )
                correct += ((predicted.cpu() == labels).float().sum())
                correct += ((predicted.cpu() == labels).float().sum())
                correct += ((predicted.cpu() == labels).float().sum())
                if self.adversary:
                    ad_probs = self.ad_net(mu)
                    _, ad_pred = torch.max(ad_probs.data, 1)
                    domain_correct += (ad_pred==modality).sum()
                
                cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
                
                if num_classes == 3:
                    labels2 = labels.clone().detach().numpy()
                    self.cm_per_class[self.curr_epoch,modality] += updatePairwiseCM(labels2,p_hat.data.cpu().numpy())
                    del labels2
                
                loss += self.classification_criterion(p_hat, Variable(labels).cuda()).data
                mse += self.reconstruction_loss(x_hat, img).data
                kld += self.klDivergence(mu, logvar).data
                
            del img, x_hat, p_hat,  predicted, mu, logvar, labels, _
            pb.update(1)
        
        pb.close() 
        correct /= self.validset_size[modality]
        bal_acc = calculate_balanced_acc(cm)
        mse /= self.validset_size[modality]
        kld /= self.validset_size[modality]
        loss /= self.validset_size[modality]
        
        print('Validation Balanced accuracy : %0.6f' % bal_acc)
        print('Train Balanced accuracy '+modname+': %0.6f' % self.train_bal_accuracy[-1][modality])
        if self.adversary:
            domain_correct = domain_correct/self.validset_size[modality]
            print('Validation Domain accuracy '+modname+': %0.6f' % domain_correct)
        
        
        if num_classes==3:
            classifications = ['MCI vs AD','CN vs AD', 'CN vs MCI']
            bal_acc_per_class = np.zeros(3)
            for c in range(3):
                print(classifications[c])
                print(self.cm_per_class[self.curr_epoch,modality,c])
                bal_acc_per_class[c] = calculate_balanced_acc(self.cm_per_class[self.curr_epoch,modality,c])
                print('Balanced acc '+modname+':',bal_acc_per_class[c])
            
            self.valid_bal_acc_per_class[self.curr_epoch, modality] = bal_acc_per_class
        
        self.valid_accuracy[self.curr_epoch, modality] = correct
        self.valid_bal_accuracy[self.curr_epoch, modality] = bal_acc
        self.valid_mse[self.curr_epoch, modality] = mse
        self.valid_kld[self.curr_epoch, modality] = kld
        self.valid_losses[self.curr_epoch, modality] = loss

        
        # Plot loss and accuracy
        self.writer.add_scalar('validation_accuracy '+modname, correct, self.curr_epoch)
        self.writer.add_scalar('validation_bal_accuracy '+modname, bal_acc, self.curr_epoch)
        self.writer.add_scalar('validation_loss '+modname, loss * 1.0 / self.validset_size[modality], self.curr_epoch)
        self.writer.add_scalar('validation_mse '+modname, mse, self.curr_epoch)
        self.writer.add_scalar('validation KLD '+modname, kld, self.curr_epoch)
        # print('MSE : ', mse)
        # print('KLD : ', kld)
        
        # Plot confusion matrices
        plot_confusion_matrix(cm, location=self.expt_folder, title='VAE on Validation Set '+modname)
        
        # F1 Score
        f1_score = calculateF1Score(cm)
        self.writer.add_scalar('valid_f1_score'+modname, f1_score, self.curr_epoch)
        print('F1 Score'+modname+': ', calculateF1Score(cm))
        self.valid_f1_Score[self.curr_epoch, modality] =(f1_score)
        return correct, loss, f1_score
    
    # plot ROC curve
    # plotROC(cm, location=self.expt_folder, title='ROC Curve(Valid)')
    
    def test(self, test_loader, modality = 1):
        self.model.eval()
        print ('Test...')
        
        correct = 0
        test_losses = 0
        cm = np.zeros((num_classes, num_classes), int)
        if modality ==1:
            mod_name = 'target'
        else:
            mod_name = 'source'
        encoder_embedding = []
        classifier_embedding = []
        pred_labels = []
        act_labels = []
        class_prob = []
        
        preds_binary = []
        labels_binary = []
        
        pb = tqdm(total=len(test_loader))
        cm_per_class = np.zeros((3,2,2))
        reconstruction_loss = []
        for i, (images, labels, _) in enumerate(test_loader):
            with torch.no_grad():
                img = Variable(images).cuda()
                
            enc_emb, cls_emb, _, _, x_hat, p_hat = self.model(img)
            _, predicted = torch.max(p_hat.data, 1)
            labels = labels.view(-1, )
            correct += ((predicted.cpu() == labels).float().sum())
            reconstruction_loss.append((torch.mean((x_hat-img)**2)).cpu().detach().numpy().squeeze())
            cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
            if num_classes==3:
                labels2 = labels.clone().detach().numpy()
                cm_per_class += updatePairwiseCM(labels2, p_hat.data.cpu().numpy())
                preds_per_class = np.zeros((3,len(labels)))
                labels_per_class = np.zeros((3,len(labels)))
                for c in range(3):
                    labels2 = labels.clone().detach().numpy()
                    p_hat2 = p_hat.clone().detach().data.cpu().numpy()
                    
                    p_hat2[:,c] = -10
                    preds_per_class[c] = np.argmax(p_hat2,axis=1)
                    labels_per_class[c] = labels2
                    labels_per_class[c, labels2==c]=-10
                    if c==0:
                        preds_per_class[c]-=1
                        labels_per_class[c]-=1
                    elif c==1:
                        preds_per_class[c]/=2
                        labels_per_class[c]/=2                       
                
                    preds_per_class[c,labels2==c] = -10
                preds_binary.append(np.array(preds_per_class,dtype='int'))
                labels_binary.append(np.array(labels_per_class,dtype='int'))
            
            loss = self.classification_criterion(p_hat, Variable(labels).cuda())
            test_losses += loss.data.item()
            
            del img
            pb.update(1)
            p_hat = torch.exp(p_hat)
            
            encoder_embedding.extend(np.array(enc_emb.data.cpu().numpy()))
            classifier_embedding.extend(np.array(cls_emb.data.cpu().numpy()))
            pred_labels.extend(np.array(predicted.cpu().numpy()))
            act_labels.extend(np.array(labels.numpy()))
            class_prob.extend(np.array(p_hat.data.cpu().numpy()))
            
            
            
        pb.close()
        print('Reconstruction loss:',np.mean(np.array(reconstruction_loss)))
        
        encoder_embedding = np.array(encoder_embedding)
        classifier_embedding = np.array(classifier_embedding)
        pred_labels = np.array(pred_labels)
        act_labels = np.array(act_labels)
        class_prob = np.array(class_prob)
        
        if num_classes==3:
            classifications = ['MCI vs AD','CN vs AD', 'CN vs MCI']
            classific = ['MCI_AD','CN_AD', 'CN_MCI']
            classif = [['MCI','AD'],['CN','AD'], ['CN','MCI']]
            
            bal_acc_per_class = np.zeros(3)
            preds_binary = np.array(preds_binary).squeeze()
            labels_binary = np.array(labels_binary).squeeze()
            for c in range(3):
                print(classifications[c])
                print(cm_per_class[c])
                bal_acc_per_class[c] = calculate_balanced_acc(cm_per_class[c])
                print('Balanced acc:',bal_acc_per_class[c])
                
                plot_embedding(encoder_embedding, labels_binary[:,c].view(), preds_binary[:,c].view(), mode='tsne', location=self.expt_folder, title='encoder_embedding_test_'+mod_name+classific[c], target_names=classif[c])
                plot_embedding(encoder_embedding, labels_binary[:,c].view(), preds_binary[:,c].view(), mode='pca', location=self.expt_folder,title='encoder_embedding_test_'+mod_name+classific[c], target_names = classif[c])
                plot_embedding(classifier_embedding, labels_binary[:,c].view(), preds_binary[:,c].view(), mode='tsne', location=self.expt_folder,title='classifier_embedding_test_'+mod_name+classific[c],target_names=classif[c])
                plot_embedding(classifier_embedding, labels_binary[:,c].view(), preds_binary[:,c].view(), mode='pca', location=self.expt_folder,title='classifier_embedding_test_'+mod_name+classific[c],target_names=classif[c])
        
        
        correct /= self.testset_size[modality]
        test_losses /= self.testset_size[modality]
        bal_acc = calculate_balanced_acc(cm)
        
        print('Test Accuracy : %0.6f' % correct)
        print('Test Bal. Acc : %0.6f' % bal_acc)
        print('Test Losses : %0.6f' % test_losses)
        
        # Plot confusion matrices
        plot_confusion_matrix(cm, location=self.expt_folder, title='VAE on Test Set')
        
        # F1 Score
        print('F1 Score : ', calculateF1Score(cm))
        
        # plot PCA or tSNE
        encoder_embedding = np.array(encoder_embedding)
        classifier_embedding = np.array(classifier_embedding)
        pred_labels = np.array(pred_labels)
        act_labels = np.array(act_labels)
        class_prob = np.array(class_prob)
        
        plot_embedding(encoder_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
                       title='encoder_embedding_test_mod'+mod_name, target_names = name_classes)
        plot_embedding(encoder_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
                       title='encoder_embedding_test_mod'+mod_name, target_names = name_classes)
        
        plot_embedding(classifier_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
                       title='classifier_embedding_test_mod'+mod_name, target_names = name_classes)
        plot_embedding(classifier_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
                       title='classifier_embedding_test_mod'+mod_name,target_names =  name_classes)
        
        
        
        # plot ROC curve
        plotROC(act_labels, class_prob, location=self.expt_folder, title='ROC (VAE on Test Set)_mod'+mod_name)
    
    '''
    def test(self, test_loader):
        self.model.eval()
        print ('Test...')
        
        encoder_embedding = []
        classifier_embedding = []
        pred_labels = []
        act_labels = []
        
        pb = tqdm(total=len(self.valid_loader))
        
        for i, (images, labels) in enumerate(test_loader):
            for itr in range(20):
                print('iteration:',itr)
                print('label:',labels.cpu())
                
                img = Variable(images, volatile=True).cuda()
                enc_emb, cls_emb, _, _, _, p_hat = self.model(img)
                _, predicted = torch.max(p_hat.data, 1)
                labels = labels.view(-1, )
                
                del img
                pb.update(1)
                
                encoder_embedding.extend(np.array(enc_emb.data.cpu().numpy()))
                classifier_embedding.extend(np.array(cls_emb.data.cpu().numpy()))
                pred_labels.extend(np.array(predicted.cpu().numpy()))
                act_labels.extend(np.array(labels.numpy()))
                
            if i == 0:
                break
        
        pb.close()
        
        
        # plot PCA or tSNE
        encoder_embedding = np.array(encoder_embedding)
        classifier_embedding = np.array(classifier_embedding)
        pred_labels = np.array(pred_labels)
        act_labels = np.array(act_labels)
        
        plot_embedding(encoder_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
                       title='encoder_embedding_test')
        plot_embedding(encoder_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
                       title='encoder_embedding_test')
        
        plot_embedding(classifier_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
                       title='classifier_embedding_test')
        plot_embedding(classifier_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
                       title='classifier_embedding_test')
    '''