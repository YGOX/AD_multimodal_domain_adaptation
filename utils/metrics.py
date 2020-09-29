import numpy as np
from configurations.modelConfig import num_classes

def updateConfusionMatrix(actual_labels, predicted_labels, n_class = None):
    """
    updates confusion matrix after every minibatch
    :param actual_labels:
    :param predicted_labels:
    :return:
    cnfusion matrix
    """
    if n_class is None:
        n_class = num_classes
    cm = np.zeros((n_class, n_class), int)
    for (al, pl) in zip(actual_labels, predicted_labels):
        #print(al, pl)
        cm[al, pl] += 1
    #print(cm)
    return cm


def updatePairwiseCM(labels,p_hat):
    
    cm_per_class = np.zeros((3,2,2),int)
    for c in range(3):
        ind_of_interest = labels != c
        predicted = np.argmax(np.delete(p_hat,c,1), 1)
        labels2 = np.copy(labels)
        if c==0:
            labels2-=1
        elif c==1:
            labels2[labels==2] = 1
        cm_per_class[c] = updateConfusionMatrix(labels2[ind_of_interest], predicted[ind_of_interest],2)
        
    return cm_per_class


#updateConfusionMatrix([0,2,0],[0,0,1])

def calculateF1Score(cm):
    if num_classes==2:
        true_negative = cm[0,0]
        false_negative = cm[1,0]

        true_positive = cm[1,1]
        false_positive = cm[0,1]

        precision = (true_positive * 1.) / (true_positive + false_positive)
        recall = (true_positive * 1.0) / (true_positive + false_negative)
        
        f1_score = 2.0 * (recall * precision) / (recall + precision)
        
    elif num_classes==3:
        f1_perclass=np.zeros(num_classes)
        nanflag = [False, False]
        for i in range(num_classes):
            if np.sum(cm[i,:])>0:
                precision = cm[i,i]/np.sum(cm[i,:])
            else:
                precision = 0
                nanflag[0] = True
            if np.sum(cm[:,i])>0:
                recall = cm[i,i]/np.sum(cm[:,i])
            else:
                recall = 0
                nanflag[1] = True
            if nanflag[0] and nanflag[1]:
                f1_perclass[i] = None
            elif nanflag[0] or nanflag[1]:
                f1_perclass[i]=0
            else:
                f1_perclass[i] = 2.0 * (recall * precision) / (recall + precision)
        
        f1_score=np.nanmean(f1_perclass)
        
    return f1_score

def calculate_balanced_acc(cm):
    L = len(cm)
    bal_acc = np.zeros(L)
    for i in range(L):
        if np.sum(cm[i,:])!=0:
            bal_acc[i] = cm[i,i]/np.sum(cm[i,:])
        else:
            bal_acc[i] = None  #accounts for some batches not having certain classes
    return np.nanmean(bal_acc)