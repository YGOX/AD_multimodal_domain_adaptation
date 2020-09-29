import os
from configurations.paths import paths, file_names
import numpy as np
import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns
sns.set()


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def plotROC(result_table, location, title='ROC(BL on Test Set)'):

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')
    plt.title(title)
    fig.savefig(os.path.join(location, title + '.png'), bbox_inches="tight")


def main(fold=10, num_classes=2):

    timestr= '20200122-105742' #directory
    base_folder = paths['output']['pet_base_folder']
    expt_folder = base_folder + timestr + '/'
    fold_per = []
    result_table = pd.DataFrame(columns=['fold_idx', 'fpr', 'tpr', 'auc'])
    for i in np.arange(fold):
        fold_path = expt_folder + 'fold%d'%(i+1) + '/'
        file_path = fold_path + 'test_results.txt'
        h5_file_path= fold_path + 'test_scores.hdf5'
        f = open(file_path, "r")
        perfor=[]
        for line in f:
            # in python 2
            # print line
            # in python 3
            string= line.split()
            digit= re.findall("\d+\.\d+", string[0])
            perfor.append(float(digit[0]))
        fold_per.append(np.expand_dims(perfor,0))

        with h5py.File(h5_file_path, 'r') as f:
            # List all groups
            scores= np.array(list(f['scores']))
            labels= np.array(list(f['labels']))

        onehot = np.zeros((labels.shape[0], num_classes), dtype=int)
        for item in range(labels.shape[0]):
            onehot[item, labels[item]] = 1
        y_true = onehot

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for c in range(num_classes):
            fpr[c], tpr[c], _ = roc_curve(y_true[:, c], scores[:, c])
            roc_auc[c] = auc(fpr[c], tpr[c])
        result_table = result_table.append({'fold_idx': 'fold%d'%(i+1),
                                            'fpr': fpr[1],
                                            'tpr': tpr[1],
                                            'auc': roc_auc[1]}, ignore_index=True)
    result_table.set_index('fold_idx', inplace=True)

    plotROC(result_table, location=expt_folder)
    matrix= np.concatenate(fold_per)
    mean= np.mean(matrix, 0)
    std= np.std(matrix, 0)
    print("mean--acc:%0.6f\n loss:%0.6f\n f1:%0.6f\n recall:%0.6f"
               "\n precision:%0.6f\n sensitivity:%0.6f\n specificity:%0.6f" %
               (mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], mean[6]))
    print("std--acc:%0.6f\n loss:%0.6f\n f1:%0.6f\n recall:%0.6f"
          "\n precision:%0.6f\n sensitivity:%0.6f\n specificity:%0.6f" %
          (std[0], std[1], std[2], std[3], std[4], std[5], std[6]))


if __name__ == '__main__':
    main()
