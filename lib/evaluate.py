""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from lib import pytorch_ssim
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

def get_abnormal_num(labels, abnormal_tar):
    abn_num = 0
    for l in labels:
        if l == abnormal_tar:
            abn_num+=1
    return abn_num

def get_recall(fpr, tpr):
    normal_recall = 1 - fpr
    abnormal_recall = tpr
    return normal_recall, abnormal_recall

def get_precision(fpr, tpr, nor_num, abn_num):
    # normal_precision = []
    # abnormal_precision = []
    # for i in range(0, len(fpr)):
    normal_precision = (nor_num * (1-fpr)) / ((nor_num * (1-fpr))+((abn_num * (1-tpr))))
    abnormal_precision = (abn_num*tpr)/((nor_num*fpr)+(abn_num*tpr))
    # normal_precision.append((nor_num * (1-fpr[i])) / ((nor_num * (1-fpr[i]))+((abn_num * (1-tpr[i])))))
    # abnormal_precision = (abn_num*tpr[i])/((nor_num*fpr[i])+(abn_num*tpr[i]))
    return normal_precision, abnormal_precision

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()
    total_num = len(labels)
    abn_num = get_abnormal_num(labels, abnormal_tar=1.)

    # True/False Positive Rates.
    # fpr 代表 0 不對的 (類別0 的 1-recall)
    # tpr 代表 1 對的 (類別1 的 recall)
    fpr, tpr, thre = roc_curve(labels, scores)
    #print('labels\n', labels)
    #diprint('scores\n', scores)
    # print('len fpr:', len(fpr))
    # print('len tpr:', len(fpr))
    # print('len thre:', len(thre))
    # print('fpr ', fpr)
    #print('tpr ', tpr)
    #print('thre ', thre)
    nor_rec, abn_rec = get_recall(fpr, tpr)
    nor_pre, abn_pre = get_precision(fpr,tpr, total_num-abn_num, abn_num)
    df_dict = {'fpr':fpr , 'tpr': tpr, 'threhold': thre, 'normal recall':nor_rec, 'normal precision': nor_pre, 'abnormal recall': abn_rec, 'abnormal precision': abn_pre}
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(os.path.join(saveto, 'fpr_tpr_thre.csv'))
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    #
    # if saveto:
    #     plt.figure()
    #     lw = 2
    #     plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
    #     plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
    #     plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic')
    #     plt.legend(loc="lower right")
    #     plt.savefig(os.path.join(saveto, "ROC.pdf"))
    #     plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap

def ssim_score(input, target):
    return 1-pytorch_ssim.ssim(input, target, window_size=11, size_average=False)