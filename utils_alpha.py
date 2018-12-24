'''
utils functions for all DS project

Ming Li @ 01/12/2016
'''
import sklearn
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse
import pandas as pd



def alpha_metrics4_sklearn(true_value, predicted, printout = True):
    '''
    Smallest function to get the metrics of predicted problem

    :param true_value: list or ndarray
    :param predicted:  list or ndarray
    :return:  dict with metrics
    '''
    metric_dict = {}
    p = metrics.precision_score(true_value, predicted)
    r = metrics.recall_score(true_value, predicted)
    f = metrics.f1_score(true_value, predicted)
    auc = metrics.roc_auc_score(true_value, predicted)
    metric_dict.update({'precision' : p})
    metric_dict.update({'recall': r})
    metric_dict.update({'F1': f})
    metric_dict.update({'AUCscore': auc})
    if printout == True:
        print(metric_dict)
    else:
        return metric_dict

def alpha_metric_roccurve(true_value, predicted):
    fpr, tpr, _ = metrics.roc_curve(true_value, predicted)
    plt.figure()
    lw = 2
    auc = alpha_metrics4_sklearn(true_value, predicted, printout = False)['AUCscore']
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("ROC Curve.png")
    plt.show()

def alpha_metric_cm(true_value, predicted, printout = True):
    cm = metrics.confusion_matrix(true_value, predicted)
    df_cm = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'])
    df_cm.index = ['Reality 0', 'Reality 1']
    df_cm = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'])
    df_cm.index = ['Reality 0', 'Reality 1']
    df_cm_percent = df_cm
    df_cm_percent['Predicted 0'] = 100 * df_cm_percent['Predicted 0'] / len(mask_image)
    df_cm_percent['Predicted 1'] = 100 * df_cm_percent['Predicted 1'] / len(mask_image)

    if printout == True:
        print('Confusion Matrix:')
        print(df_cm)
        print(df_cm_percent)
    else:
        return df_cm, df_cm_percent