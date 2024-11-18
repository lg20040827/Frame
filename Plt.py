import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import LSTM
import SGD



def plot_roc_curve():
    fpr, tpr, auc = LSTM.Get_LSTM_AUC()
    fpr1, tpr1, auc1 = SGD.Get_SGD_AUC()
    print(fpr, tpr, auc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='SGD ROC curve (area = %.4f)' % auc1)
    plt.plot(fpr, tpr, color='blue', lw=2, label='LSTM ROC curve (area = %.4f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_prc_curve():
    precision, recall, pr_auc = LSTM.Get_LSTM_PRC()
    precision1, recall1, pr_auc1 = SGD.Get_SGD_PUC()
    plt.figure()
    plt.plot(recall1, precision1,lw=2, color='blue',label='SGD PRC curve (area = %.4f)' % pr_auc1)
    plt.plot(recall, precision, lw=2, color='red',label='LSTM PRC curve (area = %.4f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.title(f'PR Curve')
    plt.show()