import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import mne
import random

from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.svm import SVC

import sys
import pickle

sys.path.append("../") #设置自定义包的搜索路径
from datasets import *

# 随机打乱数据集, 划分训练验证集, 进行K折验证, 得到train, test acc
## K折交叉验证
def KFold_Cross_Val(split_num, data, clf):
    kf = KFold(n_splits=split_num, shuffle=True)
    X=data.data
    y=data.label
    train_acc = 0
    test_acc = 0

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]

        clf.fit(x_train, y_train.squeeze())

        y_train_pred = clf.predict(x_train)
        tmp_train_acc = np.sum(y_train_pred==y_train.squeeze())/len(y_train_pred)
        y_pred = clf.predict(x_test)
        tmp_test_acc = np.sum(y_pred == y_test.squeeze())/len(y_pred)
        train_acc += tmp_train_acc
        test_acc += tmp_test_acc

        print('idx: {}, train acc: {:.4f}%, loss_val: {:.4f}%'.format(i+1,tmp_train_acc*100,tmp_test_acc*100))
    print('KFold over!')

    return train_acc/split_num, test_acc/split_num

## 训练部分, 进行20次5折交叉验证
def train_K_Fold():
    seed = random.randint(1,100)
    csp_num = 5
    data =  MIDataset(root='../data/train/', random_state=seed, is_car = False, is_csp=True,csp_num = csp_num,is_ea=True, is_bandpass=True, 
    is_eegnet=False,low=8, high=16, start=0, end=350)
    clf = SVC(kernel = 'rbf')

    split_num = 5 
    train_acc = 0
    test_acc = 0
    for i in range(20): 
        print('KFold:{}'.format(i))
        tmp_train_acc, tmp_test_acc = KFold_Cross_Val(split_num, data, clf)
        train_acc += tmp_train_acc
        test_acc += tmp_test_acc
    
    train_acc /= 20
    test_acc /= 20

    print('csp num: {}, split num: {}'.format(csp_num, split_num))

    return train_acc, test_acc


# 划分新用户为验证集, 进行四次训练验证, 输出 train test acc
## 划分新用户, 训练验证, 输出
def new_subject_val(data, clf):
    X_train = data.X_train
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    clf.fit(X_train, y_train.squeeze())

    y_train_pred = clf.predict(X_train)
    tmp_train_acc = np.sum(y_train_pred==y_train.squeeze())/len(y_train_pred)

    y_pred = clf.predict(X_test)
    tmp_test_acc = np.sum(y_pred==y_test.squeeze())/len(y_pred)

    return tmp_train_acc, tmp_test_acc


## 训练部分, 每个用户都被划到验证集一次
def train_new_subject(savemodel=False):
    train_acc = []
    test_acc = []
    seed = random.randint(1,100)
    csp_num = 5

    for i in range(4):
        clf = SVC(kernel = 'rbf')
        data = MIDataset(root='../data/train/', random_state=seed, is_car = False, is_csp=True,csp_num = csp_num,is_ea=True, is_bandpass=True, 
        is_eegnet=False,low=8, high=16, start=0, end=350, val_user=i+1)
        tmp_train_acc, tmp_test_acc = new_subject_val(data, clf)
        train_acc.append(tmp_train_acc)
        test_acc.append(tmp_test_acc)
        
        ### 保存模型
        if savemodel == True: 
            pickle.dump(clf, open("clf"+str(i+1)+".dat", 'wb'))
    
    return train_acc, test_acc



if __name__ == '__main__':
    # K Fold验证
    KFold_train_acc, KFold_test_acc = train_K_Fold()

    # 新用户验证
    savemodel = True
    news_train_acc, news_test_acc = train_new_subject(savemodel)

    # 打印输出
    print('KFold: ')
    print('train acc: {:.4f}%, test acc: {:.4f}%'.format(KFold_train_acc*100, KFold_test_acc*100))
    print('New subje Val: ')
    print('train acc:{}, mean: {:.4f}%'.format(news_train_acc, np.mean(news_train_acc)*100))
    print('test acc:{}, mean: {:.4f}%'.format(news_test_acc, np.mean(news_test_acc)*100))
