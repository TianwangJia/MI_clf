from enum import Flag
from re import X
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from scipy.linalg import fractional_matrix_power
from mne.decoding import CSP
import scipy.signal as signal

import numpy as np


class MIDataset(Dataset):

    def __init__(self,
                random_state,
                root='./data/train/',
                mode='train',
                is_car = False,
                is_ea=False,
                is_csp=False,
                csp_num = 6,
                is_eegnet=True,
                is_bandpass=False,
                low = 8, 
                high = 16,
                start = 0,
                end = 750,
                val_user=0):
        self.root = root
        self.mode = mode
        self.random_state = random_state
        self.is_car = is_car # 是否使用重参考
        self.is_ea = is_ea  # 是否使用ea
        self.is_csp = is_csp  # 是否进行csp
        self.is_eegnet = is_eegnet  # 是否使用eegnet
        self.is_bandpass = is_bandpass # 是否使用带通滤波
        self.val_user = val_user # 随机打乱划分数据集验证集, 或者挑选用户划分验证集
        self.csp_num = csp_num # csp降维后数据维度

        x1 = []
        y1 = []
        for i in range(4):
            temp = np.load(self.root + 'S' + str(i + 1) + '.npz')
            x1.append(temp['X'])
            y1.append(temp['y'])

        x = np.array(x1)  #(4, 200, 13, 750)
        y = np.array(y1)  #(4, 200)

        # reshape x: (N,H,W)  y: (N,1) 
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) # (800, 13, 750)
        y = y.reshape(-1, 1)  # (800, 1)

        # 带通滤波
        if self.is_bandpass == True:
            x = self.bandpass(x, low, high)

        # 划分时间段
        x = x[:,:,start:end]

        # 重参考
        if self.is_car == True:
            x = self.car(x)

        # EA
        if self.is_ea == True:
            x = self.ea(x)

        # CSP
        if self.is_csp == True:
            self.csp, x = self.csp_decode(x,y)  


        # 没有分割的数据, 用于交叉验证等
        self.data = x
        self.label = y
        
        if self.is_eegnet == True: 
            # reshape (N,C,H,W)
            x = x.reshape( x.shape[0], 1, x.shape[1], x.shape[2])  # (N, C, H, W) 使其变成类图片样式, C=1, 原先的13channels做H
            y = to_categorical(y) # 转换为one-hot编码
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).type(torch.FloatTensor)
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).type(torch.FloatTensor)
            # x (800, 1, 13, 750) y (800, 2)

        # 划分训练集, 验证集
        if self.val_user==0:
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=self.random_state)
        else:
            index = [1,2,3,4]
            index.remove(self.val_user)
            train_index = index
            val_index = self.val_user
            X_train = np.concatenate((x[(train_index[0]-1)*200:train_index[0]*200], x[(train_index[1]-1)*200:train_index[1]*200], x[(train_index[2]-1)*200:train_index[2]*200]),axis=0)
            y_train = np.concatenate((y[(train_index[0]-1)*200:train_index[0]*200], y[(train_index[1]-1)*200:train_index[1]*200], y[(train_index[2]-1)*200:train_index[2]*200]), axis=0)
            X_test = x[(val_index-1)*200:val_index*200]
            y_test = y[(val_index-1)*200:val_index*200]
            

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.X_train[index], self.y_train[index]
        elif self.mode == 'test':
            return self.X_test[index], self.y_test[index]

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train)
        elif self.mode == 'test':
            return len(self.X_test)

    # common average referencing
    def car(self, x):
        mean = np.mean(x, axis=1)
        for i in range(x.shape[0]):
            x[i] = x[i]-mean[i]
        return x

    # EA, 欧式对齐
    def ea(self, x):
        # tensor
        if isinstance(x, np.ndarray):
                x = x.copy()
                x = torch.from_numpy(x).type(torch.FloatTensor)

        for i in range(4):
            s = x[i * 200:(i + 1) * 200]
            cov = torch.zeros((s.shape[0], s.shape[1], s.shape[1]))
            for j in range(s.shape[0]):
                cov[j] = torch.cov(s[j])
            refEA = torch.mean(cov, 0)
            sqrtRefEA = torch.from_numpy(fractional_matrix_power(refEA,-0.5)).type(torch.FloatTensor)
            # sqrtRefEA = torch.from_numpy(fractional_matrix_power(refEA,-0.5)).type(torch.FloatTensor)+(0.00000001)*torch.eye(s.shape[1])
            x[i*200:(i+1)*200] = torch.matmul(sqrtRefEA, x[i*200:(i+1)*200])

        return np.array(x).astype('float64') # 需要修改为float64, 否则ea后加csp报错
    
    # CSP提取特征
    def csp_decode(self, x, y):
        if y.ndim > 1:
            y = y.squeeze()
        self.csp = CSP(n_components=self.csp_num, transform_into='average_power')
        x_tran = self.csp.fit_transform(x, y)

        return self.csp, x_tran
    
    # 带通滤波
    def bandpass(self,x, low=8, high=16):
        b,a = signal.butter(N=4, Wn=(low,high), btype='bandpass', fs=250)
        x_filtered = signal.filtfilt(b,a,x,axis=2)
        return x_filtered




if __name__ == '__main__':
    data = MIDataset(random_state=23,is_car=False,is_ea=True,is_csp=False, is_bandpass=True, is_eegnet=False,  val_user=0,start=0, end=750)
    # data = MIDataset(random_state=23, is_ea = True, is_csp=True, is_eegnet=False)
    x_rand = np.ones((800,13,750))
    # new = data.csp.transform(x_rand)
    # print(new.shape)

    print('X_train.shape: ', data.X_train.shape)
    print('X_train[0]: ', data.X_train[0])
    # print('y_train.shape: ', data.y_train.shape)
    # print(type(data.y_train))
    # print(data.X_test.shape)
    # print(data.y_test.shape)
    # print(data.X_test[0])
    # print(type(data.X_train))
