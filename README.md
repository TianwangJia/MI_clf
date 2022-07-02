# MI_clf

> 2022 Spring HUST AIA
>
> 人机交互课设

## 基于EEG的运动想象状态分类

### 数据采集过程

采集过程中，受试者坐在电脑前的椅子上。采集开始时，电脑屏幕上会出现一个固定的叉，提示对象准备，持续3s；然后，一个指向某一个方向的箭头作为视觉提示在屏幕上出现5s，在此期间，受试者根据箭头的方向执行特定的运动想象任务；然后，视觉提示从屏幕上消失，受试者短暂休息2s。紧接着下一个trial开始。

![image-20220702141023832](C:\Users\TWJia\AppData\Roaming\Typora\typora-user-images\image-20220702141023832.png)



### 数据集介绍

数据来自8个健康的受试者（训练受试者S1～S4，测试受试者S5～S8），每一个受试者执行两类运动想象任务：右手和双脚，脑电信号由一个13通道的脑电帽以512Hz的频率记录得到。我们提供了经过预处理后的数据：下采样到了250Hz，带通滤波至8-32Hz，划分每一次视觉提示出现后的0.5~3.5s之间的EEG信号作为一个trial。每个用户包含200个trial（右手和双脚各100个trial）。

数据以.npz和.mat格式提供, 包含:

- X: 预处理后的EEG信号, 维度: [trails * 通道 * 采样点]
- y: 类别标签向量



### Model

- Deep Learning
  - EA (Euclidean space data Alignment)
  - EEGNet
- Machine Learning
  - Bandpass filtering
  - Time division
  - EA
  - CSP (Common Spatial Pattern)
  - SVM
  - Model ensemble



### Result

- Deep Learning val acc

|                     | val acc |
| ------------------- | ------- |
| Not Across subjects | 81.5%   |
| Across subjects     | 75%     |

- Machine Learning val acc

| val subject            | 1      | 2      | 3      | 4      | mean   |
| ---------------------- | ------ | ------ | ------ | ------ | ------ |
| KFold cross validation | /      | /      | /      | /      | 75.01% |
| Across subjects        | 86.00% | 79.00% | 66.50% | 64.00% | 73.88% |

*KFold cross validation:*  take the mean of twenty five-fold cross validation

*Across subjects:* three subjects for training, one subject for validation