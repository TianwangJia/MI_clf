import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 初始化网络权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data)
        # 这里如果用 torch.nn.init.xavier_normal_会因为torchtext版本太高，不支持一维的词向量，仅仅支持二维以上的而报错
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)


class EEGNet(nn.Module):

    def __init__(self,
                channels=6,
                F1=8,
                D=2,
                F2=16,
                time=750,
                class_num=2,
                drop_out=0.25) -> None:
        super(EEGNet, self).__init__()
        self.drop_out = drop_out
        self.class_num = class_num
        self.c = channels
        self.time = time  # 采样点
        self.F1 = F1  # number of temporal filters
        self.D = D  #D = depth multiplier (number of spatial filters)
        #用于DepthwiseConv2d中每层的空间滤波器数量
        self.F2 = F2  # number of pointwise filters

        # 输入的每个trail为(13,750)数据, 13通道做行数, 所以以2D矩阵来说, 1通道

        # block1
        # in (1, C, T) out(F1, C, T)
        self.block_1 = nn.Sequential(
            # 填充0, 保持维度不变, 再卷积, left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, self.F1, (1, 64)),
            nn.BatchNorm2d(self.F1))

        # block2 DepthwiseConv2d
        # in (F1, C, T) out (D*F1, 1, T//4)
        self.block_2 = nn.Sequential(
            nn.Conv2d(self.F1,
                      self.D * self.F1,
                        kernel_size=(self.c, 1),
                      groups=self.F1), nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(self.drop_out))

        # block3 SeparableConv2d
        # in (D*F1, 1, T//4) out (F2, 1, T//32)
        # 目前 F2 = D*F1
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(self.D * self.F1,
                      self.D * self.F1,
                        kernel_size=(1, 16),
                      groups=self.D * self.F1),  # Depthwise Convolution
            nn.Conv2d(self.D * self.F1, self.F2,
                        kernel_size=(1, 1)),  # Pointwise Convolution
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out))

        self.fc = nn.Linear((self.F2 * (self.time // 32)), self.class_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return out


if __name__ == '__main__':
    x = torch.randn(32, 1, 13, 750)

    model = EEGNet()

    out = model(x)
    print(out.shape)
