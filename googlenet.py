import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10

# d定义一个卷积+ReLu的函数，[输入通道， 输出通道， 卷积核大小， stride， paading]
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding ),
        nn.BatchNorm2d(out_channel, eps=1e-3),  # 批归一化处理
        nn.ReLU(True)
    )
    return layer

# 定义一个inception模块类，模块分为四个部分，1.1*1卷积， 2 1*1卷积 + 3*3卷积， 3.1*1卷积 + 5*5卷积， 4.3*3最大池化 + 1*1卷积
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        # print("f1:", f1.shape)
        f2 = self.branch3x3(x)
        # print("f2:", f2.shape)
        f3 = self.branch5x5(x)
        # print("f3:", f3.shape)
        f4 = self.branch_pool(x)
        # print("f4:", f4.shape)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        # print("out:", output.shape)
        # 输出会发现图像的大小没有改变，通道数增加了
        return output
# 一个测试
test_net = inception(3, 64, 48, 64, 64, 96, 32)
test_x = Variable(torch.zeros(1, 3, 96, 96))
test_y = test_net(test_x)

# 定义一个googlenet网络类，让很多个inception从串联起来
class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose =verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channel=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )

        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )

        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )

        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.block1(x)
        print("block1:", x.shape)

        x = self.block2(x)
        print("block2:", x.shape)

        x = self.block3(x)
        print("block3:", x.shape)

        x = self.block4(x)
        print("block4:", x.shape)

        x = self.block5(x)
        print("block5:", x.shape)

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

