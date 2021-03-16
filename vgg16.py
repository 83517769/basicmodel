from torch import nn
import torch
import torchvision
import torch.utils.model_zoo as model_zoo
"""
Created on 22:04:10 2021/3/5

@author: (ATRer)hwh
"""
__all__ = [
     'vgg16', 'Vgg16'
]

model_urls = {

    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


class Vgg16(nn.Module):

    def __init__(self,):
        super(Vgg16, self).__init__()
        net = torchvision.models.vgg16()
        net.classifier = nn.Sequential(
           nn.Linear(512*1*1, 512),
           nn.ReLU(True),
           nn.Dropout(),
           nn.Linear(512, 256),
           nn.ReLU(True),
           nn.Dropout(),
           nn.Linear(256,10))
        self.feature = net.features
        self.avgpool = net.avgpool
        self.classifier1 = net.classifier
        # self.body = net
        # self.classifier = nn.Sequential(
        #    nn.Linear(512, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096,10)
        # )

    def forward(self, x):
        x = self.feature(x)
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.classifier1(x)
        return x


def vgg16(pretrained=False, model_root=None,**kwargs):

    model = Vgg16(**kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['vgg16'], model_root)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        print("model Alraedy !")
        model.load_state_dict(model_dict)
#        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']), strict=False)
    return model










