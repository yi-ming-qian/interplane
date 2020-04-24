import torch
import torch.nn as nn

from models_angle import resnet_scene as resnet


class ResNet(nn.Module):
    def __init__(self, orig_resnet, input_ch):
        super(ResNet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = nn.Conv2d(input_ch, 64, kernel_size=3, stride=2,
                     padding=1, bias=False)
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1

        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2

        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.avgpool = orig_resnet.avgpool
        self.fc = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()

        orig_resnet = resnet.__dict__[cfg.arch](pretrained=cfg.pretrained)
        self.backbone = ResNet(orig_resnet, cfg.input_channel)


    def forward(self, x):
        x = self.backbone(x)
        return x
