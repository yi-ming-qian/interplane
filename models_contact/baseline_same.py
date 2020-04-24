import torch
import torch.nn as nn

from models_contact import resnet_scene as resnet


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
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.avgpool(x5)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x1, x2, x3, x4, x5, x


class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()

        orig_resnet = resnet.__dict__[cfg.arch](pretrained=cfg.pretrained)
        self.backbone = ResNet(orig_resnet, cfg.input_channel)

        self.relu = nn.ReLU(inplace=True)

        channel = 64
        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv2 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv1 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv0 = nn.Conv2d(channel, channel, (1, 1))

        # lateral
        self.c5_conv = nn.Conv2d(2048, channel, (1, 1))
        self.c4_conv = nn.Conv2d(1024, channel, (1, 1))
        self.c3_conv = nn.Conv2d(512, channel, (1, 1))
        self.c2_conv = nn.Conv2d(256, channel, (1, 1))
        self.c1_conv = nn.Conv2d(128, channel, (1, 1))

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)

        # line segment prediction
        self.line_prob = nn.Conv2d(channel, 1, (1, 1), padding=0)

    def top_down(self, x):
        c1, c2, c3, c4, c5 = x

        p5 = self.relu(self.c5_conv(c5))
        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(c4))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(c3))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(c2))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(c1))

        p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0#, p1, p2, p3, p4, p5

    def forward(self, x):
        #plane_common = x[:,4:5,:,:]
        # bottom up
        c1, c2, c3, c4, c5, contact = self.backbone(x)
        # top down
        p0 = self.top_down((c1, c2, c3, c4, c5))

        # output
        prob = self.line_prob(p0)
        #prob += plane_common
        return contact, torch.sigmoid(prob)
