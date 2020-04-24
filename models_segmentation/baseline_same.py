import torch
import torch.nn as nn

from models_segmentation import resnet_scene as resnet


class ResNet(nn.Module):
    def __init__(self, orig_resnet):
        super(ResNet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=2,
                     padding=1, bias=True)
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

        self.mp1 = nn.Conv2d(256, 128, (1, 1))
        self.mp2 = nn.Conv2d(512, 256, (1, 1))
        self.mp3 = nn.Conv2d(1024, 512, (1, 1))
        #self.avgpool = orig_resnet.avgpool
        #self.fc = nn.Linear(2048, 2)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

    def aggregate(self, x):
        bn = x.size(0)
        x_n = torch.zeros_like(x)
        if bn>1:
            for i in range(bn):
                others = torch.cat((x[0:i,:,:,:], x[i+1:bn,:,:,:]))
                x_n[i] = torch.max(others,0)[0]
        return x_n

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)

        x_n = self.mp1(torch.cat((x, self.aggregate(x)),1))
        x2 = self.layer1(x_n)

        x_n = self.mp2(torch.cat((x2, self.aggregate(x2)),1))
        x3 = self.layer2(x_n)

        x_n = self.mp3(torch.cat((x3, self.aggregate(x3)),1))
        x4 = self.layer3(x_n)

        x5 = self.layer4(x4)

        # x = self.avgpool(x5)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x1, x2, x3, x4, x5


class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()

        orig_resnet = resnet.__dict__[cfg.arch](pretrained=False)
        self.backbone = ResNet(orig_resnet)

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
        self.mask1 = nn.Conv2d(channel, 1, (1, 1), padding=0)
        self.mask2 = nn.Conv2d(channel, 1, (1, 1), padding=0)

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
        c1, c2, c3, c4, c5 = self.backbone(x)
        # top down
        p0 = self.top_down((c1, c2, c3, c4, c5))

        # output
        prob1 = self.mask1(p0)
        #prob2 = self.mask2(p0)
        #prob += plane_common
        return torch.sigmoid(prob1)

class RefinementBlockMask(nn.Module):
    def __init__(self):
        super(RefinementBlockMask, self).__init__()

        use_bn = False
        self.conv_0 = ConvBlock(4, 32, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv_1 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, use_bn=use_bn)       
        self.conv_1_1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv_2 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, use_bn=use_bn)
        self.conv_2_1 = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1, use_bn=use_bn)

        self.up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
        self.up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
        self.pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                 torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))

        # self.global_up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
        # self.global_up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
        # self.global_pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
        #                                 torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))


    def accumulate(self, x):
        return torch.cat([x, (x.sum(0, keepdim=True) - x) / max(len(x) - 1, 1)], dim=1)
       
    def forward(self, x_0):
        x_0 = self.conv_0(x_0)
        x_1 = self.conv_1(self.accumulate(x_0))
        x_1 = self.conv_1_1(self.accumulate(x_1))
        x_2 = self.conv_2(self.accumulate(x_1))
        x_2 = self.conv_2_1(self.accumulate(x_2))

        y_2 = self.up_2(x_2)
        y_1 = self.up_1(torch.cat([y_2, x_1], dim=1))
        y_0 = self.pred(torch.cat([y_1, x_0], dim=1))

        # global_y_2 = self.global_up_2(x_2.mean(dim=0, keepdim=True))
        # global_y_1 = self.global_up_1(torch.cat([global_y_2, x_1.mean(dim=0, keepdim=True)], dim=1))
        # global_mask = self.global_pred(torch.cat([global_y_1, x_0.mean(dim=0, keepdim=True)], dim=1))

        # y_0 = torch.cat([global_mask[:, 0], y_0.squeeze(1)], dim=0)
        return torch.sigmoid(y_0)

class ConvBlock(torch.nn.Module):
    """The block consists of a convolution layer, an optional batch normalization layer, and a ReLU layer"""
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, mode='conv', use_bn=True):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        if mode == 'conv':
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv':
            self.conv = torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        elif mode == 'upsample':
            self.conv = torch.nn.Sequential(torch.nn.Upsample(scale_factor=stride, mode='nearest'), torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=not self.use_bn))
        elif mode == 'conv_3d':
            self.conv = torch.nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv_3d':
            self.conv = torch.nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        else:
            print('conv mode not supported', mode)
            exit(1)
            pass
        if '3d' not in mode:
            self.bn = torch.nn.BatchNorm2d(out_planes)
        else:
            self.bn = torch.nn.BatchNorm3d(out_planes)
            pass
        self.relu = torch.nn.ReLU(inplace=True)
        return

    def forward(self, inp):
        if self.use_bn:
            return self.relu(self.bn(self.conv(inp)))
        else:
            return self.relu(self.conv(inp))
