import math
import numpy as np
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()

        if nFeat is None:
            nFeat = 20
        
        if in_channels is None:
            in_channels = 3
        
        if out_channels is None:
            out_channels = 3

        
        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )
        ]

#该模块在本代码中不起作用，scale默认均为2
        for _ in range(1, int(np.log2(scale))): #当scale=2时，此循环不起作用，dual_block仅包括conv-LeakyReLU-conv;当scale=4时，此循环起作用，dual_block包括conv-LeakyReLU-conv-LeakyReLU-conv;
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x



class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class EcaLayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class MSRB(nn.Module):
    def __init__(self, conv, n_feat):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5
        #self.ca1 = EcaLayer(n_feat)
        #self.ca2 = EcaLayer(n_feat*2)
        #self.ca3 = EcaLayer(n_feat * 4)
        
        self.ca1 = CALayer(channel=n_feat)
        self.ca2 = CALayer(channel=n_feat*2)
        self.ca3 = CALayer(channel=n_feat * 4)

        self.conv_3_1 = conv(n_feat, n_feat, kernel_size_1)
        self.conv_3_2 = conv(n_feat * 2, n_feat * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feat, n_feat, kernel_size_2)
        self.conv_5_2 = conv(n_feat * 2, n_feat * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feat * 4, n_feat, 1, padding=0, stride=1)
        self.confusion1 = nn.Conv2d(n_feat * 4, n_feat * 4, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.conv_3_1(self.relu(self.conv_3_1(input_1)))
        output_5_1 = self.conv_5_1(self.relu(self.conv_5_1(input_1)))

        output_ca_31 = self.ca1(output_3_1)
        output_ca_51 = self.ca1(output_5_1)

        output_ca_31_1 = input_1+output_ca_31
        output_ca_51_1 = input_1+output_ca_51

        input_2 = torch.cat([output_ca_31_1, output_ca_51_1], 1)

        output_3_2 = self.conv_3_2(self.relu(self.conv_3_2(input_2)))
        output_5_2 = self.conv_5_2(self.relu(self.conv_5_2(input_2)))

        output_ca_32 = self.ca2(output_3_2)
        output_ca_52 = self.ca2(output_5_2)

        output_ca_32_2 = input_2 + output_ca_32
        output_ca_52_2 = input_2 + output_ca_52


        input_3 = torch.cat([output_ca_32_2, output_ca_52_2], 1)

        output = self.confusion1(self.relu(self.confusion1(input_3)))
        output_4 = self.ca3(output)
        output5 = input_3 + output_4

        output6 = self.confusion(output5)
        output6 += x
        return output6

class RCAB_ECA(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB_ECA, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(EcaLayer(channels=n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res