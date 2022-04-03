# PSM-Net
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad,
                                   dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class PSMNet(nn.Module):
    def __init__(self, feature_dim=32):
        super(PSMNet, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3,  32, 3, 2, 1, 1), nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32,  3,  1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64,  16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3,  1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3,  1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, feature_dim, kernel_size=1, padding=0, stride=1, bias=False))

        # initialization for the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output         = self.firstconv(x)
        output_layer1  = self.layer1(output)
        output_raw     = self.layer2(output_layer1)
        output         = self.layer3(output_raw)
        output_skip    = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, size=(output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, size=(output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, size=(output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, size=(output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1) 
        output_feature = self.lastconv(output_feature) 
        return output_feature

