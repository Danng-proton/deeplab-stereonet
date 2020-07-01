# coding=utf-8
# ------------------------------------------------------------------------------
# Copyright (c) NKU
# Licensed under the MIT License.
# Written by Xuanyi Li (xuanyili.edu@gmail.com)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import math
import pdb
#added
# from networks.resample2d_package.modules.resample2d import Resample2d
#from networks.channelnorm_package.modules.channelnorm import ChannelNorm

#from networks import FlowNetC
# About deeplab
affine_par = True

from networks.submodules import *
'Parameter count = 162,518,834'

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):
    
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation),
       nn.BatchNorm2d(out_channel))

def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride),
       nn.BatchNorm3d(out_channel))

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)

        # out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        ### bug?
        out = x + out
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
	        padding = 2
        elif dilation_ == 4:
	        padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
#        print '**********************************classifier******************************'
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
#            print '*************conv2d', i+1, ':', self.conv2d_list[i+1](x).shape, '****************'
            out += self.conv2d_list[i+1](x)
        return out



class FeatureExtraction(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.downsample = nn.ModuleList()# 不太懂
        in_channel = 3
        out_channel = 32
        for _ in range(k):# k代表下采样次数
            self.downsample.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=5,
                    stride=2,
                    padding=2))
            in_channel = out_channel
            out_channel = 32
        self.residual_blocks = nn.ModuleList()
        for _ in range(6):#6个残差块
            self.residual_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, rgb_img):
        output = rgb_img
        for i in range(self.k):
            output = self.downsample[i](output)
        for block in self.residual_blocks:
            output = block(output)
        return self.conv_alone(output)

class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv2d_feature = nn.Sequential(
            convbn(in_channel, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 8 , 1 , 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di))
                
        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):
        output = torch.unsqueeze(low_disparity, dim=1)
        twice_disparity = F.interpolate(
            output,
            size = corresponding_rgb.size()[-2:],
            mode='bilinear',
            align_corners=False)
        if corresponding_rgb.size()[-1]/ low_disparity.size()[-1] >= 1.5:
            twice_disparity *= 2   
            # print(corresponding_rgb.size()[-1]// low_disparity.size()[-1])
        output = self.conv2d_feature(
            torch.cat([twice_disparity, corresponding_rgb], dim=1))
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)
        
        return nn.ReLU(inplace=True)(torch.squeeze(
            twice_disparity + self.conv2d_out(output), dim=1))

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = torch.FloatTensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out



class StereoNet(nn.Module):
    def __init__(self, k, r, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k)
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(r):
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))
        self.rgb_max = 255.
        self.inplanes = 64
        self.conv_deep = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation__ = 4)

        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],(self.maxdisp + 1) // pow(2, self.k))
    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))
        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def _pred_scoremap(self,score_map):
        cost = self.conv3d_alone(score_map)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = disparityregression(disp)(pred)
        # print ("pred..........",pred.shape)
        return pred

    def forward(self, left, right):
        disp = (self.maxdisp + 1) // pow(2, self.k)
        #       **********************************deeplab******************************
        #这里确定一个空的总scoremap的形状
        batch_id,channal,h,w = left.shape
        # print("left shape",left.shape)
        left_change = torch.zeros(batch_id,channal,1,h,w)
        # left_upsidedown=left.permute(3,2,1,0)
        left_change[:, :, 0, :, :] = left
        rgb_mean = left_change.contiguous().view(left_change.size()[:2] + (-1,)).mean(dim=-1).view(left_change.size()[:2] + (1, 1, 1, ))
        # print("rgb_mean:",rgb_mean.shape)
        # print("left_change:", left_change.shape)
        # print("rgb_max:",self.rgb_max)
        optimized_left = (left_change - rgb_mean) / self.rgb_max
        # print("optimized_left:", optimized_left.shape)
        input_left = optimized_left[:,:,0,:,:]  # take left image as the input of deeplab input
        # print("input_left",input_left.shape)
        xx = self.conv_deep(input_left.cuda())
        xx = self.bn1(xx)
        xx = self.relu(xx)
        xx = self.maxpool(xx)
        xx = self.layer1(xx)
        # print( '**********************************layer1******************************')
        # print( '*************layer1:', xx.shape, '****************')
        xx = self.layer2(xx)
        # print ('**********************************layer2******************************')
        # print ('*************layer2:', xx.shape, '****************')
        xx = self.layer3(xx)
        # print ('**********************************layer3******************************')
        # print ('*************layer3:', xx.shape, '****************')
        xx = self.layer4(xx)
        # print ('**********************************layer4******************************')
        # print ('*************layer4:',xx.shape,'****************')
        score_map = self.layer5(xx)
        # print ('**********************************layer5******************************')
        # print ('*************layer5:', score_map.shape, '****************')
        # get score_map from deeplab

        # n, Channel, h, w = score_map.shape
        # score_map2 = torch.zeros(n, Channel - 1, h, w)
        # # score_map3=torch.zeros(n,Channel,h,w)
        # # score_map2=torch.zeros((n,Channel-1,h,w))
        # print ('*************before mal:', score_map2.shape, '****************')
        # for c in range(0, Channel - 1, 1):
        #     score_map2[:, c, :, :] = score_map[:, Channel - 1 - c, :, :]
        #     # score_map3[:,c,:,:]=1
        # # score_map3[:,Channel-1,:,:]=1
        # score_map2 = Variable(score_map2).cuda().float()
        # #  score_map3=Variable(score_map2).cuda().float()
        # score = torch.cat((score_map2, score_map), 1)
        # #这边要搞个scoremap把两个循环里的score合起来，再循环里一个个把score添加进去
        # # score_batch[batch_id,:,:,:]=score[:,:,:]
        # #这边要把新的总scoremap改成cost一样的形状
        # print("score_batch",score_map.shape)
        # ----------------------deeplab--------------------------------


        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)
        # print("left_feature")
        # print(refimg_feature)

        # matching
        cost = torch.FloatTensor(refimg_feature.size()[0], #batch
                                 refimg_feature.size()[1], #通道
                                 disp,
                                 refimg_feature.size()[2],#w
                                 refimg_feature.size()[3]).zero_().cuda()#h
        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]

            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature

        # print("cost")
        # print(cost)
        cost = cost.contiguous()
        # print("******************cost:",cost.shape)
        # score_reshape = torch.FloatTensor(cost.size()).zero_().cuda()
        # print("******************score:", score_map.shape)
        score_map_softmax= F.softmax(score_map, dim=1)
        # print("******************score_softmax:", score_map_softmax.shape)
        score_map_softmax_adddim1=torch.unsqueeze(score_map_softmax,1)
        # print("******************score_softmax:", score_map_softmax_adddim1.shape)
        score_reshape=score_map_softmax_adddim1.repeat(1,refimg_feature.size()[1],1,1,1)

        # for channal_id in range(refimg_feature.size()[1]):
        #     score_reshape[:,channal_id,:,:,:]=score_map_softmax[:,0,:,:,:]
        out_corr = torch.mul(score_reshape, cost)

        for f in self.filter:
            cost = f(out_corr)
        cost = self.conv3d_alone(cost)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = disparityregression(disp)(pred)

        score_map_pred = torch.squeeze(score_map_softmax, 1)
        score_map_pred = disparityregression(disp)(score_map_pred)

        
        img_pyramid_list = []
        
        for i in range(self.r):
            img_pyramid_list.append(F.interpolate(
                            left,
                            scale_factor=1 / pow(2, i),
                            mode='bilinear',
                            align_corners=False))

        img_pyramid_list.reverse()


        pred_pyramid_list= [pred]

        for i in range(self.r):
            # start = datetime.datetime.now()
            pred_pyramid_list.append(self.edge_aware_refinements[i](
                        pred_pyramid_list[i], img_pyramid_list[i]))

        length_all = len(pred_pyramid_list)


        for i in range(length_all):
            pred_pyramid_list[i] = pred_pyramid_list[i]* (
                left.size()[-1] / pred_pyramid_list[i].size()[-1])
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(pred_pyramid_list[i], dim=1),
                size=left.size()[-2:],
                mode='bilinear',
                align_corners=False),
            dim=1)
        pred_pyramid_list.append(score_map_softmax)
        # print ("score_pred...........",score_map_pred.shape)
        pred_pyramid_list.append(score_map_pred)
        return pred_pyramid_list



if __name__ == '__main__':
    model = StereoNet(k=3, r=3).cuda()
    # model.eval()
    import time
    import datetime
    import torch
    
    input = torch.FloatTensor(1,3,540,960).zero_().cuda()
    # input = torch.half(1,3,540,960).zero_().cuda()

    
    for i in range(100):
    # pass
        out = model(input, input)
        # print(len(out))
    start = datetime.datetime.now()
    for i in range(100):
        # pass
        out = model(input, input)

    end = datetime.datetime.now()
    print((end-start).total_seconds())





    


