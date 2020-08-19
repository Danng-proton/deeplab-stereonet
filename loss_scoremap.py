'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from os.path import *
from torch.autograd import Variable
#from _bilinear_sampler import *
import cv2

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        # lossvalue = torch.abs(output - target).mean()
        mask = torch.ByteTensor(target.type(torch.FloatTensor)!=0).type(torch.cuda.ByteTensor)
        lossvalue = torch.abs(output.masked_select(mask) - target.masked_select(mask)).mean()

        # x, y = target.nonzero()
        # lossvalue = torch.abs(output[x,y]-target[x,y]).mean()

        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class MultiScale(nn.Module):
    def __init__(self, args, startScale=4, numScales=5, l_weight=0.64, norm='L1'):
        super(MultiScale, self).__init__()
        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(
            self.numScales)]).cuda()  ################################Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #3 'other'
        self.args = args
        self.l_type = norm
        self.div_flow = 1
        assert (len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        # self.multiScales = [nn.MaxPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        # self.multiScales = [nn.UpsamplingNearest2d(scale_factor = 1./(self.startScale * (2**scale))) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-' + self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        ###get classification label
        # kitti max 230 sceneflow max 300
        # label=target/328*41
        # interp = nn.UpsamplingBilinear2d(size =(output[5].shape[2],output[5].shape[3]))
        # label=interp(label)
        # interp = nn.Upsample(size =(target.shape[2], target.shape[3]), mode='nearest')
        # LR consistency loss
        # lossvalue += loss_lr(data, output, target)

        if type(output) is not tuple:
            target = self.div_flow * target
            output_ = output
            # target_0 = np.zeros((1,output_.shape[2], output_.shape[3]))
            # print(type(target))
            # target=torch.squeeze(target)
            # output_reshape=cv2.resize(src=target, dsize=(int(output_.shape[3]), int(output_.shape[2])),
            #            interpolation=cv2.INTER_NEAREST)
            # target_0 =torch.from_numpy(np.expand_dims(output_reshape,0).cpu())

            # target = np.array(target)
            # target = cv2.resize(src=target, dsize=(int(output_.shape[-1]), int(output_.shape[-2])),
            #                     interpolation=cv2.INTER_NEAREST)
            # target = torch.squeeze(target).cpu()
            # print("GTshape",np.expand_dims(target,1).shape,"output",output_.shape)
            # np.expand_dims(target, 1)
            # print(type(target))
            target = F.interpolate(
                (torch.unsqueeze(target, 1)),
                size=[output_.shape[-2], output_.shape[-1]],
                mode='bilinear',
                align_corners=False)
            # target = np.expand_dims(target, 0)
            # # target = np.expand_dims(target, 0)
            # target_0 = torch.from_numpy(target)
            # target_0=Variable(target_0).cuda()
            target_0 = Variable(target).cuda()
            # scoremap_disp=output_.shape[1]
            # # print("scoremap_disp",scoremap_disp.shape(),scoremap_disp.shape())
            # max_disp=scoremap_disp*8
            # label=target_0/max_disp*scoremap_disp
            label = target_0 / 8
            # if len(output_.shape)==len(label.shape):
            #     for batch_id in range(output_.shape[0]):
            #         semanteme_loss = loss_calc(output_[batch_id,:,:,:], label[batch_id,:,:,:], target_0[batch_id,:,:,:])
            # elif output_.shape[0]==1:
            semanteme_loss = loss_calc_unsofted(output[:, :, :, :], label, target_0)
            # semanteme_loss = loss_calc(output_, label, target_0)
            lossvalue += semanteme_loss
            # print("output",torch.mean(output),"target",torch.mean(target))
            # epevalue += EPE(output_[0,:,:,:], target_0)
            epevalue += EPE(output_, target_0)
            #           print '\n\n',lossvalue,'\n\n'
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return [lossvalue, epevalue]


def loss_calc_unsofted(out, label, target):
    # type: (object, object, object) -> object

    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w

    # label = label.transpose(3,2,0,1)
    # label = torch.from_numpy(label)
    label = Variable(label).cuda().int()
    [batch,channels, h, w] = out.shape

    bin = torch.Tensor(batch,channels, h, w)
    # label_list = torch.Tensor(batchsize,channels,h,w).type(torch.cuda.ByteTensor)
    target_list = torch.Tensor(batch,channels, h, w).type(torch.cuda.ByteTensor)
    for C in range(channels):
        bin[:,C, :, :] = C
        target_list[:,C, :, :] = target[:,0, :, :]
    bin = Variable(bin).cuda().int()
    res = (bin - label).float()

    W = torch.exp(-0.5 * res.mul(res)).float()
    m = nn.LogSoftmax()
    out = torch.where(target_list == 0, torch.full_like(out, 1), out)
    out = m(out)
    W=F.softmax(W,dim=1)
    # print("------------",max(W[0,:,50,50]),max(W[0,:,50,100]),max(W[0,:,50,70]))

    out = out.mul(W)
    return -torch.mean(out)

def loss_calc_soft(out, label, target):
    # type: (object, object, object) -> object

    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w

    # label = label.transpose(3,2,0,1)
    # label = torch.from_numpy(label)
    label = Variable(label).cuda().int()
    [batch, channels, h, w] = out.shape
    bin = torch.Tensor(batch, channels, h, w)
    target_list = torch.Tensor(batch, channels, h, w).type(torch.cuda.ByteTensor)
    for C in range(channels):
        bin[:, C, :, :] = C
        target_list[:, C, :, :] = target[:, 0, :, :]
    bin = Variable(bin).cuda().int()
    res = (bin - label).float()
    W = torch.exp(-0.5 * res.mul(res)).float()
    # print("out", torch.mean(out))
    out = torch.where(target_list == 0, torch.full_like(out, 1), out)
    out = torch.log2(out + (0.00000000001))
    W = F.softmax(W, dim=1)
    # print("out", torch.mean(out))
    out = out.mul(W)
    return -torch.mean(out)


def loss_calc(out, label, target):
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w

    eps = 1e-5
    # label = label.transpose(3,2,0,1)
    # label = torch.from_numpy(label)
    label = Variable(label).cuda().int()
    [channels, h, w] = out.shape

    bin = torch.Tensor(channels, h, w)
    # label_list = torch.Tensor(batchsize,channels,h,w).type(torch.cuda.ByteTensor)
    target_list = torch.Tensor(channels, h, w).type(torch.cuda.ByteTensor)
    for C in range(channels):
        bin[C, :, :] = C
        target_list[C, :, :] = target[ 0, :, :]
    bin = Variable(bin).cuda().int()
    res = (bin - label).float()

    W = -0.5 * res.mul(res).float()
    W = F.softmax(W, dim=1)
    m = nn.LogSoftmax()
    # m = torch.log2()
    out = torch.where(target_list == 0, torch.full_like(out, 1), out)
    # out = torch.log_softmax(out)
    # out = torch.log(out)
    # out = m(out)
    out = torch.log2(out + eps)
    out = out.mul(W)
    return -torch.mean(out)

def loss_calc_manybatch(out, label, target):
   
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w

        eps = 1e-5
        #label = label.transpose(3,2,0,1)
        #label = torch.from_numpy(label)
        label = Variable(label).cuda().int()
        [batch,channels,h,w]=out.shape
        
        bin = torch.Tensor(batch,channels,h,w)
        # label_list = torch.Tensor(batchsize,channels,h,w).type(torch.cuda.ByteTensor)
        target_list = torch.Tensor(batch,channels,h,w).type(torch.cuda.ByteTensor)
        for C in range(channels):
            bin[:,C,:,:]=C
            target_list[:,C,:,:]=target[:,0,:,:]
        bin = Variable(bin).cuda().int()
        res = (bin - label).float()
        
        W = -0.5*res.mul(res).float()
        W = F.softmax(W,dim=1)
        m = nn.LogSoftmax()
        # m = torch.log2()
        out = torch.where(target_list==0,torch.full_like(out, 1), out)
        # out = torch.log_softmax(out)
        # out = torch.log(out)
        # out = m(out)
        out = torch.log2(out+eps)
        out = out.mul(W)
        return -torch.mean(out)

def generate_image_left(img, disp):
        # print img.shape, disp.shape
        return bilinear_sampler_1d_h(img, -disp)

def generate_image_right(img, disp):
        return bilinear_sampler_1d_h(img, disp)

def loss_lr(img, flow, ground_truth, alpha_image_loss = 0.85):
    left = img[:,:,0,:,:]
    right = img[:,:,1,:,:]
    left_pyramid    = scale_pyramid(left, 4)
    right_pyramid   = scale_pyramid(right, 4)

    # STORE DISPARITIES
    # with tf.variable_scope('disparities'):
    #     self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
    #     self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
    #     self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]
    # disp_net = [disp1, disp2, disp3, disp4]
    disp_est = [flow[i]/2**i for i in range(4)]
    disp_left_est   = [torch.unsqueeze(d[:,0,:,:], 1) for d in disp_est]
    # disp_right_est  = [torch.unsqueeze(d[:,1,:,:], 1) for d in disp_est]

    # GENERATE IMAGES
    # with tf.variable_scope('images'):
    #     self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
    #     self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]
    left_est    = [generate_image_left(right_pyramid[i], disp_left_est[i])      for i in range(4)]

    # right_est   = [generate_image_right(left_pyramid[i], disp_right_est[i])   for i in range(4)]
    
    # IMAGE RECONSTRUCTION
    # L1
    max_pooling = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    max_disp = max_pooling(ground_truth)
    mask = torch.ByteTensor(max_disp.type(torch.FloatTensor)==0).type(torch.FloatTensor).cuda######
    mask_pyramid = scale_pyramid(mask, 4)
    mask_pyramid = [mask_pyramid[i].trunc() for i in range(4)]

    l1_left = [torch.abs(left_est[i] - left_pyramid[i]) for i in range(4)]
    l1_left = [torch.where(mask_pyramid[i] == 1, l1_left[i], torch.full_like(l1_left[i], 0)) for i in range(4)]
    # l1_left = [torch.where(disp_est[i] < 4.5, l1_left[i], torch.full_like(l1_left[i], 0))    for i in range(4)]
    # l1_reconstruction_loss_left  = [torch.mean(l) for l in l1_left]
    l1_left = [torch.where(l1_left[i] > 10, torch.full_like(l1_left[i], 0), l1_left[i]) for i in range(4)]
    l1_reconstruction_loss_left = [torch.mean(l) for l in l1_left]
    # pdb.set_trace()
    l1_smoothness_loss = [smoothness_loss(left_pyramid[i], disp_left_est[i]) for i in range(4)]
    
    # SSIM
    ssim_left = [SSIM(left_est[i], left_pyramid[i]) for i in range(4)]
    ssim_loss_left  = [torch.mean(s) for s in ssim_left]

    # WEIGTHED SUM
    # self.image_loss_right = [self.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
    image_loss_left  = [alpha_image_loss * ssim_loss_left[i]  + (1 - alpha_image_loss) * l1_reconstruction_loss_left[i]  for i in range(4)]
    # image_loss = tf.add_n(image_loss_left + image_loss_right)
    image_loss = 0.2 * sum(image_loss_left) + 10 * sum(l1_smoothness_loss)

    return image_loss

def scale_pyramid(img, num_scales):
        scaled_imgs = [img]
        s = img.shape
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            interp = nn.Upsample(size =([nh, nw]), mode='bilinear', align_corners=True)
            scaled_imgs.append(interp(img))
        return scaled_imgs

def SSIM(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y,3, 1)

        sigma_x  = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y , 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

# SMOOTHNESS
def smoothness_loss(left, disp):
    # conv1 = nn.Conv2d(1, 1, 3, bias=False)
    # sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # sobel_kernel = sobel_kernel.reshape((1, 3, 3, 3))
    # conv1.weight.data = torch.from_numpy(sobel_kernel)
    # left = img[:,:,0,:,:]

    left_gray = color_to_gray(left)
    left_gray_gradient = gradient_1order(left)

    # pdb.set_trace()
    smoothness_loss = torch.mean(torch.abs(x_gradient_1order(disp))*torch.exp(-torch.abs(x_gradient_1order(left_gray_gradient)))+\
                        torch.abs(y_gradient_1order(disp))*torch.exp(-torch.abs(y_gradient_1order(left_gray_gradient))))
    
    return smoothness_loss

def color_to_gray(img):
    img = img.permute(0, 2, 3, 1)
    img_gray = 0.299*img[:,:,:,0]+0.587*img[:,:,:,1]+0.114*img[:,:,:,2]
    img_gray = img_gray.unsqueeze(3).permute(0, 3, 1, 2)
    return img_gray

def x_gradient_1order(img):
    img = img.permute(0,2,3,1)
    img_l = img[:,:,1:,:] - img[:,:,:-1,:]
    img_r = img[:,:,-1,:] - img[:,:,-2,:]
    img_r = img_r.unsqueeze(2)
    img  = torch.cat([img_l, img_r], 2).permute(0, 3, 1, 2)
    return img

def y_gradient_1order(img):
    # pdb.set_trace()
    img = img.permute(0,2,3,1)
    img_u = img[:,1:,:,:] - img[:,:-1,:,:]
    img_d = img[:,-1,:,:] - img[:,-2,:,:]
    img_d = img_d.unsqueeze(1)
    img  = torch.cat([img_u, img_d], 1).permute(0, 3, 1, 2)
    return img

def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad