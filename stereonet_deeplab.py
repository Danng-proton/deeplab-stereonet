# ------------------------------------------------------------------------------
# Copyright (c) NKU
# Licensed under the MIT License.
# Written by Xuanyi Li (xuanyili.edu@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.utils.data
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger
import utils.visualization as visualization
from utils.utils import GERF_loss, smooth_L1_loss
from models.StereoNet_deeplab_multi import StereoNet
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
import cv2 as cv
import numpy as np
from loss_scoremap import MultiScale
import datetime

parser = argparse.ArgumentParser(description='StereoNet_deeplab with sceneflow')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0])
# parser.add_argument('--datapath', default='/media/lxy/sdd1/stereo_coderesource/dataset_nie/SceneFlowData', help='datapath')

# parser.add_argument('--datapath', default='/data1/zh/data/sceneflow', help='datapath')
parser.add_argument('--datapath', default='/data/yyx/data/sceneflow', help='datapath')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=1,
                    help='batch size for training(default: 1)')
parser.add_argument('--itersize', default=1, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for test(default: 1)')
parser.add_argument('--save_path', type=str, default='results/train-7-12/',
                    help='the path of saving checkpoints and log when training')
parser.add_argument('--test_save_path', type=str, default='results/test-7-12/',
                    help='the path of saving checkpoints and log when testing')
#    'results/8Xmulti/checkpoint_512_sceneflow_only.pth'
parser.add_argument('--resume', type=str, default='results/train-7-12/checkpoint_2020_07_16_softmax__000003.pth',
                    help='resume path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--score_lr', type=float, default=2, help='score learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',

                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=1, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.6, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--print_freq', type=int, default=100, help='print frequence')
parser.add_argument('--train', action='store_true')
parser.add_argument('--print_maps', action='store_true')
parser.add_argument('--stages', type=int, default=4, help='the stage num of refinement')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')

# 默认使用gpu
args = parser.parse_args()
writer = SummaryWriter("board_logs")  # added tensorboard


def main():
    global args
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)
    train_left_img.sort()
    train_right_img.sort()
    train_left_disp.sort()

    test_left_img.sort()
    test_right_img.sort()
    test_left_disp.sort()

    __normalize = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True, normalize=__normalize),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, normalize=__normalize),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if args.train:
        save_path = args.save_path
    else:
        save_path = args.test_save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log = logger.setup_logger(save_path + '/712sceneflow-test.log')  ################3training
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ':' + str(value))

    # 创建一个StereoNet模型，初始化
    model = StereoNet(k=args.stages - 1, r=args.stages - 1, maxdisp=args.maxdisp)
    model = nn.DataParallel(model).cuda()
    model.apply(weights_init)
    print('init with normal')

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format((args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            # model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> will start from scratch.")
    else:
        log.info("Not Resume")
    start_full_time = time.time()
    for epoch in range(args.start_epoch, args.epoch):

        log.info('This is {}-th epoch'.format(epoch))

        if args.train:
            # train
            train(TrainImgLoader, model, save_path, optimizer, log, epoch)

        datenow = datetime.datetime.now().strftime('%Y_%m_%d')
        dateload = datetime.datetime.now().strftime('%m%d')
        checkpoint_data = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_data, "{}/checkpoint_{}_softmax__{:0>6}.pth".format(args.save_path, datenow, epoch))
        if args.train:
            # train
            savefilename = save_path + '/checkpoint-softmax-sceneflow-{}.pth'.format(dateload)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                savefilename)
            # train
        scheduler.step()  # will adjust learning rate
        # scheduler.step()#每调用step_size次，对应的学习率就会按照策略调整一次。
        if not args.train:
            # test
            test(TestImgLoader, model, save_path, log)
            # test
        log.info('full training time = {: 2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, save_path, optimizer, log, epoch=0, ):
    stages = args.stages
    losses = [AverageMeter() for _ in range(stages)]
    loss_scoremap = [AverageMeter()]
    length_loader = len(dataloader)
    counter = 0
    score_loss_funcation = MultiScale(args)
    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):

        # print(path)
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        # print(imgL.shape)
        outputs = model(imgL, imgR)

        scoremap_pred = outputs.pop(-1)  # torch.Size([1, 68, 120])
        scoremap = outputs.pop(-1)
        # print("scoremap",scoremap_origin.shape,"scoremap pred",scoremap.shape)
        outputs = [torch.squeeze(output, 1) for output in outputs]

        loss = [GERF_loss(disp_L, outputs[0], args)]
        # print("gt", disp_L.shape, "output", 0, outputs[0].shape)
        # print("lossshape",loss[0].shape)

        for i in range(len(outputs) - 1):
            # print("gt", disp_L.shape, "output", i, outputs[i].shape)
            loss.append(GERF_loss(disp_L, outputs[i + 1], args))
        # print("scoremap", scoremap.shape, "disp_l",disp_L.shape)
        loss_score = score_loss_funcation(scoremap, disp_L)
        # print("loss_score",loss_score)
        loss.append(loss_score[0] * args.score_lr)

        counter += 1
        if loss_score[0] < 0.3:
            loss_all = sum(loss) / (args.itersize)
        #############
        else:
            loss_all = loss_score[0]
        loss_all.backward()  # 反向传播
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        for idx in range(stages):
            losses[idx].update(loss[idx].item() / args.loss_weights[idx])
        loss_scoremap[0].update(loss_score[0].item())

        writer.add_scalar('train_loss_all', loss_all, epoch)
        writer.add_scalar('train_loss_score', loss_score[0], epoch)
        writer.add_scalar('train_loss_-1level', loss[-2], epoch)

        if batch_idx % args.print_freq == 0:
            # print(loss_score[0])
            info_str = ['Stage {} loss = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]

            # info_str = '\t'.join(info_str)

            # added
            info_str.extend(['score loss = {:.2f}({:.2f})'.format(loss_scoremap[0].val, loss_scoremap[0].avg)])

            info_str = '\t'.join(info_str)
            # added

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))

            # vis
            _, H, W = outputs[0].shape
            scoremap_pred = F.interpolate(
                (torch.unsqueeze(scoremap_pred, 1)),
                size=[H, W],
                mode='bilinear',
                align_corners=False)
            all_results = torch.zeros((len(outputs) + 2, 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, :, :] / 255.0
            # print(outputs[j][0, :, :]/255.0,scoremap_pred[:, :])
            all_results[-2, 0, :, :] = scoremap_pred[:, :] * 13 / 255.
            all_results[-1, 0, :, :] = disp_L[0][:, :] / 255.0
            # print("save_path",save_path)
            # 这边这边
            torchvision.utils.save_image(all_results, join(save_path, "iter-%d.jpg" % batch_idx))
            # print(imgL)
            # torchvision.utils.save_image(scoremap, join(save_path, "iter-scoremap-%d.jpg" % batch_idx))
            imgL = imgL.cpu()
            im = np.array(imgL[0, :, :, :].permute(1, 2, 0) * 255, dtype=np.uint8)

            # train_logger = SummaryWriter(log_dir=os.path.join(args.save, 'train'), comment='training')
            # validation_logger = SummaryWriter(log_dir=os.path.join(args.save, 'validation'), comment='validation')

            # 这边这边
            cv.imwrite(join(save_path, "itercolor-%d.jpg" % batch_idx), im)

    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)

    writer.export_scalars_to_json("./test.json")  # tensorboard
    writer.close()  # tensorboard


def test(dataloader, model, save_path, log):
    # print("in_test")
    stages = args.stages
    # End-point-error
    EPES = [AverageMeter() for _ in range(stages)]

    length_loader = len(dataloader)
    # print(length_loader)
    model.eval()
    # model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        visual = visualization.disp_error_image_func()
        mask = (disp_L < args.maxdisp) & (disp_L > 0)

        # mask = disp_L < args.maxdisp

        with torch.no_grad():
            outputs = model(imgL, imgR)
            scoremap_pred = outputs.pop(-1)
            score_map_softmax = outputs.pop(-1)
            for x in range(stages):

                if len(disp_L[mask]) == 0:
                    EPES[x].update(0)

                    continue
                output = torch.squeeze(outputs[x], 1)
                EPES[x].update((output[mask] - disp_L[mask]).abs().mean())

        # if batch_idx%100==0:
        info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPES[x].val, EPES[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

        disp_ests = outputs[:stages]
        scoremap = scoremap_pred
        # print("before print_maps")
        if args.print_maps:
            # print("print maps")
            # sceneflow
            test_gt_path = save_path + 'driving_test_gt/'
            if not os.path.exists(test_gt_path):
                os.makedirs(test_gt_path)
            test_pred1_path = save_path + 'driving_test_pred/'
            if not os.path.exists(test_pred1_path):
                os.makedirs(test_pred1_path)
            test_pred1_scoremap = save_path + 'driving_test_scoremap/'
            if not os.path.exists(test_pred1_scoremap):
                os.makedirs(test_pred1_scoremap)
            test_pred1_errormap_pred1 = save_path + 'driving_test_errormap_pred/'
            if not os.path.exists(test_pred1_errormap_pred1):
                os.makedirs(test_pred1_errormap_pred1)
            test_pred1_errormap_scoremap = save_path + 'driving_test_errormap_scoremap/'
            if not os.path.exists(test_pred1_errormap_scoremap):
                os.makedirs(test_pred1_errormap_scoremap)
            # print(test_gt_path)
            _, H, W = disp_ests[0].shape
            gt = torch.zeros((H, W))
            im_pred1 = torch.zeros((H, W))
            # print(disp_gt.shape)
            disp_gt = disp_L
            gt = np.array(disp_gt[0, :, :].cpu(), dtype=np.uint16)
            # print(disp_gt.shape)
            cv.imwrite(join(test_gt_path, "sceneflow-gt-%d.png" % batch_idx), gt * 255)

            im_pred1 = np.array(disp_ests[3][0, :, :].cpu(), dtype=np.uint16)
            # print(im_pred1.shape)
            cv.imwrite(join(test_pred1_path, "sceneflow-pred-%d.png" % batch_idx), im_pred1 * 255)
            # print ("im_pred1................",im_pred1.shape,type(im_pred1))

            im_scoremap = np.array(scoremap[0, :, :].cpu(), dtype=np.uint16)
            # print("im_scoremap")

            # print(im_scoremap.shape)
            # print(im_scoremap.dtype)
            cv.imwrite(join(test_pred1_scoremap, "sceneflow-scoremap-%d.png" % batch_idx), im_scoremap * 255)
            # --------------------
            errormap_pred1 = visual(disp_ests[3][:, :, :].cpu(), disp_gt[:, :, :].cpu())
            # print ("errormap_pred1................", type(errormap_pred1))
            errormap_pred1 = np.squeeze(errormap_pred1).numpy().transpose(1, 2, 0) * 255
            # print("errormap_pred1")
            # print(errormap_pred1)
            cv.imwrite(join(test_pred1_errormap_pred1, "sceneflow-pred-errormap-%d.png" % batch_idx), errormap_pred1)

            _, size_w, size_h = scoremap.shape
            disp_gt_resized = cv.resize(disp_gt[0, :, :].cpu().numpy(), (size_h, size_w), interpolation=cv.INTER_AREA)
            disp_gt_resized = torch.unsqueeze(torch.from_numpy(disp_gt_resized), 0)
            # --------------------
            errormap_scoremap = visual(scoremap[:, :, :].cpu(), disp_gt_resized[:, :, :].cpu())
            errormap_scoremap = np.squeeze(errormap_scoremap).numpy().transpose(1, 2, 0) * 255
            # print("errormap_scoremap")
            # print(errormap_scoremap)

            ################
            cv.imwrite(join(test_pred1_errormap_scoremap, "sceneflow-score-errormap-%d.png" % batch_idx),
                       errormap_scoremap)
            ################
    # #kitti
    # test_gt_path=save_path+'/driving_test_gt/'
    # test_pred1_path=save_path+'/driving_test_pred/'

    # _, H, W = disp_ests[0].shape
    # gt=torch.zeros((H,W))
    # im_pred1 = torch.zeros(( H, W))
    # #print(disp_gt.shape)
    # gt = np.array(disp_gt[0,:, :].cpu(), dtype=np.uint16)
    # cv.imwrite(join(test_gt_path, "sceneflow-gt-%d.png" % batch_idx),gt)
    # im_pred1 = np.array(disp_ests[2][0,:, :].cpu(), dtype=np.uint16)
    # cv.imwrite(join(test_pred1_path, "sceneflow-%d.png" % batch_idx),im_pred1)

    # vis
    # _, H, W = outputs[0].shape
    # all_results = torch.zeros((len(outputs)+1, 1, H, W))
    # for j in range(len(outputs)):
    #     all_results[j, 0, :, :] = outputs[j][0, :, :]/255.0
    # all_results[-1, 0, :, :] = disp_L[:, :]/255.0
    # torchvision.utils.save_image(all_results, join(save_path, "iter-%d.jpg" % batch_idx))
    # # print(imgL)
    # im = np.array(imgL[0,:,:,:].permute(1,2,0)*255, dtype=np.uint8)
    # print(im.shape)
    # cv.imwrite(join(save_path, "itercolor-%d.jpg" % batch_idx),im)

    # _, H, W = outputs[0].shape
    # all_results_color = torch.zeros((H, 5*W))
    # all_results_color[:,:W]= outputs[0][0, :, :]
    # all_results_color[:,W:2*W]= outputs[1][0, :, :]
    # # print(disp_L)
    # all_results_color[:,2*W:3*W]= outputs[2][0, :, :]
    # all_results_color[:,3*W:4*W]= outputs[3][0, :, :]

    # all_results_color[:,4*W:5*W]= disp_L[:, :]

    # _, H, W = outputs[3].shape
    # all_results_color = torch.zeros((H, W))
    # all_results_color[:,:,:]= outputs[3][0, :, :]

    # im_color = cv.applyColorMap(np.array(all_results_color*2, dtype=np.uint8), cv.COLORMAP_JET)
    # cv.imwrite(join(save_path, "iterpredcolor-%d.jpg" % batch_idx),im_color)

    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPES[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, nn.Conv3d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def D_metric(D_est, D_gt, mask, thres):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > thres) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())


def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())


if __name__ == '__main__':
    main()


