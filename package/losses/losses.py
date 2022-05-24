from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from ChamferDistance.chamfer2D.dist_chamfer_2D import chamfer_2DDist as cham2D

from package.utils.contour_utils import evolve_active_cvnet

import time

import neural_renderer as nr
import numpy as np


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def ignore_null(gt, shp_x, nl_cls=250):
    shp_x = list(shp_x)
    nC = shp_x[1]
    gt[gt == nl_cls] = nC + 1
    shp_x[1] = nC + 1
    return gt, shp_x


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            gt, shp_x = ignore_null(gt, shp_x)
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
            y_onehot = y_onehot[:, :-1, :, :]
    # input(y_onehot.shape)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1e-6,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc


class DistanceLossFast(nn.Module):
    def __init__(self, dataset_name, delta_t=3e-3, max_steps=70):
        super(DistanceLossFast, self).__init__()
        self.delta_t = delta_t
        self.max_steps = max_steps
        self.chamfer_dist = cham2D()
        self.criterion = SoftDiceLoss()

        self.image_size = 256
        self.half_dim = self.image_size / 2
        self.renderer = nr.Renderer(camera_mode='look_at', image_size=self.image_size, light_intensity_ambient=1,
                                    light_intensity_directional=1, perspective=False)
        self.camera_distance = 1
        self.elevation = 0
        self.azimuth = 0
        self.renderer.eye = nr.get_points_from_angles(self.camera_distance, self.elevation, self.azimuth)

        if dataset_name == 'vaihingen':
            self.lamda = 100
        elif dataset_name == 'bing':
            self.lamda = 10
        else:
            self.lamda = 0
            self.delta_t = 3e-4 # for inria


    def forward(self, alpha, beta, data, 
        init_contour, init_contour0, gt_snake_x, gt_snake_y, faces, gt_mask, L=60):
        tic = time.time()
        # import pdb
        # pdb.set_trace()
        contour_x, contour_y = evolve_active_cvnet(
            alpha, beta, init_contour, init_contour0, 
            delta_t=self.delta_t, max_steps=self.max_steps)
        infer_time = time.time() - tic

        # chamfer loss
        pred = torch.stack([contour_x, contour_y], dim=2)
        gt = torch.stack([gt_snake_x, gt_snake_y], dim=2)
        dist1, dist2, idx1, idx2 = self.chamfer_dist(pred, gt)
        loss = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))



        # mask dice loss
        P = torch.stack([contour_x, contour_y], dim=2)
        P = (P - self.half_dim) / self.half_dim
        z = torch.ones((P.shape[0], P.shape[1], 1)).cuda()
        PP = torch.cat((P, z), 2)
        PP[:, :, 1] = PP[:, :, 1]*-1

        pred_mask = self.renderer(PP, faces, mode='silhouettes').unsqueeze(dim=1)
        loss += self.criterion(pred_mask, gt_mask) * self.lamda


        return loss, contour_x, contour_y, infer_time







