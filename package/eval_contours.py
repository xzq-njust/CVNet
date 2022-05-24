from __future__ import division, print_function

import torch
import os
import datetime
import shutil
import numpy as np
import skimage

# from skimage.color import gray2rgb
# from skimage.color import rgb2gray

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# import heapq
# import operator
import cv2
import time
# from scipy import ndimage

from package.config import config
from package.utils.train_utils import unpack_sample, plot_sample_eval_contours, fig2img
from package.utils.data_utils import draw_poly_mask, compute_iou # , WCov_metric
from package.utils.eval_utils import db_eval_boundary
from package.train_contours import ModelAndLoss


def run(cfg, Dataset, Network):
    restore = cfg['eval_model']
    time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    exp_id = "eval_{}_{}".format(cfg['name'], time)
    save_folder = os.path.join(os.path.dirname(restore), exp_id)

    os.makedirs(save_folder)
    results_text = os.path.join(save_folder, "results.txt")
    print("Creating {}".format(save_folder))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_and_loss = ModelAndLoss(Network, restore, cfg['name']).to(device)
    model_and_loss.eval()


    dataset = Dataset(mode='test')
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=int(cfg['batch_size']), 
        num_workers=int(cfg['num_workers']),
        shuffle=False)

    running_intersection = 0
    running_union = 0
    example_iou = 0
    example_f1 = 0
    example_wcov = 0
    area = 0
    add_time = 0

    f_bound_n_fg = [0] * 5
    f_bound_fg_match = [0] * 5
    f_bound_gt_match= [0] * 5
    f_bound_n_gt = [0] * 5

    with open(results_text, 'w') as f:
        for i, sample in enumerate(dataloader):
            with torch.no_grad():
                unpack_sample(sample)
                loss, contour_x, contour_y, output, t = model_and_loss(sample)
                alpha, data, beta = output
                add_time += t

            for j in range(contour_x.shape[0]):
                predict_mask = draw_poly_mask(
                    contour_x[j].detach().squeeze().cpu().numpy(),
                    contour_y[j].detach().squeeze().cpu().numpy(),
                    (dataset.final_size, dataset.final_size),
                    outline=1)

                sequence_id = sample['sequence_id'][j]
                

                if 'bing' not in cfg['name']:
                    gt_mask = sample['mask'][j].detach().squeeze().cpu().numpy() # Vaihingen, inria
                else:
                    gt_mask = draw_poly_mask( # Bing Huts
                            sample['gt_snake_x'][j].squeeze().long().cpu().numpy(),
                            sample['gt_snake_y'][j].squeeze().long().cpu().numpy(),
                            (dataset.final_size, dataset.final_size),
                            outline=1)
                intersection, union, iou = compute_iou(predict_mask, gt_mask)


                f1 =  2 * intersection / (intersection + union)
                running_intersection += intersection
                running_union += union
                example_iou += iou
                example_f1 += f1
                gt_area = np.count_nonzero(gt_mask)
                wcov = gt_area * iou
                example_wcov += wcov
                area += gt_area

                text = "Example {}: {}".format(sequence_id, iou)
                print(text)
                f.write(text + "\n")

                ##################### visualize #####################
                # if cfg['name'] == 'vaihingen':
                #     root = cfg['data_path']
                #     image = cv2.imread(os.path.join(root, 'building_{}.tif'.format(sequence_id)))
                # elif cfg['name'] == 'bing':
                #     root = cfg['data_path']
                #     image = cv2.imread(os.path.join(root, 'building_{}.png'.format(sequence_id)))
                # elif cfg['name'] == 'inria':
                #     root = cfg['img_path']
                #     image = cv2.imread(os.path.join(root, '{}.tif'.format(sequence_id)))
                # image = cv2.resize(image, (256,256))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                
                # plot_sample_eval_contours(
                #     image,
                #     predict_mask,
                #     alpha[j].detach().squeeze().cpu().numpy(),
                #     beta[j].detach().squeeze().cpu().numpy(),
                #     data[j].detach().squeeze().cpu().numpy(),
                #     contour_x[j].detach().squeeze().cpu().numpy(),
                #     contour_y[j].detach().squeeze().cpu().numpy(),
                #     sequence_id,
                #     save_folder,
                #     caption="IOU = {}".format(iou)
                # )


                for bounds in range(5):
                    _, _, _, fg_match, n_fg, gt_match, n_gt = db_eval_boundary(predict_mask, gt_mask, bound_th=bounds + 1)
                    f_bound_fg_match[bounds] += fg_match
                    f_bound_n_fg[bounds] += n_fg
                    f_bound_gt_match[bounds] += gt_match
                    f_bound_n_gt[bounds] += n_gt

        example_iou /= len(dataset)
        text = "mIOU: {}".format(example_iou)
        print(text)
        f.write(text + "\n")

        example_f1 /= len(dataset)
        text = "mF1: {}".format(example_f1)
        print(text)
        f.write(text + "\n")

        example_wcov /= area
        text = "mwcov: {}".format(example_wcov)
        print(text)
        f.write(text + "\n")

        f_bound = [None] * 5
        for bounds in range(5):
            precision = f_bound_fg_match[bounds] / f_bound_n_fg[bounds]
            recall = f_bound_gt_match[bounds] / f_bound_n_gt[bounds]
            f_bound[bounds] = 2 * precision * recall / (precision + recall)

        text = ""
        for bounds in range(5):
            text += "F({})={},".format(bounds + 1, f_bound[bounds])
        text += "F(avg) = {}\n".format(sum(f_bound) / 5)
        print("F(avg) = {}\n".format(sum(f_bound) / 5))
        f.write(text)

        average_time = add_time / len(dataset)
        text = "average inference time: {}".format(average_time)
        print(text)
        f.write(text + "\n")

    return save_folder




