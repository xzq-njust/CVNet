from __future__ import division, print_function

import torch
import os
import datetime
import heapq
import shutil
import numpy as np
import cv2
import logging
from PIL import Image
from tqdm import tqdm
import time

from package.utils.train_utils import AverageMeter, unpack_sample, save_config
from package.losses.losses import DistanceLossFast
from package.utils.data_utils import draw_poly_mask, compute_iou
torch.manual_seed(1234)
np.random.seed(1234)


# Log setting
def _add_file_handler(logger,
                      filename=None,
                      mode='w',
                      level=logging.INFO):
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def init_logger(log_dir=None, level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger()

    filename = '{}.log'.format(get_time_str())
    log_file = os.path.join(log_dir, filename)
    logger = _add_file_handler(logger, log_file, level=level)
    return logger


def norm_input(_x, seg_out, a=0.9):
    _x = a * _x + (1 - a) * seg_out
    return _x


# Get model and loss
class ModelAndLoss(torch.nn.Module):
    def __init__(self, Network, restore, dataset_name='vaihingen'):
        super(ModelAndLoss, self).__init__()
        if dataset_name == 'bing':
            self.net = Network(model_name='drn_d_38')
        else:
            self.net = Network()
        print("Loading checkpoint from {}".format(restore))
        checkpoint = torch.load(restore)
        print("Loaded checkpoint")
        self.net.load_state_dict(checkpoint['state_dict'])
        self.distance_loss = DistanceLossFast(dataset_name)

    def forward(self, sample):
        output = self.net(sample['image'])

        assert len(output) == 3 or len(output) == 6
        alpha, data, beta = output[-3:]
        output = alpha, data, beta

        loss, contour_x, contour_y, time = self.distance_loss(
            alpha, beta, data, 
            sample['init_contour'],
            sample['init_contour0'],
            sample['gt_snake_x'],
            sample['gt_snake_y'],
            sample['faces'],
            sample['mask'])

        return loss, contour_x, contour_y, output, time



def plot_contour(image, contour_x, contour_y, gt, sequence_id, save_folder):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    contours = np.array([contour_x, contour_y]).T
    contours = [contours[:, np.newaxis, :]]
    gt = [gt[:, np.newaxis, :]]
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    image = cv2.drawContours(image, gt, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_folder, "{}.png".format(sequence_id)), image)


def run(cfg_exp, config_object, Dataset, Network):
    # Environment setup
    time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    exp_id = "{}_{}".format(cfg_exp['name'], time)
    save_folder = os.path.join(cfg_exp['save_path'], exp_id)
    keep_best = int(cfg_exp['keep_best'])
    checkpoint_heap = []
    val_losses = AverageMeter()
    global_iter = 0
    np.seterr(divide='ignore', invalid='ignore')
    restore = cfg_exp['restore']

    # Sanity checks
    assert keep_best > 0
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("Created {}".format(save_folder))
        save_config(config_object, save_folder)
    else:
        print("Save folder already exists")

    batch_size = int(cfg_exp['batch_size'])
    num_workers = int(cfg_exp['num_workers'])
    num_epochs = int(cfg_exp['num_epochs'])
    learning_rate = float(cfg_exp['learning_rate'])
    momentum = float(cfg_exp['momentum'])
    weight_decay = float(cfg_exp['weight_decay'])
    patience = int(cfg_exp['patience'])
    save_delay = int(cfg_exp['save_delay'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # setting a log
    logger = init_logger(log_dir=save_folder)

    # Get dataloaders
    train_dataset = Dataset(mode='train')
    val_dataset = Dataset(mode='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=batch_size, num_workers=num_workers, shuffle=False)


    model_and_loss = ModelAndLoss(Network, restore, cfg_exp['name']).to(device) 
    if torch.cuda.device_count() > 1:
        model_and_loss.net = torch.nn.DataParallel(model_and_loss.net)


    # Get optimizer and scheduler; should use the first one if you don't have a test set

    if cfg_exp['name'] == 'vaihingen':
        for name, param in model_and_loss.net.named_parameters(): # freeze several layers for vaihingen
            if "base" in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(model_and_loss.net.parameters(), 
            lr=learning_rate, weight_decay=weight_decay)
        print('optimizer: Adam')
    else:
        optimizer = torch.optim.SGD(model_and_loss.net.parameters(), 
            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        print('optimizer: SGD')

    if 'gamma' in cfg_exp:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=float(cfg_exp['gamma']), step_size=patience)
        print("Scheduler: StepLR with gamma {}".format(cfg_exp['gamma']))

    for epoch_num in range(num_epochs):
        logger.info('epoch: {}'.format(epoch_num+1))
        # Train
        total_loss = 0
        iter_count = 0
        model_and_loss.train()   # Turn off updating batchnorm

        for batch_idx, sample in enumerate(train_loader):
            unpack_sample(sample)
            optimizer.zero_grad()
            loss, contour_x, contour_y, _, _ = model_and_loss(sample)
            logger.info("Loss is {}".format(loss))
            loss.backward()
            optimizer.step()

            # for j in range(contour_x.shape[0]):
            #     sequence_id = sample['sequence_id'][j].cpu().item()
            #     plot_contour(
            #         train_dataset.unnormalize(sample['image'][j].squeeze()).detach().cpu().numpy().transpose(1, 2, 0)*255,
            #         contour_x[j].detach().squeeze().long().cpu().numpy(),
            #         contour_y[j].detach().squeeze().long().cpu().numpy(),
            #         sample['gt_snake'][j].squeeze().long().cpu().numpy(),
            #         sequence_id,
            #         save_folder)


        # update learning-rate
        if 'gamma' in cfg_exp:
            scheduler.step()

        # Val
        model_and_loss.eval()
        running_intersection = 0
        running_union = 0
        example_iou = 0
        for batch_idx, sample in enumerate(val_loader):
            with torch.no_grad():
                unpack_sample(sample)
                loss, contour_x, contour_y, _, _ = model_and_loss(sample)

                for j in range(contour_x.shape[0]):
                    predict_mask = draw_poly_mask(
                        contour_x[j].detach().squeeze().long().cpu().numpy(),
                        contour_y[j].detach().squeeze().long().cpu().numpy(),
                        (val_dataset.final_size, val_dataset.final_size),
                        outline=1)
                    
                    if cfg_exp['name'] == 'bing':
                        gt_mask = draw_poly_mask( # for Bing Huts, due to the imprecise label
                            sample['gt_snake_x'][j].squeeze().long().cpu().numpy(),
                            sample['gt_snake_y'][j].squeeze().long().cpu().numpy(),
                            (val_dataset.final_size, val_dataset.final_size),
                            outline=1)
                    else:
                        gt_mask = sample['mask'][j].detach().squeeze().cpu().numpy() # for vaihingen, inria
                    # sequence_id = sample['sequence_id'][j].cpu().item()
                    # plot_contour(
                    #     val_dataset.unnormalize(sample['image'][j].squeeze()).detach().cpu().numpy().transpose(1, 2, 0)*255,
                    #     contour_x[j].detach().squeeze().long().cpu().numpy(),
                    #     contour_y[j].detach().squeeze().long().cpu().numpy(),
                    #     sample['gt_snake'][j].squeeze().long().cpu().numpy(),
                    #     sequence_id,
                    #     save_folder)
                    intersection, union, iou = compute_iou(predict_mask, gt_mask)
                    running_intersection += intersection
                    running_union += union
                    example_iou += iou
        average_iou = example_iou / len(val_dataset)
        logger.info("IoU is {}%".format(average_iou*100))


        # Save model; keep record of the validation loss in a heap
        # so the largest values can be popped off for checkpoint deletion
        if epoch_num + 1 >= save_delay:
            checkpoint_path = os.path.join(save_folder,
                "chk-{:04}-{:05}.pth.tar".format(epoch_num+1, average_iou))
            if isinstance(model_and_loss.net, torch.nn.DataParallel):
                checkpoint = {
                    'epoch': epoch_num + 1,
                    'state_dict': model_and_loss.net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'miou': average_iou}
            else:
                checkpoint = {
                    'epoch': epoch_num + 1,
                    'state_dict': model_and_loss.net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'miou': average_iou}
            heapq.heappush(checkpoint_heap, (average_iou, checkpoint_path))

            # Avoid saving checkpoints if not necessary to reduce bandwidth consumption
            if len(checkpoint_heap) > keep_best:
                _, checkpoint_to_delete = heapq.heappop(checkpoint_heap)
                if checkpoint_path == checkpoint_to_delete:
                    pass
                else:
                    torch.save(checkpoint, checkpoint_path)
                    os.remove(checkpoint_to_delete)
            else:
                torch.save(checkpoint, checkpoint_path)

        # On the very last epoch, save the checkpoint
        if epoch_num + 1 == num_epochs:
            checkpoint_path = os.path.join(save_folder,
                "chk-{:04}-{:05}.pth.tar".format(epoch_num+1, average_iou))
            torch.save(checkpoint, checkpoint_path)
    
    # Set aside very best checkpoint
    _, best_checkpoint_path = checkpoint_heap[0]
    shutil.copyfile(best_checkpoint_path, os.path.join(save_folder, "best_chk.pth.tar"))

    return os.path.abspath(best_checkpoint_path)

