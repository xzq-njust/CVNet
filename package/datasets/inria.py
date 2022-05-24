from __future__ import print_function, division

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image

from package.config import config
from package.datasets.base_dataset import BaseDataset



class InriaDataset(BaseDataset):
    MEAN = np.array([0.42825925, 0.4411106, 0.40593693])
    STD = np.array([0.21789166, 0.20679809, 0.20379359])
    ORIG_SIZE = 128
    ORIG_DIST_PER_PX = 0.30

    def __init__(self, mode, final_size=256, discretize_points=60,
        init_contour_radius=20, normalize=True):
        self.cfg = config['inria']
        super(InriaDataset, self).__init__(
            mode, self.cfg['image_extension'], init_contour_radius,
            final_size=final_size, discretize_points=discretize_points,
            normalize=normalize)

        self.normalize = transforms.Normalize(self.MEAN, self.STD)
        self.unnormalize = transforms.Normalize(-self.MEAN / self.STD, 1 / self.STD)

        self.image_path = self.cfg['img_path']
        self.mask_one_path = self.cfg['mask_one_path']
        self.mask_all_path = self.cfg['mask_all_path']

        # import pdb
        # pdb.set_trace()

        image_files = self.cfg[mode]
        with open(image_files, 'r') as f:
            for line in f:
                self.image_paths.append(line.strip())

        gt_csv = self.cfg['gt_polygon_file']
        df = pd.read_csv(gt_csv, index_col=0, header=None)

        self.gt_polygons = {}
        for img_name in self.image_paths:
            img_name = img_name[:-4] # delete .tif
            num_vertices = df.loc[img_name][1] 
            polygon = np.array(df.loc[img_name][1:num_vertices * 2 + 1].tolist()).reshape(num_vertices, 2)
            polygon = polygon * self.final_size / self.ORIG_SIZE
            self.gt_polygons[img_name] = polygon

    def get_image(self, img_path):
        image = Image.open(img_path)
        image = transforms.functional.resize(image, self.final_size)
        return image


    def get_masks(self, mask_all_path, mask_one_path):
        image = Image.open(mask_all_path).convert('L') 
        mask_all = transforms.functional.resize(image, self.final_size)
        image = Image.open(mask_one_path).convert('L') 
        mask_one = transforms.functional.resize(image, self.final_size)
        return mask_all, mask_one

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_path, img_name)
        mask_one_path = os.path.join(self.mask_one_path, img_name)
        mask_all_path = os.path.join(self.mask_all_path, img_name)

        image = self.get_image(img_path)

        polygon = self.gt_polygons[img_name[:-4]].copy()
        mask_all, mask_one = self.get_masks(mask_all_path, mask_one_path)

        # Data augmentation
        if 'train' in self.mode:
            mask_collection = [mask_all, mask_one]
            image, mask_collection, polygon = self.random_flips(image, mask_collection, polygon)
            image, mask_collection, polygon = self.random_scale(image, mask_collection, polygon)

            # Randomly rotate image, with resolution of 0.01 radians
            rotation_angle = np.round(np.random.uniform(0, 2 * np.pi), decimals=2)
            image, mask_collection, polygon = self.rotate_image_masks_poly(
                image, mask_collection, polygon, rotation_angle)           
                
            mask_all, mask_one = mask_collection

        semantic_mask = self.get_semantic_mask(mask_all)

        # Compute distance transforms
        distance_transform_mask = mask_all
        distance_transform = self.get_distance_transform(distance_transform_mask)
        distance_transform_inside = self.get_distance_transform(distance_transform_mask, side='inside')
        distance_transform_outside = self.get_distance_transform(distance_transform_mask, side='outside')


        origin = self.midpoint
        init_contour = self.initialize_contour(origin=origin)
        init_contour0 = self.initialize_contour(origin=origin, ratio_radius=0.9)
        faces = self.get_faces(init_contour)

        # Get interpolated ground truth polygon
        gt_snake = self.interpolate_gt_snake(polygon)
        gt_snake_x = gt_snake[:, 0]
        gt_snake_y = gt_snake[:, 1]

        # Transform to torch tensors
        image = transforms.functional.to_tensor(image)
        mask_all = transforms.functional.to_tensor(mask_all)
        mask_one = transforms.functional.to_tensor(mask_one)
        semantic_mask = torch.from_numpy(semantic_mask).long()
        distance_transform = torch.from_numpy(distance_transform).float()
        distance_transform_inside = torch.from_numpy(distance_transform_inside).float()
        distance_transform_outside = torch.from_numpy(distance_transform_outside).float()
        init_contour = torch.from_numpy(init_contour).float()
        init_contour0 = torch.from_numpy(init_contour0).float()
        faces = torch.tensor(faces)
        gt_snake = torch.from_numpy(gt_snake).float()
        gt_snake_x = torch.from_numpy(gt_snake_x).float()
        gt_snake_y = torch.from_numpy(gt_snake_y).float()

        # Normalize image
        if self.normalize_flag:
            image = self.normalize(image)

        # NOTE: There is no nice way to include the ground truth polygon because 
        # they have variable length, so they cannot be placed into batches; 
        # include sequence id instead to have back reference
        sample = {
            'image': image,
            'mask_all': mask_all,
            'mask': mask_one,
            'semantic_mask': semantic_mask,
            'distance_transform': distance_transform,
            'distance_transform_inside': distance_transform_inside,
            'distance_transform_outside': distance_transform_outside,
            'init_contour': init_contour,
            'init_contour0': init_contour0,
            'faces': faces,
            'sequence_id': img_name[:-4],
            'gt_snake': gt_snake,
            'gt_snake_x': gt_snake_x,
            'gt_snake_y': gt_snake_y
        }
        return sample