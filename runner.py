import os
import sys
import importlib

from package.datasets import vaihingen, bing, inria
from package.models.drn_contours import DRNContours
from package import pretrain, train_baseline, train_contours, eval_contours
from package.utils.train_utils import overwrite_config
import package.config


def pretrain_vaihingen():
    importlib.reload(package.config)
    config = package.config.config
    checkpoint_path = train_baseline.run(config['pretrain_vaihingen'],
                                         config,
                                         vaihingen.VaihingenDataset,
                                         DRNContours)
    overwrite_config('vaihingen', 'restore', checkpoint_path)


def pretrain_bing():
    importlib.reload(package.config)
    config = package.config.config
    checkpoint_path = pretrain.run(config['pretrain_bing'],
                                   config,
                                   bing.BingDataset,
                                   DRNContours)
    overwrite_config('bing', 'restore', checkpoint_path)


def pretrain_inria():
    importlib.reload(package.config)
    config = package.config.config
    checkpoint_path = pretrain.run(config['pretrain_inria'],
                                   config,
                                   inria.InriaDataset,
                                   DRNContours)
    overwrite_config('inria', 'restore', checkpoint_path)


def train_vaihingen():
    importlib.reload(package.config)
    config = package.config.config
    checkpoint_path = train_contours.run(config['vaihingen'],
                                         config,
                                         vaihingen.VaihingenDataset,
                                         DRNContours)
    overwrite_config('vaihingen', 'eval_model', checkpoint_path)


def eval_vaihingen():
    importlib.reload(package.config)
    config = package.config.config
    print("Eval vaihingen")
    save_dir = eval_contours.run(config['vaihingen'],
                                 vaihingen.VaihingenDataset,
                                 DRNContours)


def train_bing():
    importlib.reload(package.config)
    config = package.config.config
    checkpoint_path = train_contours.run(config['bing'],
                                         config,
                                         bing.BingDataset,
                                         DRNContours)
    overwrite_config('bing', 'eval_model', checkpoint_path)


def eval_bing():
    importlib.reload(package.config)
    config = package.config.config
    print("Eval bing")
    save_dir = eval_contours.run(config['bing'],
                                 bing.BingDataset,
                                 DRNContours)


def train_inria():
    importlib.reload(package.config)
    config = package.config.config
    checkpoint_path = train_contours.run(config['inria'],
                                         config,
                                         inria.InriaDataset,
                                         DRNContours)
    overwrite_config('inria', 'eval_model', checkpoint_path)


def eval_inria():
    importlib.reload(package.config)
    config = package.config.config
    print("Eval inria")
    save_dir = eval_contours.run(config['inria'],
                                 inria.InriaDataset,
                                 DRNContours)


if __name__ == "__main__":
    # pretrain_vaihingen()
    # train_vaihingen()
    eval_vaihingen()

    # pretrain_bing()
    # train_bing()
    eval_bing()

    # pretrain_inria()
    # train_inria()
    eval_inria()

