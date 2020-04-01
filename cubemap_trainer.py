# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import carla_dataset
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from carla_dataset.config import load_csv
from carla_dataset.data import Side
from carla_dataset.intrinsics import PinholeIntrinsics
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from datasets.carla_dataset import CarlaDataset, just_side
from trainer import Trainer, get_pinhole_front
from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

#TODO cube padding
class CubemapTrainer(Trainer):
    def __init__(self, options):
        super().__init__(options)

    def get_datasets(self, options, data_lambda, intrinsics):
        train_dataset = CarlaDataset(load_csv(options.train_data), data_lambda, intrinsics,
                                     self.opt.frame_ids, 4, is_train=True, is_cubemap=True)

        val_dataset = CarlaDataset(load_csv(options.val_data), data_lambda, intrinsics,
                                   self.opt.frame_ids, 4, is_train=True, is_cubemap=True)

        return train_dataset, val_dataset

    def get_params(self, options):
        return nn.Conv2d, get_pinhole_front, PinholeIntrinsics()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, all_inputs in enumerate(self.train_loader):

            sides_inputs = {s: just_side(s, all_inputs) for s in list(Side)}

            before_op_time = time.time()
            outputs = {}
            losses = {}
            for s, inputs in sides_inputs.items():
                res = self.process_batch(inputs)
                outputs[s] = res[0]
                losses[s] = res[1]

            cube_losses = self.compute_cubemap_losses(outputs)

            #TODO make total loss first?

            self.model_optimizer.zero_grad()
            cube_losses.backwards()

            total_loss = cube_losses
            for loss in losses.values():
                loss["loss"].backward()
                total_loss += loss["loss"]

            self.model_optimizer.step()
            self.model_lr_scheduler.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, total_loss.cpu().data)

                #TODO everything here on down needs to use cubemap properly

                for s, inputs in sides_inputs.items():
                    if "depth_gt" in inputs:
                        self.compute_depth_losses(inputs, outputs[s], losses[s])
                    self.log(f"train_{s.name.lower()}", inputs, outputs[s], losses[s])

                self.val()

            self.step += 1

    def compute_cubemap_losses(self, outputs):
        pass

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            for s in list(Side):
                side_inputs = just_side(s, inputs)
                outputs, losses = self.process_batch(side_inputs)

                if "depth_gt" in side_inputs:
                    self.compute_depth_losses(side_inputs, outputs, losses)

                self.log(f"val_{s.name.lower()}", side_inputs, outputs, losses)
                del outputs, losses
            del inputs

        self.set_train()

