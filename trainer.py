# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path

import carla_dataset
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from carla_dataset.config import load_csv, Config
from carla_dataset.data import Side
from carla_dataset.intrinsics import Pinhole90Intrinsics, PinholeIntrinsics, CylindricalIntrinsics
from imageio import imwrite
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from tqdm import tqdm, trange

from carla_utils import convert_to_cubemap_batch, get_datasets, get_params
from datasets.carla_dataset_loader import CarlaDataset
from depth_utils import normalize_depth_for_display
from networks.cube_padding import CubicConv2d, sides_from_batch
from networks.cube_poses import CubePosesAndLoss
from networks.cylindrical_padding import CylindricalConv2d
from options import Mode
from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        Path(self.log_path).mkdir(exist_ok=True, parents=True)
        (Path(self.log_path) / "command").open('w+').write(" ".join(sys.argv))

        # checking height and width are multiples of 32
        # assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        # assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.parallel = not self.opt.no_cuda and torch.cuda.device_count() > 1

        if self.parallel and self.opt.mode is Mode.Cubemap:
            assert self.opt.batch_size % torch.cuda.device_count() == 0, f"Cubemap batch size ({self.opt.batch_size})" \
                                                                         f" must be evenly divisible by the number of" \
                                                                         f" GPUs ({torch.cuda.device_count()})"

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        conv_layer, data_lambda, intrinsics = get_params(options)
        self.intrinsics = intrinsics

        self.height = self.opt.height or self.intrinsics.height
        self.width = self.opt.width or self.intrinsics.width

        self.models["encoder"] = networks.ResnetEncoder(conv_layer,
                                                        self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.store_model("encoder")

        self.models["depth"] = networks.DepthDecoder(conv_layer,
                                                     self.get_num_ch_enc(self.models["encoder"]), self.opt.scales)
        self.store_model("depth")

        if self.use_pose_net:  # true
            if self.opt.pose_model_type == "separate_resnet":  # true
                self.models["pose_encoder"] = networks.ResnetEncoder(conv_layer,
                                                                     self.opt.num_layers,
                                                                     self.opt.weights_init == "pretrained",
                                                                     num_input_images=self.num_pose_frames)
                self.store_model("pose_encoder")

                self.models["pose"] = networks.PoseDecoder(conv_layer,
                                                           self.get_num_ch_enc(self.models["pose_encoder"]),
                                                           num_input_features=1,
                                                           num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(conv_layer,
                                                           self.get_num_ch_enc(self.models["encoder"]),
                                                           self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(conv_layer,
                                                       self.num_input_frames if self.opt.pose_model_input == "all"
                                                       else 2)

            self.store_model("pose")

        if self.opt.predictive_mask:  # false
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(conv_layer,
                                                                   self.get_num_ch_enc(self.models["encoder"]),
                                                                   self.opt.scales,
                                                                   num_output_channels=(len(self.opt.frame_ids) - 1))
            self.store_model("predictive_mask")

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ",
              f"{self.device}" + (f" on {torch.cuda.device_count()} GPUs" if self.parallel else ""))

        num_train_samples = len(load_csv(options.train_data)) * 1000
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset, val_dataset = get_datasets(options, data_lambda, intrinsics)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = self.wrap_model(SSIM())  # TODO can I parallelize?
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            # TODO should be able to paralalize
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w, options.mode)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w, options.mode)
            self.project_3d[scale].to(self.device)

        if options.mode is Mode.Cubemap:
            self.models["cube_pose_and_loss"] = self.wrap_model(CubePosesAndLoss())
            self.models["cube_pose_and_loss"].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.train_items = len(train_dataset)
        self.val_items = len(val_dataset)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            self.train_items, self.val_items))

        self.save_opts()

    def store_model(self, name):
        if self.parallel:
            self.models[name] = nn.DataParallel(self.models[name])

        self.models[name].to(self.device)
        self.parameters_to_train += list(self.models[name].parameters())

    def wrap_model(self, model):
        if self.parallel:
            return nn.DataParallel(model)
        else:
            return model

    def get_num_ch_enc(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module.num_ch_enc
        else:
            return model.num_ch_enc

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in trange(self.opt.num_epochs, desc="Epochs", ncols=200):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        # print("Training Epoch", self.step)
        self.set_train()

        pbar = tqdm(total=self.train_items, desc="Batches", ncols=200)

        batches = len(self.train_loader)
        for batch_idx, inputs in enumerate(self.train_loader):
            self.model_optimizer.zero_grad()

            if self.opt.mode is Mode.Cubemap:
                inputs = convert_to_cubemap_batch(inputs, self.opt.frame_ids, self.opt.scales)

            before_op_time = time.time()

            # with torch.autograd.detect_anomaly():
            outputs, losses = self.process_batch(inputs, batch_idx if batch_idx % 200 == 0 else 0)

            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            pbar.set_postfix_str(f"Epoch {self.epoch}, Loss: {losses['loss'].cpu().data:.6f}", refresh=True)

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            pbar.update(self.opt.batch_size)

        pbar.close()
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, save_images=0):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, save_images)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:  # true
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    cam_T_cam = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                    if self.opt.mode is Mode.Cubemap:
                        outputs[("cube_pose_loss", 0, f_i)], outputs[("cam_T_cam", 0, f_i)] = self.models[
                            "cube_pose_and_loss"](cam_T_cam)
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = cam_T_cam

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    cam_T_cam = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

                    if self.opt.mode is Mode.Cubemap:
                        outputs[("cube_pose_loss", 0, f_i)], outputs[("cam_T_cam", 0, f_i)] = self.models[
                            "cube_pose_and_loss"](cam_T_cam)
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = cam_T_cam

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        # TODO could have something to do with the swap here
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        if self.opt.mode is Mode.Cubemap:
            inputs = convert_to_cubemap_batch(inputs, self.opt.frame_ids, self.opt.scales)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.height, self.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":  # false
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                    if self.opt.mode is Mode.Cubemap:
                        _, T = self.models["cube_pose_and_loss"](T)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                # outputs[("sample", frame_id, scale)] = pix_coords

                if self.opt.mode is Mode.Cubemap:
                    color = inputs[("color", frame_id, source_scale)]
                    sides = sides_from_batch(color)
                    color = torch.cat(sides, dim=3)
                else:
                    color = inputs[("color", frame_id, source_scale)]

                sampled = F.grid_sample(
                    color,
                    pix_coords,
                    padding_mode="border")

                # undo concating along width
                # original = color[0]
                # transformed = sampled[0]
                # original = original.detach().cpu().permute(1, 2, 0).numpy()
                # transformed = transformed.detach().cpu().permute(1, 2, 0).numpy()
                # imwrite("/home/rnett/original.png", original)
                # imwrite("/home/rnett/transformed.png", transformed)

                if self.opt.mode is Mode.Cubemap:
                    width = sampled.shape[3] // 6
                    sides = [sampled[:, :, :, width * j: width * (j + 1)] for j in range(6)]
                    sampled = sides_to_batch(*sides)

                outputs[("color", frame_id, scale)] = sampled

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, save_images=0):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        # TODO pose loss will be the same for every scale, could pull out of loop
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            idx = 0 if not self.opt.mode is Mode.Cubemap else 4
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]

                if save_images > 0 and scale == 0:
                    dir = Path(self.log_path) / f"images/epoch_{self.epoch}/batch_{save_images}/frame_{frame_id}"

                    original = inputs[("color", frame_id, source_scale)]
                    dir.mkdir(parents=True, exist_ok=True)

                    p = pred.detach()[idx, ...].permute(1, 2, 0) * 256
                    t = target.detach()[idx, ...].permute(1, 2, 0) * 256
                    o = original.detach()[idx, ...].permute(1, 2, 0) * 256

                    imwrite(dir / f"pred.png", p.cpu().detach().numpy())
                    imwrite(dir / f"target.png", t.cpu().detach().numpy())
                    imwrite(dir / f"original.png", o.cpu().detach().numpy())

                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            if save_images > 0 and scale == 0:
                dir = Path(self.log_path) / f"images/epoch_{self.epoch}/batch_{save_images}/"
                d, _ = disp_to_depth(disp.detach()[idx, ...], self.opt.min_depth, self.opt.max_depth)
                imwrite(dir / f"depth.png", normalize_depth_for_display(d.cpu().detach().squeeze().numpy()))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.height, self.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).to(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        # if save_images:
        #     print()

        total_loss /= self.num_scales

        if self.opt.mode is Mode.Cubemap:
            pose_loss = 0
            for frame_id in self.opt.frame_ids[1:]:
                pose_loss = pose_loss + outputs[("cube_pose_loss", 0, frame_id)].mean()

            pose_loss = pose_loss * self.opt.cube_pose_loss_factor

            losses["loss/cube_pose_loss"] = pose_loss

            total_loss += pose_loss

        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        # print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
        #                           sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer: SummaryWriter = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(6, self.opt.batch_size)):  # write a maxmimum of six images (for cubemap support)
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.height
                to_save['width'] = self.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
