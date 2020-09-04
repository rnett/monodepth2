# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from carla_dataset.config import Config, load_csv
from torch.utils.data import DataLoader
from typing import List

from carla_utils import get_params
from datasets.carla_dataset_loader import CarlaDataset
from layers import transformation_from_parameters
from networks.cube_poses import CubePosesAndLoss
from utils import readlines
from options import Mode, MonodepthOptions
from datasets import KITTIOdomDataset
import networks


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def get_gt_poses(configs: List[Config]):
    for config in configs:
        with config.pose_data as d:
            for j in range(d.absolute_pose.shape[0] - 2):
                i = j + 1
                start = d.absolute_pose[i]
                end = d.absolute_pose[i + 1]
                transform = end[:3] - start[:3]

                start_dir = start[3:]
                end_dir = end[3:]

                # http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
                angle = np.acos(np.dot(start_dir, end_dir))
                axis = np.cross(start_dir, end_dir)
                # normalize to unit vector
                axis = axis / np.linalg.norm(axis)
                yield transformation_from_parameters(angle * axis, transform)


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    conv_layer, data_lambda, intrinsics = get_params(opt)
    configs = load_csv(opt.test_data)
    dataset = CarlaDataset(configs, data_lambda, intrinsics,
                           [0, 1], 4, is_train=False, is_cubemap=opt.mode is Mode.Cubemap)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(conv_layer, opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(conv_layer, pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    if opt.mode is Mode.Cubemap:
        cube_poses = CubePosesAndLoss(include_loss=False)
        cube_poses.cuda()
        cube_poses.eval()

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation = pose_decoder(features)

            cam_T_cam = transformation_from_parameters(axisangle[:, 0], translation[:, 0])

            if opt.mode is Mode.Cubemap:
                cam_T_cam = cube_poses(cam_T_cam)

            pred_poses.append(cam_T_cam.cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    ates = []
    num_frames = pred_poses.shape[0]
    gt_poses = get_gt_poses(configs)
    for i in range(0, num_frames - 1):
        gt_pose = next(gt_poses)
        local_xyzs = np.array(dump_xyz(pred_poses[np.newaxis, i]))
        gt_local_xyzs = np.array(dump_xyz(gt_pose[np.newaxis, ...]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
