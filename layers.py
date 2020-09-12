# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from carla_dataset.data import Side

from networks.cube_padding import sides_from_batch, sides_to_batch
from options import Mode


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, conv_layer, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(conv_layer, in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, conv_layer, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        self.custom_padding = conv_layer is not torch.nn.Conv2d

        if self.custom_padding:
            self.conv = conv_layer(int(in_channels), int(out_channels), 3, padding=1)
        else:
            if use_refl:
                self.pad = nn.ReflectionPad2d(1)
            else:
                self.pad = nn.ZeroPad2d(1)

            self.conv = conv_layer(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        if self.custom_padding:
            out = x
        else:
            out = self.pad(x)

        out = self.conv(out)
        return out



class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud

    Axes of cloud are X is right, Y is down, Z is forward

    """

    def __init__(self, batch_size, height, width, mode: Mode):
        super(BackprojectDepth, self).__init__()

        if mode is Mode.Cubemap:
            batch_size *= 6

        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.mode = mode

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        # turns it into pinhole/cylindrical coords
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)

        # turns it into world coords
        if self.mode is Mode.Cylindrical:
            X = torch.sin(cam_points[:, 0:1, :])
            Y = cam_points[:, 1:2, :]
            Z = torch.cos(cam_points[:, 0:1, :])
            cam_points = torch.cat([X, Y, Z], dim=1) * depth.view(self.batch_size, 1, -1)
        elif self.mode is Mode.Cubemap:
            # same as pinhole
            cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        else:
            cam_points = depth.view(self.batch_size, 1, -1) * cam_points

        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

'''
Axes: X is Right, Y is Down, Z is Forward 

'''

def side_to_front(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, side: Side) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    if side is Side.Back:  # back: +/-180 Y
        new_X = -X
        new_Y = Y
        new_Z = -Z
    elif side is Side.Front:  # front: 0
        new_X = X
        new_Y = Y
        new_Z = Z
    elif side is Side.Top:  # top: -90 X
        new_X = X
        new_Y = -Z
        new_Z = Y
    elif side is Side.Bottom:  # bottom: +90 X
        new_X = X
        new_Y = Z
        new_Z = -Y
    elif side is Side.Left:  # left: +90 Y
        new_X = -Z
        new_Y = Y
        new_Z = X
    elif side is Side.Right:  # right: -90 Y
        new_X = Z
        new_Y = Y
        new_Z = -X
    else:
        raise ValueError

    return new_X, new_Y, new_Z


def front_to_side(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, side: Side) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    if side is Side.Back:  # back: +/-180 Y
        new_X = -X
        new_Y = Y
        new_Z = -Z
    elif side is Side.Front:  # front: 0
        new_X = X
        new_Y = Y
        new_Z = Z
    elif side is Side.Top:  # top: -90 X
        new_X = X
        new_Y = Z
        new_Z = -Y
    elif side is Side.Bottom:  # bottom: +90 X
        new_X = X
        new_Y = -Z
        new_Z = Y
    elif side is Side.Left:  # left: +90 Y
        new_X = Z
        new_Y = Y
        new_Z = -X
    elif side is Side.Right:  # right: -90 Y
        new_X = -Z
        new_Y = Y
        new_Z = X
    else:
        raise ValueError

    return new_X, new_Y, new_Z

def concat_coords(X, Y, Z):
    return torch.cat([X, Y, Z], dim=1)


def split_coords(P):
    return P[:, 0:1, :], P[:, 1:2, :], P[:, 2:3, :]


"""
Use BackProjectDepth w/ pinhole to get world coords for each image

Project3D:
Offset angle of coords properly for each side
Then do P matmul
Combine/stack into single sphere image, Convert to spherical coords
Find out-of-frame values, calculate frame offset & offset in target frame 
Convert to pinhole, un-offset-angle
Replace out-of-frame values with their offset in the target frame (converted to pinhole)
Do transforms to get pix values (this will transform out-of-frame values as if they are in-frame, which is the same 
as if they are in their target frame, as all frames are transformed identically)
Add frame offsets (in pix values) to out-of-frame values, concatenate & offset frames
"""


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, mode: Mode, eps=1e-7):
        super(Project3D, self).__init__()

        if mode is Mode.Cubemap:
            batch_size *= 6

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.mode = mode

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

    def forward(self, points, K, T):
        P = T[:, :3, :]

        # poses are already side-pov
        cam_points = torch.matmul(P, points)

        if self.mode is Mode.Cylindrical:
            X = cam_points[:, 0:1, :]
            Y = cam_points[:, 1:2, :]
            Z = cam_points[:, 2:3, :]

            h = Y / (torch.sqrt(X * X + Z * Z) + self.eps)
            theta = torch.atan2(X, Z)
            pix_coords = torch.cat([theta, h], dim=1)
        elif self.mode is Mode.Cubemap:
            side_coords = sides_from_batch(cam_points)
            world_coords = [concat_coords(*side_to_front(*split_coords(coords), side=side)) for coords, side in zip(side_coords, list(Side))]

            offsets = []

            for coords, side in zip(side_coords, list(Side)):
                side_world_coords = world_coords[side.value]

                mags = torch.argmax(torch.abs(side_world_coords), dim=1, keepdim=True)

                mag_values = torch.sign(side_world_coords).permute(0, 2, 1).gather(dim=-1, index=mags.permute(0, 2, 1)).squeeze(dim=2)

                mags = (mags + 1) * mag_values.unsqueeze(1)

                # Axes: X is Right, Y is Down, Z is Forward
                # 1 -> +X -> Right, -1 -> -X -> Left, 2 -> Y -> Bottom, -2 -> -Y -> Top, 3 -> Z -> Front, -3 -> -Z -> Back
                # side -> mags value
                indices = {Side.Top: -2, Side.Bottom: 2, Side.Left: -1, Side.Right: 1, Side.Front: 3, Side.Back: -3}
                sides = torch.zeros(size=(self.batch_size // 6, 1, self.height * self.width), dtype=torch.float).to(points.device) + side.value

                split_world_coords = split_coords(side_world_coords)

                for target_side in list(Side):
                    if target_side is side:
                        continue

                    mask = mags == indices[target_side]
                    if mask.any():
                        sides[mask] = target_side.value
                        all_mask = mask.repeat(1, 3, 1)
                        coords[all_mask] = concat_coords(*front_to_side(*split_world_coords, side=target_side))[all_mask]

                offsets.append(sides)

            for i in range(len(offsets)):
                offsets[i] = offsets[i].view(self.batch_size // 6, self.height, self.width)

            # side_coords is the in-side location, offsets is which side (absolute)
            # offset should be the current side for most

            # treat like pinhole (except add offset at the end)

            cam_points = sides_to_batch(*side_coords)
            pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        else:
            # [x, y] = [x, y] / z
            pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        # [x, y] *= [f_x, f_y]
        # converts from pinhole/cylindrical to pix coords
        pix_coords = torch.cat([pix_coords, self.ones], dim=1)
        pix_coords = torch.matmul(K[:, :3, :3], pix_coords)
        pix_coords = pix_coords[:, :2, :]

        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)

        pix_coords = pix_coords.permute(0, 2, 3, 1)

        # X is first, but Width is second

        if self.mode is Mode.Cubemap:
            # concat along width
            offsets = torch.cat(offsets, dim=2)
            sides = sides_from_batch(pix_coords)
            pix_coords = torch.cat(sides, dim=2)
            pix_coords[..., 0] += (offsets * self.width)

        # change from a 0-width range to a -1-1 range.
        # x is 0, but width is 1
        pix_coords[..., 0] /= self.width * (6 if self.mode is Mode.Cubemap else 1) - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
