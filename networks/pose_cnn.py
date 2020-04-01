# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from typing import ClassVar

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, conv_layer, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.conv_0 = conv_layer(3 * num_input_frames, 16, 7, 2, 3)
        self.conv_1 = conv_layer(16, 32, 5, 2, 2)
        self.conv_2 = conv_layer(32, 64, 3, 2, 1)
        self.conv_3 = conv_layer(64, 128, 3, 2, 1)
        self.conv_4 = conv_layer(128, 256, 3, 2, 1)
        self.conv_5 = conv_layer(256, 256, 3, 2, 1)
        self.conv_6 = conv_layer(256, 256, 3, 2, 1)

        self.pose_conv = conv_layer(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = 7

        self.relu = nn.ReLU(True)

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.__getattr__(f"conv_{i}")(out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
