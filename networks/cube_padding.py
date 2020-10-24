from __future__ import annotations

from typing import Optional, Tuple, Union

import carla_dataset
import cv2
import numpy as np
import torch
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torch import Tensor, nn


def sides_from_batch(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''

    :param x:
    :return: top, bottom, left, right, front, back, each with the same shape as x except 1/6th the batch dim
    '''
    if x.shape[0] % 6 != 0:
        raise ValueError(
            f"Batch dimension of cube tensor most be a multiple of 6, got {x.shape[0]}"
            f"{'' if torch.cuda.device_count() <= 1 else '  Ensure that the batch on each GPU is a multiple of 6.'}")

    shape = list(x.shape)
    shape[0] //= 6
    shape.insert(1, 6)

    x = x.reshape(shape).transpose(0, 1)
    return x[0], x[1], x[2], x[3], x[4], x[5]


def sides_to_batch(top: Tensor, bottom: Tensor, left: Tensor, right: Tensor, front: Tensor, back: Tensor) -> Tensor:
    assert top.shape == bottom.shape and top.shape == left.shape and top.shape == right.shape and top.shape == \
           front.shape and top.shape == back.shape

    shape = list(top.shape)
    shape[0] *= 6

    return torch.stack([top, bottom, left, right, front, back], dim=0).transpose(0, 1).reshape(shape)


class Side:
    def __init__(self, x: Optional[Tensor]):
        self.x = x
        self.is_none = self.x is None

    def rot90(self) -> Side:
        if self.is_none:
            return Side(None)

        return Side(self.x.transpose(2, 3).flip(3))

    def rotNeg90(self) -> Side:
        if self.is_none:
            return Side(None)

        return Side(self.x.transpose(2, 3).flip(2))  # should it be flip(2, 3)?  Docs said just 3

    def flipVertical(self) -> Side:
        if self.is_none:
            return Side(None)

        return Side(self.x.flip(2))

    def flipHorizontal(self) -> Side:
        if self.is_none:
            return Side(None)

        return Side(self.x.flip(3))

    def left_strip(self, size) -> Side:
        if size == 0:
            return Side(None)

        return Side(self.x[:, :, :, :size])

    def right_strip(self, size) -> Side:
        if size == 0:
            return Side(None)

        return Side(self.x[:, :, :, -size:])

    def top_strip(self, size) -> Side:
        if size == 0:
            return Side(None)

        return Side(self.x[:, :, :size, :])

    def bottom_strip(self, size) -> Side:
        if size == 0:
            return Side(None)

        return Side(self.x[:, :, -size:, :])


def pad_side(x: Union[Side, Tensor], top_strip: Union[Side, Tensor], bottom_strip: Union[Side, Tensor],
             left_strip: Union[Side, Tensor], right_strip: Union[Side, Tensor]) -> Tensor:
    if isinstance(x, Side):
        x = x.x
    if isinstance(top_strip, Side):
        top_strip = top_strip.x
    if isinstance(bottom_strip, Side):
        bottom_strip = bottom_strip.x
    if isinstance(left_strip, Side):
        left_strip = left_strip.x
    if isinstance(right_strip, Side):
        right_strip = right_strip.x

    vertical = torch.cat([top_strip, x, bottom_strip], dim=2)

    if top_strip.shape[2] >= left_strip.shape[3]:
        top_left_corner = top_strip[:, :, :, 0:1].repeat(1, 1, 1, left_strip.shape[3])
    else:
        top_left_corner = left_strip[:, :, 0:1, :].repeat(1, 1, top_strip.shape[2], 1)

    if bottom_strip.shape[2] >= left_strip.shape[3]:
        bottom_left_corner = bottom_strip[:, :, :, 0:1].repeat(1, 1, 1, left_strip.shape[3])
    else:
        bottom_left_corner = left_strip[:, :, -1:, :].repeat(1, 1, bottom_strip.shape[2], 1)

    if top_strip.shape[2] >= right_strip.shape[3]:
        top_right_corner = top_strip[:, :, :, -1:].repeat(1, 1, 1, right_strip.shape[3])
    else:
        top_right_corner = right_strip[:, :, 0:1, :].repeat(1, 1, top_strip.shape[2], 1)

    if bottom_strip.shape[2] >= right_strip.shape[3]:
        bottom_right_corner = bottom_strip[:, :, :, -1:].repeat(1, 1, 1, right_strip.shape[3])
    else:
        bottom_right_corner = right_strip[:, :, -1:, :].repeat(1, 1, bottom_strip.shape[2], 1)

    full_right = torch.cat([top_right_corner, right_strip, bottom_right_corner], dim=2)
    full_left = torch.cat([top_left_corner, left_strip, bottom_left_corner], dim=2)

    return torch.cat([full_left, vertical, full_right], dim=3)


def cube_pad(x: Tensor, pad_size) -> Tensor:
    top, bottom, left, right, front, back = sides_from_batch(x)
    top = Side(top)
    bottom = Side(bottom)
    left = Side(left)
    right = Side(right)
    front = Side(front)
    back = Side(back)

    left_size, right_size, top_size, bottom_size = get_pad_size(pad_size)

    if left_size == 0 and right_size == 0 and top_size == 0 and bottom_size == 0:
        return x

    padded_top = pad_side(top, back.top_strip(top_size).flipVertical(), front.top_strip(bottom_size),
                          left.top_strip(left_size).rot90(), right.top_strip(right_size).rotNeg90())
    padded_bottom = pad_side(bottom, front.bottom_strip(top_size), back.bottom_strip(bottom_size).flipVertical(),
                             left.bottom_strip(left_size).rotNeg90(), right.bottom_strip(right_size).rot90())
    padded_left = pad_side(left, top.left_strip(top_size).rotNeg90(), bottom.left_strip(bottom_size).rot90(),
                           back.right_strip(left_size), front.left_strip(right_size))
    padded_right = pad_side(right, top.right_strip(top_size).rot90(), bottom.right_strip(bottom_size).rotNeg90(),
                            front.right_strip(left_size), back.left_strip(right_size))
    padded_front = pad_side(front, top.bottom_strip(top_size), bottom.top_strip(bottom_size),
                            left.right_strip(left_size), right.left_strip(right_size))
    padded_back = pad_side(back, top.top_strip(top_size).flipVertical(),
                           bottom.bottom_strip(bottom_size).flipVertical(),
                           right.right_strip(left_size), left.left_strip(right_size))
    return sides_to_batch(padded_top, padded_bottom, padded_left, padded_right, padded_front, padded_back)


class CubicConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='cubic'):
        self.cube_padding = padding_mode == "cubic" and padding > 0
        self.amount_cube_padding = padding

        if self.cube_padding:
            padding_mode = "zeros"
            padding = 0

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cube_padding:
            return super().forward(cube_pad(input, self.amount_cube_padding))
        else:
            return super().forward(input)


def get_pad_size(lrtd_pad):
    if type(lrtd_pad) == np.int:
        p_l = lrtd_pad
        p_r = lrtd_pad
        p_t = lrtd_pad
        p_d = lrtd_pad
    else:
        [p_l, p_r, p_t, p_d] = lrtd_pad
    return p_l, p_r, p_t, p_d

if __name__ == '__main__':
    sides = ['top', 'bottom', 'left', 'right', 'front', 'back']
    batch = []
    for i in range(10):
        all_sides = []
        for s in sides:
            img = imread(f"E:\\carla\\town01\\clear\\noon\\cars_30_peds_200_index_0\\raw\\frames\\{s}_{i}_rgb.png")[..., :3]

            img = carla_dataset.data.crop_pinhole_to_90(img)
            # plt.imshow(img)
            # plt.show()
            all_sides.append(img)

        batch.append(np.stack(all_sides, axis=0))

    batch = torch.from_numpy(np.concatenate(batch, axis=0)).permute(0, 3, 1, 2)

    # s = Side(batch)
    # r = s.flipHorizontal()

    padded = cube_pad(batch, 30)

    # together = torch.cat([s.x, r.x], dim=3)

    for image in padded:
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.show()
        print()

    print()
