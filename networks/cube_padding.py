from __future__ import annotations
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Parameter


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
    shape[0] /= 6
    shape.insert(0, 6)

    x = x.reshape(shape)
    return x[0], x[1], x[2], x[3], x[4], x[5]


def sides_to_batch(top: Tensor, bottom: Tensor, left: Tensor, right: Tensor, front: Tensor, back: Tensor) -> Tensor:
    return torch.stack([top, bottom, left, right, front, back], dim=0)

class Side:
    def __init__(self, x: Tensor):
        self.x = x

    def rot90(self) -> Side:
        return Side(self.x.transpose(2, 3))

    def rotNeg90(self) -> Side:
        return Side(self.x.transpose(2, 3).flip(3)) # should it be flip(1, 2)?

    def flipVertical(self) -> Side:
        return Side(self.x.flip(1))

    def flipHorizontal(self) -> Side:
        return Side(self.x.flip(3))

    def left_strip(self, size) -> Side:
        return Side(self.x[:, :, :, :size])

    def right_strip(self, size) -> Side:
        return Side(self.x[:, :, :, -size:])

    def top_strip(self, size) -> Side:
        return Side(self.x[:, :, :, :size])

    def bottom_strip(self, size) -> Side:
        return Side(self.x[:, :, :, -size:])


def pad_side(x: Union[Side, Tensor], top_strip: Union[Side, Tensor], bottom_strip: Union[Side, Tensor], left_strip: Union[Side, Tensor], right_strip: Union[Side, Tensor]) -> Tensor:

    if isinstance(x, Side):
        x = x.x
    if isinstance(top_strip, Side):
        top_strip = top_strip.x
    if isinstance(bottom_strip, Side):
        bottom_strip = bottom_strip.x
    if isinstance(left_strip, Side):
        left_strip = left_strip.x
    if isinstance(x, Side):
        right_strip = right_strip.x

    vertical = torch.cat([top_strip, x, bottom_strip], dim=2)

    if top_strip.shape[1] >= left_strip.shape[2]:
        top_left_corner = top_strip[:, :, :, 0:1].repeat(1, 1, 1, left_strip.shape[2])
    else:
        top_left_corner = left_strip[:, :, 0:1, :].repeat(1, 1, top_strip.shape[1], 1)

    if bottom_strip.shape[1] >= left_strip.shape[2]:
        bottom_left_corner = bottom_strip[:, :, :, 0:1].repeat(1, 1, 1, left_strip.shape[2])
    else:
        bottom_left_corner = left_strip[:, :, -1:, :].repeat(1, 1, bottom_strip.shape[1], 1)

    if top_strip.shape[1] >= right_strip.shape[2]:
        top_right_corner = top_strip[:, :, :, -1:].repeat(1, 1, 1, right_strip.shape[2])
    else:
        top_right_corner = right_strip[:, :, 0:1, :].repeat(1, 1, top_strip.shape[1], 1)

    if bottom_strip.shape[1] >= right_strip.shape[2]:
        bottom_right_corner = bottom_strip[:, :, :, -1:].repeat(1, 1, 1, right_strip.shape[2])
    else:
        bottom_right_corner = right_strip[:, :, -1:, :].repeat(1, 1, bottom_strip.shape[1], 1)

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
    padded_back = pad_side(back, top.top_strip(top_size).flipVertical(), bottom.bottom_strip(bottom_size).flipVertical(),
                           right.right_strip(left_size), left.left_strip(right_size))
    return sides_to_batch(padded_top, padded_bottom, padded_left, padded_right, padded_front, padded_back)

class CubicConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='cubic'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "cubic":
            return super().forward(cube_pad(input, 1))
        else:
            return super().forward(input)


# Everything below is from https://github.com/hsientzucheng/CP-360-Weakly-Supervised-Saliency/blob/master/model
# /cube_pad.py

def get_pad_size(lrtd_pad):
    if type(lrtd_pad) == np.int:
        p_l = lrtd_pad
        p_r = lrtd_pad
        p_t = lrtd_pad
        p_d = lrtd_pad
    else:
        [p_l, p_r, p_t, p_d] = lrtd_pad
    return p_l, p_r, p_t, p_d


class CubePad(nn.Module):
    def __init__(self, lrtd_pad, use_gpu=True):
        super(CubePad, self).__init__()
        self.CP = CubePadding(lrtd_pad, use_gpu)

    def forward(self, x):
        """
            Input shape:  [6N, C, H, W]
            Output shape: [6N, C, H + (top down padding), W + (left right padding)]
        """
        if x.size()[0] % 6 != 0:
            print('CubePad size mismatch!')
            exit()
        batch_size = int(x.size()[0] / 6)
        tmp = []
        for i in range(batch_size):
            patch = x[i * 6:i * 6 + 6, :, :, :]
            tmp.append(self.CP(patch))
        result = torch.cat(tmp, dim=0)
        return result


class CubePadding(nn.Module):
    """
        Cube padding support astmetric padding and rectangle input
        Order of cube faces: 123456 => bdflrt (back, bottom, front, left, right, top)
        The surrounding volume of cube padding includes 4 concatenated plates
                                  //＝＝＝//|
        4 plates (t, d, l, r):   //  t  // |
                                ||＝＝＝|| r|
                               l||  f  || /
                                ||＝＝＝||/
                                   d
    """

    def __init__(self, lrtd_pad, use_gpu=True):
        super(CubePadding, self).__init__()
        self.use_gpu = use_gpu
        # self.pad = pad
        if type(lrtd_pad) == np.int:
            self.p_l = lrtd_pad
            self.p_r = lrtd_pad
            self.p_t = lrtd_pad
            self.p_d = lrtd_pad
        else:
            [self.p_l, self.p_r, self.p_t, self.p_d] = lrtd_pad

    def flip(self, tensor, dim):
        idx = [i for i in range(tensor.size(dim) - 1, -1, -1)]

        if self.use_gpu:
            idx = Parameter(torch.LongTensor(idx), requires_grad=False)
        else:
            idx = Parameter(torch.LongTensor(idx), requires_grad=False)

        inverted_tensor = tensor.index_select(dim, idx)
        return inverted_tensor

    def make_cubepad_edge(self, feat_td, feat_lr):
        td_pad = feat_td.size(2)
        lr_pad = feat_lr.size(3)

        if td_pad > lr_pad:
            return feat_lr.repeat(1, 1, td_pad, 1)
        else:
            return feat_td.repeat(1, 1, 1, lr_pad)

        # avg_feat = (tile_lr+tile_td)*0.5
        # return avg_feat

    def forward(self, x):
        """
            Input shape:  [6, C, H, W]
            Output shape: [6, C, H + p_t + p_d, W + p_l + p_r]
            Method: Create 4 plates -> Create corners -> Concatenate
        """
        p_l = self.p_l
        p_r = self.p_r
        p_t = self.p_t
        p_d = self.p_d

        f_back = x[0]
        f_down = x[1]
        f_front = x[2]
        f_left = x[3]
        f_right = x[4]
        f_top = x[5]

        # Construct top, down, left, right padding volume if needed
        if p_t != 0:
            _t12 = torch.cat(
                [torch.unsqueeze(self.flip(f_top[:, :p_t, :], 2), 0),
                 torch.unsqueeze(f_front[:, -p_t:, :], 0)], 0)
            _t123 = torch.cat(
                [_t12, torch.unsqueeze(f_top[:, -p_t:, :], 0)], 0)
            _t1234 = torch.cat(
                [_t123, torch.unsqueeze(f_top[:, :, :p_t].permute(0, 2, 1), 0)], 0)
            _t12345 = torch.cat(
                [_t1234, torch.unsqueeze(
                    self.flip((f_top[:, :, -p_t:].permute(0, 2, 1)), 2), 0)], 0)
            _t123456 = torch.cat(
                [_t12345, torch.unsqueeze(self.flip(f_back[:, :p_t, :], 2), 0)], 0)
        if p_d != 0:
            _d12 = torch.cat(
                [torch.unsqueeze(self.flip(f_down[:, -p_d:, :], 2), 0),
                 torch.unsqueeze(self.flip(f_back[:, -p_d:, :], 2), 0)], 0)
            _d123 = torch.cat(
                [_d12, torch.unsqueeze(f_down[:, :p_d, :], 0)], 0)
            _d1234 = torch.cat(
                [_d123, torch.unsqueeze(self.flip(f_down[:, :, :p_d].permute(0, 2, 1), 2), 0)], 0)
            _d12345 = torch.cat(
                [_d1234, torch.unsqueeze(f_down[:, :, -p_d:].permute(0, 2, 1), 0)], 0)
            _d123456 = torch.cat(
                [_d12345, torch.unsqueeze(f_front[:, :p_d, :], 0)], 0)
        if p_l != 0:
            _l12 = torch.cat(
                [torch.unsqueeze(f_right[:, :, -p_l:], 0),
                 torch.unsqueeze(self.flip(f_left[:, -p_l:, :].permute(0, 2, 1), 1), 0)], 0)
            _l123 = torch.cat(
                [_l12, torch.unsqueeze(f_left[:, :, -p_l:], 0)], 0)
            _l1234 = torch.cat(
                [_l123, torch.unsqueeze(f_back[:, :, -p_l:], 0)], 0)
            _l12345 = torch.cat(
                [_l1234, torch.unsqueeze(f_front[:, :, -p_l:], 0)], 0)
            _l123456 = torch.cat(
                [_l12345, torch.unsqueeze(f_left[:, :p_l, :].permute(0, 2, 1), 0)], 0)
        if p_r != 0:
            _r12 = torch.cat(
                [torch.unsqueeze(f_left[:, :, :p_r], 0),
                 torch.unsqueeze(f_right[:, -p_r:, :].permute(0, 2, 1), 0)], 0)
            _r123 = torch.cat(
                [_r12, torch.unsqueeze(f_right[:, :, :p_r], 0)], 0)
            _r1234 = torch.cat(
                [_r123, torch.unsqueeze(f_front[:, :, :p_r], 0)], 0)
            _r12345 = torch.cat(
                [_r1234, torch.unsqueeze(f_back[:, :, :p_r], 0)], 0)
            _r123456 = torch.cat(
                [_r12345, torch.unsqueeze(self.flip(f_right[:, :p_r, :].permute(0, 2, 1), 1), 0)], 0)

        # For edge corner
        if p_r != 0 and p_t != 0:
            p_tr = self.make_cubepad_edge(
                _t123456[:, :, -p_t:, -1:], _r123456[:, :, :1, :p_r])
        if p_t != 0 and p_l != 0:
            p_tl = self.make_cubepad_edge(
                _t123456[:, :, :p_t, :1], _l123456[:, :, :1, :p_l])
        if p_d != 0 and p_r != 0:
            p_dr = self.make_cubepad_edge(
                _d123456[:, :, -p_d:, -1:], _r123456[:, :, -1:, -p_r:])
        if p_d != 0 and p_l != 0:
            p_dl = self.make_cubepad_edge(
                _d123456[:, :, :p_d, :1], _l123456[:, :, -1:, -p_l:])

        # Concatenate each padding volume
        if p_r != 0:
            _rp123456p = _r123456
            if 'p_tr' in locals():
                _rp123456 = torch.cat([p_tr, _r123456], 2)
            else:
                _rp123456 = _r123456

            if 'p_dr' in locals():
                _rp123456p = torch.cat([_rp123456, p_dr], 2)
            else:
                _rp123456p = _rp123456
        if p_l != 0:
            _lp123456p = _l123456
            if 'p_tl' in locals():
                _lp123456 = torch.cat([p_tl, _l123456], 2)
            else:
                _lp123456 = _l123456
            if 'p_dl' in locals():
                _lp123456p = torch.cat([_lp123456, p_dl], 2)
            else:
                _lp123456p = _lp123456
        if p_t != 0:
            t_out = torch.cat([_t123456, x], 2)
        else:
            t_out = x
        if p_d != 0:
            td_out = torch.cat([t_out, _d123456], 2)
        else:
            td_out = t_out
        if p_l != 0:
            tdl_out = torch.cat([_lp123456p, td_out], 3)
        else:
            tdl_out = td_out
        if p_r != 0:
            tdlr_out = torch.cat([tdl_out, _rp123456p], 3)
        else:
            tdlr_out = tdl_out
        return tdlr_out
