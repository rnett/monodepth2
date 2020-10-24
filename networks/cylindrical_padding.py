import torch
import torch.nn.functional as F

from math import floor, ceil

from torch import nn


class CylindricalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='cylindrical'):
        self.cylindrical_padding = padding_mode == "cylindrical" and padding > 0
        self.amount_cylindrical_padding = padding

        if self.cylindrical_padding:
            padding_mode = "zeros"
            padding = 0

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cylindrical_padding:
            wrapped_inputs = wrap_pad(input, self.amount_cylindrical_padding)
            return super().forward(wrapped_inputs)
        else:
            return super().forward(input)




def wrap_pad(tensor: torch.Tensor, wrap_padding, axis=(2, 3)):
    """Apply cylindrical wrapping to one axis and zero padding to another.
    By default, this wraps horizontally and pads vertically. The axes can be
    set with the `axis` keyword, and the wrapping/padding amount can be set
    with the `wrap_pad` keyword.
    """
    rank = tensor.ndimension()
    if axis[0] >= rank or axis[1] >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    # handle single-number wrap/pad input
    if isinstance(wrap_padding, list) or isinstance(wrap_padding, tuple):
        wrapping = wrap_padding[1]
        padding = wrap_padding[0]
    elif isinstance(wrap_padding, int):
        wrapping = padding = wrap_padding

    return F.pad(wrap(tensor, wrapping, axis=axis[1]), [0, 0, padding, padding], mode='constant')


def wrap(tensor: torch.Tensor, wrapping, axis=2):
    """Wrap cylindrically, appending evenly to both sides.
    """
    rank = tensor.ndimension()
    if axis >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    rpad = tensor.narrow(axis, 0, wrapping)
    lpad = tensor.narrow(axis, tensor.shape[axis] - wrapping, wrapping)

    return torch.cat([lpad, tensor, rpad], dim=axis)

def unwrap(tensor: torch.Tensor, wrapping, axis=2):
    """Removes wrapping from an image.
    For odd wrapping amounts, this assumes an extra column on the [-1] side.
    """
    rank = tensor.ndimension()
    if axis >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    return tensor.narrow(axis, floor(wrapping/2), tensor.shape[axis] - wrapping)