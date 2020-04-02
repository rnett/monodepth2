import torch
import torch.nn.functional as F

from math import floor, ceil

from torch import nn


class CylindricalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='cylindrical'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "cylindrical":
            wrap_padding = [k-1 for k in self.kernel_size]
            wrapped_inputs = wrap_pad(input, wrap_padding)
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

    # set padding dimensions
    paddings = [[0, 0]] * rank
    paddings[axis[0]] = [floor(padding/2), ceil(padding/2)]

    return F.pad(wrap(tensor, wrapping, axis=axis[1]), paddings, mode='CONSTANT')


def wrap(tensor: torch.Tensor, wrapping, axis=2):
    """Wrap cylindrically, appending evenly to both sides.
    For odd wrapping amounts, the extra column is appended to the [-1] side.
    """
    rank = tensor.ndimension()
    if axis >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    sizes = [-1] * rank

    sizes[axis] = ceil(wrapping/2)
    rstarts = [0]*rank

    rpad = tensor
    for i, start, size in zip(range(len(rstarts)), rstarts, sizes):
        rpad = rpad.narrow(i, start, size)

    sizes[axis] = floor(wrapping/2)
    lstarts = [0]*rank
    lstarts[axis] = tensor.shape[axis] - floor(wrapping/2)

    lpad = tensor
    for i, start, size in zip(range(len(lstarts)), lstarts, sizes):
        lpad = rpad.narrow(i, start, size)

    return torch.cat([lpad, tensor, rpad], dim=axis)

def unwrap(tensor, wrapping, axis=2):
    """Removes wrapping from an image.
    For odd wrapping amounts, this assumes an extra column on the [-1] side.
    """
    rank = tensor.shape.ndims
    if axis >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    sizes = [-1] * rank
    sizes[axis] = tensor.shape.as_list()[axis] - wrapping

    starts = [0] * rank
    starts[axis] = floor(wrapping/2)


    for i, start, size in zip(range(len(starts)), starts, sizes):
        tensor = tensor.narrow(i, start, size)

    return tensor
    #return tensor[:,:,1:-1,:]