import torch
from torch import Tensor, nn
import torch.nn.functional as F

from networks.cube_padding import sides_from_batch, sides_to_batch


def get_cube_poses_and_loss(x: Tensor) -> (Tensor, Tensor):
    top, bottom, left, right, front, back = sides_from_batch(x)
    # TODO need to rotate to front.  poses are the same, but coordinate systems are relative to each face
    poses = torch.stack([top, bottom, left, right, front, back], dim=1)
    loss = poses.std(dim=1, keepdim=False, unbiased=False)
    pose = poses.mean(dim=6, keepdim=False)
    # TODO rotate poses back, store in top, bottom, etc
    pose = sides_to_batch(top, bottom, left, right, front, back)
    return loss, pose
