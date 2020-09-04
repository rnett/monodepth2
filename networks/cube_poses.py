import numpy as np
import torch
from torch import Tensor, nn

from networks.cube_padding import sides_from_batch, sides_to_batch


def rotation_x(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), np.sin(angle), 0],
        [0, -np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_y(angle):
    return np.array([
        [np.cos(angle), 0, -np.sin(angle), 0],
        [0, 1, 0, 0],
        [np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_z(angle):
    return np.array([
        [np.cos(angle), np.sin(angle), 0, 0],
        [-np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_matrix(x_angle, y_angle, z_angle):  # + is from y to z, z to x, and x to y, respectively

    rot = np.eye(4, dtype=np.float32)

    if z_angle != 0:
        rot = rot @ rotation_z(z_angle)

    if y_angle != 0:
        rot = rot @ rotation_y(y_angle)

    if x_angle != 0:
        rot = rot @ rotation_x(x_angle)

    return torch.from_numpy(rot)


def change_basis(T, change):
    return change @ T @ change.T

# TODO check this
# TODO Use axes swap/neg instead?  Don't think it would work w/ rotation
# TODO 99% sure this is wrong.  EX: Left doesn't rorate around the Y axis
class CubePosesAndLoss(nn.Module):
    def __init__(self, include_loss=True):
        super(CubePosesAndLoss, self).__init__()
        # These are rotations FROM the side, TO front  i.e. side_to_front

        rot = np.pi / 2

        self.top_R = nn.Parameter(rotation_matrix(0, -rot, 0), requires_grad=False)
        self.bottom_R = nn.Parameter(rotation_matrix(0, rot, 0), requires_grad=False)
        self.left_R = nn.Parameter(rotation_matrix(0, 0, -rot), requires_grad=False)
        self.right_R = nn.Parameter(rotation_matrix(0, 0, rot), requires_grad=False)
        self.back_R = nn.Parameter(rotation_matrix(0, 0, np.pi), requires_grad=False)

        self.filler = nn.Parameter(torch.from_numpy(np.array([0, 0, 0, 1], dtype='float32')), requires_grad=False)

        self.include_loss = include_loss

    def forward(self, T: Tensor) -> (Tensor, Tensor):
        top, bottom, left, right, front, back = sides_from_batch(T)

        # make all poses from forward's PoV

        # top = change_basis(top, self.top_R)
        # bottom = change_basis(bottom, self.bottom_R)
        left = change_basis(left, self.left_R)
        right = change_basis(right, self.right_R)
        back = change_basis(back, self.back_R)

        # calculate loss (std) and mean pose
        # TODO maybe leave out top and bottom, as they are nearly impossible to judge
        # top, bottom,
        poses = torch.stack([left, right, front, back], dim=1)
        pose = poses.mean(dim=1, keepdim=False)

        # translate back to original PoV
        top = change_basis(pose, self.top_R.T)
        bottom = change_basis(pose, self.bottom_R.T)
        left = change_basis(pose, self.left_R.T)
        right = change_basis(pose, self.right_R.T)
        back = change_basis(pose, self.back_R.T)
        front = pose

        pose = sides_to_batch(top, bottom, left, right, front, back)
        # pose = torch.cat([pose, self.filler.reshape(1, 1, 4).repeat(pose.shape[0], 1, 1)], dim=1)

        if self.include_loss:
            loss = poses.std(dim=1, keepdim=False, unbiased=False).sum(1).sum(1)
            return loss, pose
        else:
            return pose
