import numpy as np
import torch
from torch import Tensor, nn

from networks.cube_padding import sides_from_batch, sides_to_batch


def rotation_matrix(x_angle, y_angle, z_angle): # + is from y to z, z to x, and x to y, respectively
    return torch.from_numpy(np.array([
        [np.cos(y_angle) * np.cos(z_angle),
         np.sin(x_angle) * np.sin(y_angle) * np.cos(z_angle) - np.cos(x_angle) * np.sin(z_angle),
         np.cos(x_angle) * np.sin(y_angle) * np.cos(z_angle) + np.sin(x_angle) * np.sin(z_angle)],
        [np.cos(y_angle) * np.sin(z_angle),
         np.sin(x_angle) * np.sin(y_angle) * np.sin(z_angle) + np.cos(x_angle) * np.cos(z_angle),
         np.cos(x_angle) * np.sin(y_angle) * np.sin(z_angle) - np.sin(x_angle) * np.cos(z_angle)],
        [-np.sin(y_angle), np.sin(x_angle) * np.cos(y_angle), np.cos(x_angle) * np.cos(y_angle)]
    ], dtype='float32'))

#TODO check this
class CubePosesAndLoss(nn.Module):
    def __init__(self):
        super(CubePosesAndLoss, self).__init__()
        # Z: In, out, + is forward ; X: Side to side, + is left ; Y: up and down, + is up
        self.top_R = nn.Parameter(rotation_matrix(np.pi / 4, 0, 0).T, requires_grad=False)
        self.bottom_R = nn.Parameter(rotation_matrix(-np.pi / 4, 0, 0).T, requires_grad=False)
        self.left_R = nn.Parameter(rotation_matrix(0, -np.pi / 4, 0).T, requires_grad=False)
        self.right_R = nn.Parameter(rotation_matrix(0, np.pi / 4, 0).T, requires_grad=False)
        self.back_R = nn.Parameter(rotation_matrix(0, np.pi / 2, 0).T, requires_grad=False) #TODO do I need to flip X?
        self.filler = nn.Parameter(torch.from_numpy(np.array([0, 0, 0, 1], dtype='float32')), requires_grad=False)

        # transofrm from P2 to P1 according to stackoverflow: P1^T @ P2 @ P1
        # unclear answer though

    def forward(self, T: Tensor) -> (Tensor, Tensor):
        T = T[:, :3, :]
        top, bottom, left, right, front, back = sides_from_batch(T)

        # make all poses from forward's PoV

        top = torch.matmul(self.top_R, top)
        bottom = torch.matmul(self.bottom_R, bottom)
        left = torch.matmul(self.left_R, left)
        right = torch.matmul(self.right_R, right)
        back = torch.matmul(self.back_R, back)

        # calculate loss (std) and mean pose
        # TODO maybe leave out top and bottom, as they are nearly impossible to judge
        poses = torch.stack([top, bottom, left, right, front, back], dim=1)
        loss = poses.std(dim=1, keepdim=False, unbiased=False).sum(1).sum(1)
        pose = poses.mean(dim=1, keepdim=False)

        # translate back to original PoV
        top = torch.matmul(self.top_R.T, pose)
        bottom = torch.matmul(self.bottom_R.T, pose)
        left = torch.matmul(self.left_R.T, pose)
        right = torch.matmul(self.right_R.T, pose)
        back = torch.matmul(self.back_R.T, pose)

        pose = sides_to_batch(top, bottom, left, right, front, back)
        pose = torch.cat([pose, self.filler.reshape(1, 1, 4).repeat(pose.shape[0], 1, 1)], dim=1)
        return loss, pose
