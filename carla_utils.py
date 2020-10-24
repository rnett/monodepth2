import torch
from carla_dataset.config import Config, load_csv
from carla_dataset.data import Side
from carla_dataset.intrinsics import CylindricalIntrinsics, Pinhole90Intrinsics, PinholeIntrinsics
from torch import nn

from datasets.carla_dataset_loader import CarlaDataset
from networks.cube_padding import CubicConv2d
from networks.cylindrical_padding import CylindricalConv2d
from options import Mode


def get_datasets(options, data_lambda, intrinsics):
    train_dataset = CarlaDataset(load_csv(options.train_data), data_lambda, intrinsics,
                                 options.frame_ids, 4, is_train=True, is_cubemap=options.mode is Mode.Cubemap, width=options.width, height=options.height)

    val_dataset = CarlaDataset(load_csv(options.val_data), data_lambda, intrinsics,
                               options.frame_ids, 4, is_train=True, is_cubemap=options.mode is Mode.Cubemap, width=options.width, height=options.height)

    return train_dataset, val_dataset


def get_pinhole_front(r: Config):
    return r.pinhole_data.front


def get_cylindrical(r: Config):
    return r.cylindrical_data


def get_params(options):
    if options.mode is Mode.Cylindrical:
        return CylindricalConv2d, get_cylindrical, CylindricalIntrinsics()
    elif options.mode is Mode.Cubemap:
        return CubicConv2d, get_pinhole_front, Pinhole90Intrinsics()
    else:
        return nn.Conv2d, get_pinhole_front, PinholeIntrinsics()



def convert_to_cubemap_batch(inputs, frame_ids, scales, do_color=True):
    '''
    Color and depth (and color_aug) have inputs for each side, need to gather them
    :param inputs:
    :return: inputs
    '''
    if do_color:
        for frame_id in frame_ids:
            for scale in scales:
                color = []
                color_aug = []
                for s in list(Side):
                    color.append(inputs[(f"{s.name.lower()}_color", frame_id, scale)])
                    del inputs[(f"{s.name.lower()}_color", frame_id, scale)]
                    color_aug.append(inputs[(f"{s.name.lower()}_color_aug", frame_id, scale)])
                    del inputs[(f"{s.name.lower()}_color_aug", frame_id, scale)]

                color = torch.stack(color, dim=1)
                color_aug = torch.stack(color_aug, dim=1)

                color = color.reshape(-1, *list(color.shape)[2:])
                color_aug = color_aug.reshape(-1, *list(color_aug.shape)[2:])
                inputs[("color", frame_id, scale)] = color
                inputs[("color_aug", frame_id, scale)] = color_aug

    #TODO test depth
    if "front_depth_gt" in inputs:
        depth_gt = []
        for s in list(Side):
            depth_gt.append(inputs[f"{s.name.lower()}_depth_gt"])
            del inputs[f"{s.name.lower()}_depth_gt"]

        depth_gt = torch.stack(depth_gt, dim=1)

        depth_gt = depth_gt.reshape(-1, *list(depth_gt.shape)[2:])
        inputs["depth_gt"] = depth_gt


    for scale in scales:
        # K and inv_K are constant across batch, no need to reorder
        inputs[("K", scale)] = inputs[("K", scale)].repeat(6, 1, 1)
        inputs[("inv_K", scale)] = inputs[("inv_K", scale)].repeat(6, 1, 1)


    #TODO intrinsics

    return inputs