from __future__ import absolute_import, division, print_function

import os
import cv2
import logging
import numpy as np
import shutil

import torch
from carla_dataset.config import load_csv
from collections import OrderedDict

from imageio import imsave
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_utils import convert_to_cubemap_batch, get_datasets, get_params
from datasets.carla_dataset_loader import CarlaDataset
from layers import disp_to_depth
from utils import readlines
from options import Mode, MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def un_mod_key(key: str):
    if key.startswith("module."):
        return key[7:]
    else:
        return key


def un_mod(weights: OrderedDict):
    return OrderedDict([(un_mod_key(k), v) for k, v in weights.items()])


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).astype('float32').mean()
    a2 = (thresh < 1.25 ** 2).astype('float32').mean()
    a3 = (thresh < 1.25 ** 3).astype('float32').mean()
    a4 = (thresh < 1.25 ** 4).astype('float32').mean()
    a5 = (thresh < 1.25 ** 5).astype('float32').mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = (np.abs(gt - pred) / gt).mean()

    sq_rel = (((gt - pred) ** 2) / gt).mean()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, a4, a5


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def gray2rgb(im, cmap='gray'):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt  # doesn't work in docker
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, cmap='plasma'):
    depth = depth / np.nanmax(depth)
    depth = gray2rgb(depth, cmap=cmap)
    return (depth) * 255


def save_depth_image(path, img):
    depth = normalize_depth_for_display(img)
    imsave(path, depth)


def evaluate(opt):
    logging.getLogger("imageio").setLevel(logging.ERROR)
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        if opt.eval_model is None:
            opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        else:
            if opt.load_weights_folder is not None:
                raise ValueError("Can't specify eval_model and load_weights_folder, they conflict")

            opt.eval_model = Path(opt.eval_model)
            models = Path(opt.eval_model) / "models"
            weights = [p for p in models.iterdir() if p.name.startswith("weights")]
            weights = [int(p.name.split("_")[1]) for p in weights]
            opt.load_weights_folder = models / f"weights_{max(weights)}"

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = un_mod(torch.load(encoder_path))

        conv_layer, data_lambda, intrinsics = get_params(opt)
        dataset = CarlaDataset(load_csv(opt.test_data), data_lambda, intrinsics,
                               [0], 1, is_train=False, is_cubemap=opt.mode is Mode.Cubemap, width=opt.width, height=opt.height)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(conv_layer, opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(conv_layer, encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(un_mod(torch.load(decoder_path)))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in tqdm(dataloader):
                if opt.mode is Mode.Cubemap:
                    data = convert_to_cubemap_batch(data, [0], 4)
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        raise ValueError("Not supported for carla")

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        raise ValueError("Not supported for carla")

    print("-> Evaluating")

    if opt.eval_stereo:
        raise ValueError("Not supported for carla")
    else:
        print("   Mono evaluation - using median scaling")

    gt_depth_dataset = CarlaDataset(load_csv(opt.test_data), data_lambda, intrinsics,
                                    [0], 1, is_train=False, is_cubemap=opt.mode is Mode.Cubemap, load_depth=True,
                                    load_color=False, width=opt.width, height=opt.height)
    gt_depth_dataloader = DataLoader(gt_depth_dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                     pin_memory=True, drop_last=False)

    errors = []
    ratios = []

    if opt.eval_model is not None:

        image_dir = opt.eval_model / "eval_images"
        if image_dir.exists():
            shutil.rmtree(image_dir)

        image_dir.mkdir()

    i = 0
    for gt_data in tqdm(gt_depth_dataloader):
        if opt.mode is Mode.Cubemap:
            gt_data = convert_to_cubemap_batch(gt_data, [0], 4, do_color=False)

        all_gt_depth = gt_data["depth_gt"].squeeze().numpy()
        gt_height, gt_width = all_gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        all_pred_depth = 1 / pred_disp

        mask = np.logical_and(all_gt_depth > MIN_DEPTH, all_gt_depth < MAX_DEPTH)

        pred_depth = all_pred_depth[mask]
        gt_depth = all_gt_depth[mask]

        all_gt_depth[all_gt_depth > MAX_DEPTH] = MAX_DEPTH
        all_gt_depth[all_gt_depth < MIN_DEPTH] = MIN_DEPTH

        all_pred_depth[all_pred_depth > MAX_DEPTH] = MAX_DEPTH
        all_pred_depth[all_pred_depth < MIN_DEPTH] = MIN_DEPTH

        pred_depth *= opt.pred_depth_scale_factor
        all_pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            all_pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        all_pred_depth[all_pred_depth < MIN_DEPTH] = MIN_DEPTH
        all_pred_depth[all_pred_depth > MAX_DEPTH] = MAX_DEPTH

        if opt.eval_model is not None and i % 500 == 0:
            save_depth_image(str(image_dir / f"{i}_gt_depth.png"), all_gt_depth)
            save_depth_image(str(image_dir / f"{i}_pred_depth.png"), all_pred_depth)

        errors.append(compute_errors(gt_depth, pred_depth))
        i += 1

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "a4", "a5"))
    print(("&{: 8.3f}  " * 9).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
