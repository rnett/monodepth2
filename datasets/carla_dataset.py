import random
from typing import List, Callable, Any, Dict

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
from carla_dataset.config import Config
from carla_dataset.data import DataSource, Side, SplitData, crop_pinhole_to_90
from carla_dataset.intrinsics import Intrinsics
from torchvision import transforms


class CarlaDataset(data.Dataset):
    """

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 configs: List[Config],
                 get_source: Callable[[Config], DataSource],
                 intrinsics: Intrinsics,
                 frame_idxs,  # 0, 1, -1 sort of thing
                 num_scales,
                 is_cubemap=False,
                 is_train=False,
                 load_depth=False):
        super(data.Dataset, self).__init__()

        # index -> (source to use, index in source)
        self.sources = []

        self.is_cubemap = is_cubemap

        for c in configs:
            s: DataSource = get_source(c)
            with s as d:
                frames = (d.color.shape[0] - 2)
            for j in range(frames):
                if self.is_cubemap:
                    self.sources.append((c.pinhole_data, j + 1))
                else:
                    self.sources.append((s, j + 1))

        self.height = intrinsics.height
        self.width = intrinsics.width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = load_depth

        self.K = intrinsics.normalized_K

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k or "color" in k[0]:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k or "color" in k[0]:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """

        """
        Psudocode:
            figure out which recording $index is in.  Return that recording's index, the one before it, and the one after (using settings?)
            ignore first/last images in recordings
        
        """

        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        source, frame = self.sources[index]
        with source as d:
            for i in self.frame_idxs:
                if self.is_cubemap:
                    d: SplitData
                    for s in list(Side):
                        inputs[(f"{s.name.lower()}_color", i, -1)] = Image.fromarray(crop_pinhole_to_90(d[s].color[frame + i]), 'RGB')
                else:
                    inputs[("color", i, -1)] = Image.fromarray(d.color[frame + i], 'RGB')

            if self.load_depth:
                if self.is_cubemap:
                    d: SplitData
                    for s in list(Side):
                        inputs[f"{s.name.lower()}_depth_gt"] = d[s].depth[frame + i]
                else:
                    inputs["depth_gt"] = d.depth[frame]

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if self.is_cubemap:
                for s in list(Side):
                    del inputs[(f"{s.name.lower()}_color", i, -1)]
                    del inputs[(f"{s.name.lower()}_color_aug", i, -1)]
            else:
                del inputs[("color", i, -1)]
                del inputs[("color_aug", i, -1)]

        return inputs

def just_side(side: Side, inputs: Dict[str, Any]):
    outs = {}
    for k in inputs:
        if k.startswith(side.name.lower()):
            outs[k[len(side.name)+1:]] = inputs[k]
