import argparse
import random
from typing import Any, Tuple

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torchvision.transforms import transforms


def create_transform(opt: argparse.Namespace) -> Any:
    return create_crop_and_random_flip_transform(opt.out_size)


def modify_transform_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


def create_crop_and_random_flip_transform(out_size: int):
    def crop_and_random_flip_transform(a_img: Image, b_img: Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        reference :
        https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/6
        """
        assert a_img.width == b_img.width and a_img.height == b_img.height
        resize = transforms.Pad(padding=(0, int((a_img.width - a_img.height) / 2)), fill=255)
        a_img = resize(a_img)
        b_img = resize(b_img)

        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(
            a_img, output_size=(out_size, out_size),
        )
        a_img = tf.crop(a_img, i, j, h, w)
        b_img = tf.crop(b_img, i, j, h, w)

        # Random Horizontal Flipping
        if random.random() > 0.5:
            a_img = tf.hflip(a_img)
            b_img = tf.hflip(b_img)

        a_tensor = tf.to_tensor(a_img)
        b_tensor = tf.to_tensor(b_img)

        a_normalize = transforms.Normalize((0.5,), (0.5,))
        a_tensor = a_normalize(a_tensor)

        b_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        b_tensor = b_normalize(b_tensor)

        return a_tensor, b_tensor
    return crop_and_random_flip_transform
