import argparse
import os
from typing import Any, Dict, Union

import torch
import torchvision.utils as vutils

from . import base_model
from .abstract_model import AbstractModel
from .losses import loss_options, losses


def create_model(opt: argparse.Namespace) -> AbstractModel:
    return Pix2PixModel(opt)


def modify_model_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add new dataset-specific options, and rewrite default values for existing options.
    """
    parser = base_model.modify_model_commandline_options(parser)
    # loss
    parser.add_argument('--gan_loss_name', type=str, required=True, choices=losses.keys())
    parser.add_argument('--l1_loss_name', type=str, required=True, choices=losses.keys())
    opt, _ = parser.parse_known_args()
    gan_loss_modify_commandline_options = loss_options[opt.gan_loss_name]
    l1_loss_modify_commandline_options = loss_options[opt.l1_loss_name]
    parser = gan_loss_modify_commandline_options(parser)
    parser = l1_loss_modify_commandline_options(parser)

    parser.add_argument('--input_nch', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--direction', type=str, default='a2b', choices=['a2b', 'b2a'])
    return parser


class Pix2PixModel(base_model.BaseModel):
    """pix2pix model, for learning a mapping from input images to output images given paired.
    画像から画像を変換するようなモデルを扱う
    Generatorには入力画像を、Discriminatorには入力画像と出力画像のセットを入れる
    Generator : n_layer, pixel (画像サイズ128x128, 256x256のみに対応)
    Discriminator : resnet, unet (画像サイズ128x128, 256x256のみに対応)

    reference code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, opt: argparse.Namespace) -> None:
        """Initialize the pix2pix class.
        """
        super().__init__(opt)

        self.input_nch = opt.input_nch

        self.visual_names = ['real_a', 'fake_b', 'real_b']

        if self.is_train:
            # criterion gan
            self._gan_criterion = losses[opt.gan_loss_name](opt)
            self._gan_criterion.to(self.device)
            # criterion l1
            self._l1_criterion = losses[opt.l1_loss_name](opt)
            self._l1_criterion.to(self.device)
            # register
            self.criteria['gan'] = self._gan_criterion
            self.criteria['l1'] = self._l1_criterion

    def set_input(self, input_: Dict[str, Any]) -> None:
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        a2b = self.opt.direction == 'a2b'
        self.real_a = input_['a' if a2b else 'b'].to(self.device)
        self.real_b = input_['b' if a2b else 'a'].to(self.device)
        return

    def forward(self) -> None:
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        self.fake_b = self._generator_module(self.real_a)
        return

    def backward_discriminator(self) -> None:
        """Calculate GAN loss for the discriminator
        """
        # Fake; stop backward to the generator by detaching fake_B
        fake_ab = torch.cat((self.real_a, self.fake_b), 1)  # we use cGANs; need to feed both input and output to D.
        pred_fake = self._discriminator_module(fake_ab.detach())
        loss_discriminator_gan_fake = self._gan_criterion(pred_fake, False)
        # Real
        real_ab = torch.cat((self.real_a, self.real_b), 1)
        pred_real = self._discriminator_module(real_ab)
        loss_discriminator_gan_real = self._gan_criterion(pred_real, True)
        # combine loss and calculate gradients
        loss_discriminator = (loss_discriminator_gan_fake + loss_discriminator_gan_real) * 0.5
        loss_discriminator.backward()
        # register
        self.losses['discriminator_gan_fake'] = loss_discriminator_gan_fake
        self.losses['discriminator_gan_real'] = loss_discriminator_gan_real
        return

    def backward_generator(self) -> None:
        """Calculate GAN and L1 loss for the generator
        """
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((self.real_a, self.fake_b), 1)
        pred_fake = self._discriminator_module(fake_ab)
        loss_generator_gan = self._gan_criterion(pred_fake, True)
        # Second, G(A) = B
        loss_generator_l1 = self._l1_criterion(self.fake_b, self.real_b)
        # combine loss and calculate gradients
        loss_generator = loss_generator_gan + loss_generator_l1
        loss_generator.backward()
        # register
        self.losses['generator_gan'] = loss_generator_gan
        self.losses['generator_l1'] = loss_generator_l1
        return

    def save_current_image(self, epoch: Union[int, str]) -> None:
        real_a = self.real_a.repeat(1, int(3 / self.input_nch), 1, 1)
        output_image = torch.cat([real_a, self.fake_b, self.real_b], dim=3)
        vutils.save_image(
            output_image,
            os.path.join(self.save_dir, 'pix2pix_epoch_%s.png' % epoch),
            normalize=True,
        )
        return
