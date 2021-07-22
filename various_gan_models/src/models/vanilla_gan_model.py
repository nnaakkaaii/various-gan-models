import argparse
import os
from typing import Any, Dict, Union

import torchvision.utils as vutils

from . import base_model
from .abstract_model import AbstractModel
from .losses import loss_options, losses
from .utils.produce_noise import produce_noise


def create_model(opt: argparse.Namespace) -> AbstractModel:
    return VanillaGANModel(opt)


def modify_model_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_model.modify_model_commandline_options(parser)
    # loss
    parser.add_argument('--gan_loss_name', type=str, required=True, choices=losses.keys())
    opt, _ = parser.parse_known_args()
    gan_loss_modify_commandline_options = loss_options[opt.gan_loss_name]
    parser = gan_loss_modify_commandline_options(parser)
    return parser


class VanillaGANModel(base_model.BaseModel):
    """Vanilla GAN Model
    通常のGANの生成モデル
    DiscriminatorとGeneratorをそれぞれ選択して使用する
    Generator : vanilla_fc_module
    """
    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__(opt)

        self.visual_names = ['real_data', 'fake_data']

        if self.is_train:
            # criterion gan
            self._gan_criterion = losses[opt.gan_loss_name](opt)  # BCEWithLogitsLoss
            self._gan_criterion.to(self.device)
            # register
            self.criteria['gan'] = self._gan_criterion

    def set_input(self, input_: Dict[str, Any]) -> None:
        self.real_data = input_['data'].to(self.device)
        return

    def forward(self) -> None:
        noise = produce_noise(self.opt, self.device)
        self.fake_data = self._generator_module(noise)
        return

    def backward_discriminator(self) -> None:
        # Fake
        pred_fake = self._discriminator_module(self.fake_data.detach())
        loss_discriminator_gan_fake = self._gan_criterion(pred_fake, False)
        # Real
        pred_real = self._discriminator_module(self.real_data)
        loss_discriminator_gan_real = self._gan_criterion(pred_real, True)
        # combine loss and calculate gradients
        loss_discriminator = (loss_discriminator_gan_fake + loss_discriminator_gan_real) * 0.5
        loss_discriminator.backward()
        # register
        self.losses['discriminator_gan_fake'] = loss_discriminator_gan_fake
        self.losses['discriminator_gan_real'] = loss_discriminator_gan_real
        return

    def backward_generator(self) -> None:
        # Fake
        pred_fake = self._discriminator_module(self.fake_data)
        loss_generator_gan = self._gan_criterion(pred_fake, True)
        # calculate gradients
        loss_generator_gan.backward()
        # register
        self.losses['generator_gan'] = loss_generator_gan
        return

    def save_current_image(self, epoch: Union[int, str]) -> None:
        output_image = self.fake_data.repeat(1, int(3 / self.output_nch), 1, 1)
        vutils.save_image(
            output_image,
            os.path.join(self.save_dir, 'vanilla_gan_epoch_%s.png' % epoch),
            normalize=True,
        )
        return
