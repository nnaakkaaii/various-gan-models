from typing import Optional

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str, target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        """Initialize the GANLoss class
        :param gan_mode:
        :param target_real_label:
        :param target_fake_label:
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.loss: Optional[nn.Module] = None
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create label tensors with the same size as the input.
        :param prediction:
        :param target_is_real:
        :return:
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate loss given Discriminator's output and ground truth labels.
        :param prediction:
        :param target_is_real:
        :return:
        """
        if self.gan_mode in ['lsgan', 'vanilla'] and self.loss is not None:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                return - prediction.mean()
            else:
                return prediction.mean()
        raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)
