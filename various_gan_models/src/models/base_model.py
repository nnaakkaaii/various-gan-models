import abc
import argparse
import os
from typing import Any, Dict, List, Union

import torch

from .abstract_model import AbstractModel
from .modules.discriminator_modules import \
    module_options as discriminator_module_options
from .modules.discriminator_modules import modules as discriminator_modules
from .modules.generator_modules import \
    module_options as generator_module_options
from .modules.generator_modules import modules as generator_modules
from .optimizers import optimizer_options, optimizers
from .schedulers import scheduler_options, schedulers
from .utils.init_weights import init_weight_options, init_weights
from .utils.init_weights.apply_init_weight import apply_init_weight


def modify_model_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add new model-specific options, and rewrite default values for existing options.
    """
    # module
    parser.add_argument('--discriminator_module_name', type=str, required=True, choices=discriminator_modules.keys())
    parser.add_argument('--generator_module_name', type=str, required=True, choices=generator_modules.keys())
    opt, _ = parser.parse_known_args()
    discriminator_module_modify_commandline_options = discriminator_module_options[opt.discriminator_module_name]
    generator_module_modify_commandline_options = generator_module_options[opt.generator_module_name]
    parser = discriminator_module_modify_commandline_options(parser)
    parser = generator_module_modify_commandline_options(parser)

    # optimizer
    parser.add_argument('--discriminator_optimizer_name', type=str, required=True, choices=optimizers.keys())
    parser.add_argument('--generator_optimizer_name', type=str, required=True, choices=optimizers.keys())
    opt, _ = parser.parse_known_args()
    discriminator_optimizer_modify_commandline_options = optimizer_options[opt.discriminator_optimizer_name]
    generator_optimizer_modify_commandline_options = optimizer_options[opt.generator_optimizer_name]
    parser = discriminator_optimizer_modify_commandline_options(parser)
    parser = generator_optimizer_modify_commandline_options(parser)

    # scheduler
    parser.add_argument('--discriminator_scheduler_name', type=str, required=True, choices=schedulers.keys())
    parser.add_argument('--generator_scheduler_name', type=str, required=True, choices=schedulers.keys())
    opt, _ = parser.parse_known_args()
    discriminator_scheduler_modify_commandline_options = scheduler_options[opt.discriminator_scheduler_name]
    generator_scheduler_modify_commandline_options = scheduler_options[opt.generator_scheduler_name]
    parser = discriminator_scheduler_modify_commandline_options(parser)
    parser = generator_scheduler_modify_commandline_options(parser)

    # init weight
    parser.add_argument('--init_weight_name', type=str, required=True, choices=init_weights.keys())
    opt, _ = parser.parse_known_args()
    init_weight_modify_commandline_options = init_weight_options[opt.init_weight_name]
    parser = init_weight_modify_commandline_options(parser)

    return parser


class BaseModel(AbstractModel, metaclass=abc.ABCMeta):
    """This class is an abstract class for models.
    """
    def __init__(self, opt: argparse.Namespace) -> None:
        """Initialize the BaseModel class.
        """
        super().__init__(opt)

        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.output_nch = opt.output_nch
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        # generator module
        self._generator_module = generator_modules[opt.generator_module_name](opt)
        apply_init_weight(self._generator_module, opt, init_weight=init_weights[opt.init_weight_name])
        if self.is_train:
            # discriminator module
            self._discriminator_module = discriminator_modules[opt.discriminator_module_name](opt)
            apply_init_weight(self._discriminator_module, opt, init_weight=init_weights[opt.init_weight_name])
            # generator optimizer
            self._generator_optimizer = optimizers[opt.generator_optimizer_name](self._generator_module.parameters(), opt)
            # discriminator optimizer
            self._discriminator_optimizer = optimizers[opt.discriminator_optimizer_name](self._discriminator_module.parameters(), opt)
            # generator scheduler
            self._generator_scheduler = schedulers[opt.generator_scheduler_name](self._generator_optimizer, opt)
            # discriminator scheduler
            self._discriminator_scheduler = schedulers[opt.discriminator_scheduler_name](self._discriminator_optimizer, opt)

        # register
        if not self.is_train:
            self.modules['generator'] = self._generator_module
        else:
            self.modules['generator'] = self._generator_module
            self.modules['discriminator'] = self._discriminator_module
            self.optimizers['generator'] = self._generator_optimizer
            self.optimizers['discriminator'] = self._discriminator_optimizer
            self.schedulers['generator'] = self._generator_scheduler
            self.schedulers['discriminator'] = self._discriminator_scheduler

        self.module_transfer_to_device()

    def setup(self, opt: argparse.Namespace) -> None:
        """Called in construct
        Setup (Load and print networks)
            -- load networks    : if not training mode or continue_train is True, then load opt.epoch
            -- print networks
        """
        if not self.is_train or opt.continue_train:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)
        return

    @abc.abstractmethod
    def backward_discriminator(self) -> None:
        pass

    @abc.abstractmethod
    def backward_generator(self) -> None:
        pass

    @staticmethod
    def set_requires_grad(nets: Union[torch.nn.Module, List[torch.nn.Module]], requires_grad: bool = False) -> None:
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        return

    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights; called in every training iteration.
        """
        self.forward()                                                  # compute fake images: G(A)
        # update discriminator
        self.set_requires_grad([self._discriminator_module], True)      # enable backward for D
        self._discriminator_optimizer.zero_grad()                       # set D's gradients to zero
        self.backward_discriminator()                                   # calculate gradients for D
        self._discriminator_optimizer.step()                            # update D's weights
        # update generator
        self.set_requires_grad([self._discriminator_module], False)     # D requires no gradients when optimizing G
        self._generator_optimizer.zero_grad()                           # set G's gradients to zero
        self.backward_generator()                                       # calculate gradients for G
        self._generator_optimizer.step()                                # update G's weights
        return

    def module_transfer_to_device(self) -> None:
        """transfer all modules to device
        """
        for name, module in self.modules.items():
            module.to(self.device)
            if self.device.type == 'cuda':
                self.modules[name] = torch.nn.DataParallel(module, self.gpu_ids)
        return

    def eval(self) -> None:
        """make models eval mode during test time.
        """
        for module in self.modules.values():
            module.eval()
        return

    def train(self) -> None:
        """turn to train mode
        """
        for module in self.modules.values():
            module.train()
        return

    def test(self) -> None:
        """ Forward function used in test time.
        """
        with torch.no_grad():
            self.forward()
        return

    def update_learning_rate(self) -> None:
        """Update learning rates for all the networks; called at the end of every epoch
        """
        optimizer = list(self.optimizers.values())[0]
        old_lr = optimizer.param_groups[0]['lr']
        for name, scheduler in self.schedulers.items():
            if name == 'generator' and self.opt.generator_scheduler_name == 'plateau':
                scheduler.step(self.metric)
            elif name == 'discriminator' and self.opt.discriminator_scheduler_name == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
        return

    def get_current_losses(self) -> Dict[str, float]:
        """Return training losses / errors. train_option.py will print out these errors on console, and save them to a file
        """
        errors_ret = dict()
        for name, loss in self.losses.items():
            if isinstance(name, str):
                errors_ret[name] = float(loss)
        return errors_ret

    def save_networks(self, epoch: Union[int, str]) -> None:
        """Save all the networks to the disk.
        """
        for name, module in self.modules.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(module.module.cpu().state_dict(), save_path)
                module.cuda(self.gpu_ids[0])
            else:
                torch.save(module.cpu().state_dict(), save_path)
        return

    def __patch_instance_norm_state_dict(
            self, state_dict: Any, module: torch.nn.Module, keys: List[str], i: int = 0) -> None:
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)
        """
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
        return

    def load_networks(self, epoch: int) -> None:
        """Load all the networks from the disk.
        """
        for name, module in self.modules.items():
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                if isinstance(module, torch.nn.DataParallel):
                    module = module.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, module, key.split('.'))
                module.load_state_dict(state_dict)
        return

    def print_networks(self, verbose: bool) -> None:
        """Print the total number of parameters in the network and (if verbose) network architecture
        """
        print('---------- Networks initialized -------------')
        for name, module in self.modules.items():
            num_params = 0
            for param in module.parameters():
                num_params += param.numel()
            if verbose:
                print(module)
            print('[Network %s] Total number of parameters: %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        return
