import argparse

from src.datasets import datasets
from src.transforms import transforms
from src.dataloaders import dataloaders
from src.models import models
from src.loggers import loggers
from src.options.train_option import TrainOption


def train(opt: argparse.Namespace) -> None:
    transform = transforms[opt.transform_name](opt)
    dataset = datasets[opt.dataset_name](transform, opt)
    dataloader = dataloaders[opt.dataloader_name](dataset, opt)
    dataset_size = len(dataset)
    dataloader_size = len(dataloader)
    print('The number of training images = %s' % dataset_size)

    model = models[opt.model_name](opt)
    model.setup(opt)

    logger = loggers[opt.logger_name](model, opt)
    logger.set_dataset_length(dataloader_size)
    logger.save_options()

    for epoch in range(opt.epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        logger.start_epoch()
        for data in dataloader:
            model.set_input(data)
            model.optimize_parameters()
            logger.end_iter()
        logger.end_epoch()
        model.update_learning_rate()
    logger.end_all_training()
    return


if __name__ == '__main__':
    opt = TrainOption().parse()
    train(opt)
