import os
import jax
import torch
import time
import random
import shutil
import logging
import argparse
import tempfile
import numpy as np
from jax import numpy as jnp
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

import utils
import wide_resnet
import model_builder
import dataloader_builder
from ntk_computer import compute_ntk


def main(config, logger):

    model_config = config['model']
    train_config = config['train']
    eval_config = config['eval']
    data_config = config['data']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    num_devices = torch.cuda.device_count()

    dataloaders = dataloader_builder.build(config, logger)
    models = model_builder.build(model_config, data_config, logger)

    if not isinstance(models['model'], torch.nn.DataParallel):
        if torch.cuda.device_count() > 1:
            models['model'] = utils.DataParallel(models['model'])
        models['model'] = models['model'].to(device)

    al_params = data_config['al_params']
    subset = dataloaders['subset']

    models['ntk_params'] = utils.convert_params_to_jax(models['ntk_params'], models['model'])

    labeled_dataset, unlabeled_dataset = dataloaders['train'].dataset, dataloaders['unlabeled'].dataset
    labeled_set, subset = dataloaders['labeled_set'], dataloaders['subset']

    X_train, y_train = utils.get_full_data(labeled_dataset, labeled_set)

    import IPython; IPython.embed()
    ntk_kernel = compute_ntk(X_train, models, 10, num_devices, model_config, True)


if __name__ == '__main__':

    # Fixing the pthread_cancel glitch while using python 3.8 (you can comment these two lines if you're on 3.7)
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',  help="Path to a config")
    args = parser.parse_args()

    logger = utils.load_log(tempfile.NamedTemporaryFile().name)
    config = utils.load_config(args.config_path)

    main(config, logger)

