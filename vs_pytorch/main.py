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
from timeit import timeit
from jax import numpy as jnp
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from vs_pytorch import utils
from vs_pytorch import wide_resnet
from vs_pytorch import model_builder
from vs_pytorch import dataloader_builder
from vs_pytorch.ntk_computer import compute_ntk
from vs_pytorch.torch_utils import calculate_jacobian_wrt_params_torch, calculate_jacobian_wrt_params_modern_torch
from vs_pytorch.jax_utils import calculate_jacobian_wrt_params_jax, jax_flatten


def main(config, logger):

    model_config = config['model']
    train_config = config['train']
    eval_config = config['eval']
    data_config = config['data']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    num_devices = torch.cuda.device_count()

    dataloaders = dataloader_builder.build(data_config, logger)
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

    n = 10
    loop = 10
    args = [models['model'], X_train[:n]]
    print('Here')
    calculate_jacobian_wrt_params_modern_torch(*args)
    print('After')
    # prev_time = time.time()
    # J1 = calculate_jacobian_wrt_params_torch(*args)
    # elapsed = time.time() - prev_time
    # print(f'Elapsed: {elapsed:.3f}s')
    # import IPython; IPython.embed()
    # elapsed = timeit(lambda: calculate_jacobian_wrt_params_torch(*args), number=loop)
    # print(f'torch jacobian: {elapsed:.3f}s')

    # calculate_jacobian_wrt_params_modern_torch(*args)
    # import IPython; IPython.embed()

    J_train = jnp.asarray(X_train[:n].detach().cpu().numpy().transpose(0, 2, 3, 1), dtype=jnp.float64)
    torch.cuda.empty_cache()

    args = [models['apply_fn'], models['ntk_params'], J_train, models['rng'], None]
    j_fn = calculate_jacobian_wrt_params_jax(*args)
    jj_fn = jax.jit(j_fn)
    pj_fn = jax.pmap(j_fn)
    import IPython; IPython.embed()
    # J2 = j_fn(J_train)
    # J2 = jax_flatten(jax.tree_leaves(J2))
    # elapsed = timeit(lambda: j_fn(J_train, {}), number=loop)
    # print(f'jax jacobian: {elapsed:.3f}s')

    # ntk_kernel = compute_ntk(X_train, models, 10, num_devices, model_config, True)


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

