import re
import jax
import yaml
import torch
import logging
import numpy as np
from colorlog import ColoredFormatter
from jax.ops import index, index_update
from jax import tree_util, vmap, pmap, lax


def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def get_full_data(data, indices):
    X, y = [], []
    for idx in indices:
        Xi, yi = data[idx]['inputs'], data[idx]['labels']
        X.append(Xi.unsqueeze(0))
        y.append(yi)
    X = torch.cat(X)
    y = torch.tensor(y)
    return (X, y)


class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Data
# ====

def to_device(dict, device):
    for key in dict.keys():
        if isinstance(dict[key], torch.Tensor):
            dict[key] = dict[key].to(device)
    return dict


# Divisors of a number
# ====================

def get_sorted_divisors(num):
    candidates = np.arange(1, num+1)
    return candidates[np.mod(num, candidates) == 0]


# Get params size of jax neural network
# =====================================

def get_params_size(params):
    return sum(x.size for x in jax.tree_leaves(params))


def get_ntk_input_shape(data_config, num_input_channels):
    crop_size = data_config['transform']['crop_size']
    input_shape = (-1, crop_size, crop_size, num_input_channels)
    return input_shape


def convert_params_to_jax(params, model):
    treedef = tree_util.tree_structure(params)
    new_jax_params = tree_util.tree_leaves(params)
    running_mean = None
    running_var = None
    bn_weight = None

    if hasattr(model, 'module'):
        model = model.module

    model.eval()
    for i, (name, param) in enumerate(model.named_parameters()):
        if i == len(new_jax_params) - 1:  # last fc layer bias
            new_param = param.view(1, -1).detach().cpu().numpy()
        elif i == len(new_jax_params) - 2:  # last fc layer weight
            new_param = param.detach().cpu().numpy().swapaxes(1, 0)
        elif re.findall("bn\d.weight", name):  # batch norm weight
            name_list = name.split('.')
            if len(name_list) <= 2:
                running_mean = getattr(getattr(model, name_list[0]), 'running_mean')
                running_var = getattr(getattr(model, name_list[0]), 'running_var')
            else:
                running_mean = getattr(getattr(
                    getattr(model, name_list[0])[int(name_list[1])], name_list[2]), 'running_mean')
                running_var = getattr(getattr(
                    getattr(model, name_list[0])[int(name_list[1])], name_list[2]), 'running_var')
            bn_weight = param

            new_param = torch.diag(param / (torch.sqrt(running_var) + 1e-5)).detach().cpu().numpy()
        elif re.findall("bn\d.bias", name):
            param = param - (running_mean / (torch.sqrt(running_var) + 1e-5)) * bn_weight
            new_param = param.view(1, 1, 1, -1).detach().cpu().numpy()
        else:
            if len(param.shape) == 1:
                param = param.view(-1, 1, 1, 1)
            new_param = param.detach().cpu().numpy().transpose(2, 3, 1, 0)
        try:
            new_jax_params[i] = index_update(new_jax_params[i], index[:], new_param)
        except:
            import IPython;
            IPython.embed()

    new_jax_params = tree_util.tree_unflatten(treedef, new_jax_params)
    return new_jax_params