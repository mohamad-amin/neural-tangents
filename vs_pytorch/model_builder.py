import jax
import jax.numpy as jnp
import numpy as np
import neural_tangents as nt
from functools import partial
import wide_resnet
from jax import jit
import utils
from scipy.special import log_softmax

MODELS = {
    'wide_resnet': wide_resnet.Wide_ResNet,
    #'shallow_net': models.ShallowNet,
}

NTK_MODELS = {
    'wide_resnet': wide_resnet.Wide_Resnet_NTK,
    #'shallow_net': models.ShallowNet_NTK,
}

def build(model_config, data_config, logger):
    backbone = model_config['backbone']
    model_arch = model_config['model_arch']
    acquisition = model_config['acquisition']
    ntk_config = model_config.get('ntk', {})
    data_name = data_config['name']

    if data_name in ['cifar10', 'svhn']:
        num_input_channels = 3
    else:
        num_input_channels = 1
    model_arch['num_input_channels'] = num_input_channels

    # Build a model
    models = {}
    if backbone in MODELS:
        model = MODELS[backbone](**model_arch)
        models['model'] = model
        models['use_ntk'] = False
        logger.info(
            'A model {} is built.'.format(backbone))

        if 'eer' in acquisition:
            retrain_model = MODELS[backbone](**model_arch)
            models['retrain_model'] = retrain_model

        if 'ntk' in acquisition:
            init_fn, apply_fn, _ = NTK_MODELS[backbone](**model_arch)

            # Define a loss function
            loss_fn_name = ntk_config.get('loss_fn', 'cross_entropy')
            loss_fn = None
            if loss_fn_name == 'cross_entropy':
                loss_fn = lambda fx, y_hat: -jnp.mean(jax.experimental.stax.logsoftmax(fx) * y_hat)
                #loss_fn = lambda fx, y_hat: -np.mean(log_softmax(fx) * y_hat)

            ntk_implementation = ntk_config.get('kernel_implementation', 1)

            # Initialize ntk params
            rng = jax.random.PRNGKey(ntk_config['seed'])
            ntk_fn = nt.empirical_ntk_fn(
                partial(apply_fn, **{'rng': rng}), vmap_axes=0, implementation=ntk_implementation, trace_axes=())
            ntk_fn_batched = nt.batch(
                ntk_fn, device_count=-1, batch_size=ntk_config['ntk_batch'], store_on_device=False)
            _, ntk_params = init_fn(rng, utils.get_ntk_input_shape(data_config, num_input_channels))
            ntk_fn_batch_builder = partial(nt.batch, kernel_fn=ntk_fn, device_count=-1, store_on_device=False)
            single_device_ntk_fn = partial(nt.batch, kernel_fn=ntk_fn, device_count=1, store_on_device=False)

            models.update({
                'ntk_fn': ntk_fn_batched,
                'ntk_fn_builder': ntk_fn_batch_builder,
                'ntk_params': ntk_params,
                'ntk_params_size': utils.get_params_size(ntk_params),
                'single_device_ntk_fn': single_device_ntk_fn,
                'apply_fn': apply_fn,
                'loss_fn': loss_fn,
                'rng': rng,
                'use_ntk': True
            })
            logger.info(
                'A NTK model {} is built.'.format(backbone))
    else:
        logger.error(
            'Specify valid backbone or model type among {}.'.format(MODELS.keys())
        ); exit()

    return models

