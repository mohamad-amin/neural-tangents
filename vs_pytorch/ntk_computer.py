import time
import numpy as np
from jax import numpy as jnp
from functools import partial

from vs_pytorch import utils


def ntk_fn_dynamic_batched(ntk_fn_builder, num_devices, params, max_batch_size, x1, x2=None):
    x2a = x1 if x2 is None else x2
    n1, n2 = x1.shape[0], x2a.shape[0]
    assert n1 * n2 % (num_devices * num_devices) == 0, "both {} and {} should be divisible by #GPUs!".format(n1, n2)
    optimal_batch_size = np.gcd(n1, n2) // num_devices
    divisors = utils.get_sorted_divisors(optimal_batch_size)[::-1]
    batch_size = divisors[divisors < max_batch_size][0]
    print('n1 - {} n2 - {} batch size - {}'.format(n1, n2, batch_size))
    return ntk_fn_builder(batch_size=batch_size)(x1, x2, params)


def compute_ntk(X, models, ntk_max_batch_size, num_devices, model_config, double_precision=True):

    dtype = jnp.float64 if double_precision else jnp.float32
    X = jnp.asarray(X.detach().cpu().numpy().transpose(0, 2, 3, 1), dtype=dtype)

    ntk_fn_builder = models['ntk_fn_builder']
    ntk_fn_batched, params = models['ntk_fn'], models['ntk_params']
    num_classes, ntk_config = model_config['model_arch']['num_classes'], model_config['ntk']

    dynamic_batched_ntk_fn = partial(ntk_fn_dynamic_batched, ntk_fn_builder,
                                     num_devices, params, ntk_max_batch_size)
    prev_time = time.time()
    kernel = dynamic_batched_ntk_fn(X, None)  # n x n x kn x kn
    t = time.time() - prev_time
    print('Computing took {:4f}sec'.format(t))

    return kernel
