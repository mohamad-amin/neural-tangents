from functools import partial

import jax
import numpy as np
from jax import jacobian, vmap, pmap
from neural_tangents.utils.empirical import eval_shape, _canonicalize_axes, _get_f_params


def jax_flatten(grads):
    g = []
    for v in grads:
        g.append(jax.device_get(v).reshape(v.shape[0], v.shape[1], -1))
    return np.concatenate(g, axis=2)


def calculate_jacobian_wrt_params_jax(f, params, x, rng, vmap_axes=None):

    f = partial(f, **{'rng': rng})

    fx = eval_shape(f, params, x, **{})
    x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x, fx, **{})

    def j_fn(x, *args):
      fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes, **{})
      jx = jacobian(fx)(params)
      return jx

    if x_axis is not None or kw_axes:
      in_axes = [x_axis]
      j_fn = vmap(j_fn, in_axes=in_axes, out_axes=fx_axis)

    return j_fn
