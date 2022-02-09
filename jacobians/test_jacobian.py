import sys
import torch
import torch.nn as nn

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as fnn

from timeit import timeit
from typing import Sequence


class Net(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden_ndim, n_layers):
        super().__init__()
        assert n_layers > 2
        layers = [nn.Linear(n_inputs, hidden_ndim, bias=False), nn.ReLU()]
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_ndim, hidden_ndim, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_ndim, n_outputs, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FlaxNet(fnn.Module):
    n_outputs: int
    hidden_ndim: int
    n_layers: int

    @fnn.compact
    def __call__(self, x):
        assert self.n_layers > 2
        x = fnn.Dense(features=self.hidden_ndim, use_bias=False)(x)
        x = fnn.relu(x)
        for i in range(self.n_layers - 2):
            x = fnn.Dense(features=self.hidden_ndim, use_bias=False)(x)
            x = fnn.relu(x)
        x = fnn.Dense(features=self.n_outputs, use_bias=False)(x)
        return x


def register_hooks(model: nn.Module):

    targets = (nn.Linear, nn.ReLU)

    def forward_postprocess(module, input, output):
        data_input = input[0]

        setattr(module, 'data_input', data_input)
        setattr(module, 'data_output', output)

    for module in model.modules():
        if isinstance(module, targets):
            module.register_forward_hook(forward_postprocess)


def jax_jacobian(rng, input_shape, model, Jx_fn):
    x = jax.random.normal(rng, input_shape)
    y = model(x)
    J = Jx_fn(x)

    import IPython; IPython.embed()

    return J


def my_jax_jacobian(J_fn, variables):
    J = J_fn(variables)['params']
    layer_params = []
    for layer in ['Dense_0', 'Dense_1', 'Dense_2', 'Dense_3']:
        p = jax.device_get(J[layer]['kernel'])
        p = p.reshape(p.shape[0], p.shape[1], -1)
        layer_params.append(p)
    del J
    return np.concatenate(layer_params, axis=2)


def manual_jacobian(bs, n_inputs, model, device, mode='rev'):
    x = torch.randn(bs, n_inputs, device=device)
    y = model(x)
    modules = []
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        modules.append(module)

    if mode == 'rev':
        J = manual_jacobian_rev(modules)
    else:
        J = manual_jacobian_fwd(modules)

    import IPython; IPython.embed()

    return J


def manual_jacobian_rev(modules):
    J = None

    for module in modules[::-1]:
        if J is None:
            J = module.weight
        elif isinstance(module, nn.ReLU):
            a = module.data_output
            d = (a > 0).type(a.dtype)
            if J.ndimension() == 2:
                J = torch.einsum('ab,nb->nab', J, d)
            else:
                J = torch.einsum('nab,nb->nab', J, d)
        elif isinstance(module, nn.Linear):
            J = torch.einsum('nab,bc->nac', J, module.weight)

    return J


def manual_jacobian_fwd(modules):
    J = None

    for module in modules:
        if J is None:
            J = module.weight
        elif isinstance(module, nn.ReLU):
            a = module.data_output
            d = (a > 0).type(a.dtype)
            if J.ndimension() == 2:
                J = torch.einsum('na,ab->nab', d, J)
            else:
                J = torch.einsum('na,nab->nab', d, J)
        elif isinstance(module, nn.Linear):
            J = torch.einsum('ab,nbc->nac', module.weight, J)

    return J


def reverse_mode_jacobian_with_repeat(bs, n_inputs, n_outputs, model, device):
    x = torch.randn(bs, n_inputs, device=device)

    repeat_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repeat_arg)
    xr = xr.transpose(0, 1)
    xr.requires_grad_(True)
    y = model(xr)
    I = torch.eye(n_outputs, device=xr.device)
    I = I.repeat(bs, 1, 1)

    Jx = torch.autograd.grad(y, xr, grad_outputs=I, retain_graph=True)[0]

    import IPython; IPython.embed()

    return Jx


def flatten(grads):
    g = []
    for v in grads:
        g.append(v.reshape(-1))
    return torch.cat(g)


def my_torch_jacobian(model, x):
    ggs = []
    model.zero_grad()
    out = model(x)
    for i in range(out.shape[0]):  # for each datapoint
        gs = []
        for j in range(out.shape[1]):  # for each output neuron
            ps = torch.autograd.grad(out[i, j], model.parameters(), retain_graph=True)
            gs.append(flatten(ps))
        ggs.append(torch.stack(gs).to(torch.device('cpu')))
    del gs
    torch.cuda.empty_cache()
    J = torch.stack(ggs).detach().cpu().numpy()
    return J


def real_torch_jacobian(model, x):
    y = model(x)
    import IPython; IPython.embed()
    grads = torch.autograd.grad(y, model.parameters(), retain_graph=True)


def main(mode):
    bs = 40
    n_inputs = 1000
    hidden_ndim = 1000
    n_outputs = 10
    n_layers = 4
    loop = 100
    print('-------------')
    print(f'mode: {mode}')
    print(f'bs: {bs}')
    print(f'n_inputs: {n_inputs}')
    print(f'hidden_ndim: {hidden_ndim}')
    print(f'n_outputs: {n_outputs}')
    print(f'n_layers: {n_layers}')
    print('-------------')
    print(f'loop: {loop}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rng = jax.random.PRNGKey(0)
    input_shape = (bs, n_inputs)
    model = FlaxNet(n_outputs=n_outputs, hidden_ndim=hidden_ndim, n_layers=n_layers)
    batch = jnp.ones(input_shape)
    variables = model.init(jax.random.PRNGKey(0), batch)

    x = jax.random.normal(rng, input_shape)
    f_params = lambda p: model.apply(variables, x)
    J_fn = jax.jit(jax.jacobian(f_params))

    args = [J_fn, variables]
    j_jax = my_jax_jacobian(*args)

    x = torch.Tensor(np.array(jax.device_get(x))).to(device)

    model = Net(n_inputs, n_outputs, hidden_ndim, n_layers)
    model = model.to(device)
    # args = [model, x]
    # j_torch = my_torch_jacobian(*args)

    # my_torch_jacobian(model, x)
    # real_torch_jacobian(model, x)

    # # real_torch_jacobian(model, x)
    #
    # # print((j_torch - j_jax).mean())
    #
    import IPython; IPython.embed()

    if mode == 'torch.auto':
        # ------------------
        # PyTorch auto-diff
        print(f'device: {device}')

        model = Net(n_inputs, n_outputs, hidden_ndim, n_layers)
        model = model.to(device)
        args = [bs, n_inputs, n_outputs, model, device]
        reverse_mode_jacobian_with_repeat(*args)
        # elapsed = timeit(lambda: reverse_mode_jacobian_with_repeat(*args), number=loop)
        # print(f'torch auto rev: {elapsed:.3f}s')
    elif mode == 'torch.man':
        # ------------------
        # PyTorch manual-diff
        print(f'device: {device}')

        model = Net(n_inputs, n_outputs, hidden_ndim, n_layers)
        model = model.to(device)
        register_hooks(model)
        args = [bs, n_inputs, model, device]
        manual_jacobian(*args)
        elapsed = timeit(lambda: manual_jacobian(*args), number=loop)
        print(f'torch manual rev: {elapsed:.3f}s')
    elif mode == 'torch.greg':
        # ------------------
        # PyTorch auto-diff Greg
        print(f'device: {device}')

        model = Net(n_inputs, n_outputs, hidden_ndim, n_layers)
        model = model.to(device)
        x = torch.randn(bs, n_inputs, device=device)
        args = [model, x]
        my_torch_jacobian(*args)
        elapsed = timeit(lambda: my_torch_jacobian(*args), number=loop)
        print(f'torch auto rev: {elapsed:.3f}s')
    else:
        # ------------------
        # JAX
        print(f'device: {jax.devices()}')

        rng = jax.random.PRNGKey(0)
        input_shape = (bs, n_inputs)
        model = FlaxNet(n_outputs=n_outputs, hidden_ndim=hidden_ndim, n_layers=n_layers)
        batch = jnp.ones(input_shape)
        variables = model.init(jax.random.PRNGKey(0), batch)

        x = jax.random.normal(rng, input_shape)
        f_params = lambda p: model.apply(variables, x)
        J_fn = jax.jit(jax.jacobian(f_params))

        args = [J_fn, variables]
        j_jax = my_jax_jacobian(*args)

        elapsed = timeit(lambda: my_jax_jacobian(*args), number=loop)
        print(f'jit(jax.jacrev): {elapsed:.3f}s')

#

if __name__ == '__main__':
    modes = ['torch.auto', 'torch.man', 'jax', 'torch.greg']
    message = f'You need to specify the computational model from {modes}.'
    assert len(sys.argv) > 1, message
    mode = sys.argv[1]
    assert mode in modes, message
    main(mode)