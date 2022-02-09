import torch
from torch import nn, Tensor
from torch.autograd.functional import jacobian

from functorch import jacrev, grad, vmap
from functorch import make_functional_with_buffers

from vs_pytorch.utils import to_device

from copy import deepcopy
from typing import List, Tuple


def flatten(grads):
    g = []
    for v in grads:
        g.append(v.reshape(-1))
    return torch.cat(g)


def calculate_jacobian_wrt_params_torch(model, inputs):
    ggs = []
    model.zero_grad()
    input_dict = to_device({'inputs': inputs.requires_grad_()}, torch.device('cuda'))
    out = model(input_dict)['logits']
    for i in range(out.shape[0]):  # for each datapoint
        gs = []
        for j in range(out.shape[1]):  # for each output neuron
            ps = torch.autograd.grad(out[i, j], model.parameters(), retain_graph=True)
            gs.append(flatten(ps))
        ggs.append(torch.stack(gs).detach().cpu())
        del gs
        if i % 10 == 0:
            print(i)
    torch.cuda.empty_cache()
    J = torch.stack(ggs).numpy()
    return J


def calculate_jacobian_wrt_params_modern_torch(model, inputs, logits=10):

    # input_dict = to_device({'inputs': inputs.requires_grad_()}, torch.device('cuda'))
    inputs = inputs.to(torch.device('cuda'))

    func, params, buffers = make_functional_with_buffers(model)

    import IPython; IPython.embed()

    # Compute a jacobian of the parameter for each datapoint
    result = vmap(jacrev(func), (None, None, 0))(params, buffers, inputs)


    return result
