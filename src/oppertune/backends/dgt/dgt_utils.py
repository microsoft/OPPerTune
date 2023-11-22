from typing import Optional

import numpy as np
import torch
from torch import nn

__all__ = (
    "get_initialized_bias",
    "XLinear",
    "ScaleBinarizer",
    "Sparser",
)


def get_initialized_bias(
    in_features: int,
    out_features: int,
    initialization_mean: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> nn.Parameter:
    if initialization_mean is None:
        initialization_mean = torch.zeros(size=(out_features,), dtype=dtype)
    assert initialization_mean.shape == (out_features,)

    k = 1 / np.sqrt(in_features)
    lb = initialization_mean - k
    ub = initialization_mean + k
    init_val = torch.distributions.uniform.Uniform(lb, ub).sample()  # type: ignore

    return nn.Parameter(init_val, requires_grad=True)  # type: ignore


class XLinear(nn.Module):
    """
    Provides more options to nn.Linear.

    If 'weight' is not None, fixes the weights of the layer to this.

    If 'bias' is None, it means that bias is learnable. In this case, whether all bias units
    should have the same bias or not is given by 'same'.

    If 'bias' is not None, then the provided value is assumed to the fixed bias (that is not
    updated/learnt). The value of 'same' is ignored here.

    Notes:
        - Number of neurons is out_features
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        same: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)

        if weight is not None:
            self.linear.weight = nn.Parameter(weight, requires_grad=False)

        if bias is None:
            # Sample from a uniform distribution whose bounds are determined by the number of input features.
            self._bias = get_initialized_bias(in_features, 1 if same else out_features, dtype=dtype)
        else:
            self._bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return self.linear(x) + self._bias

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self._bias


class ScaleBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        fac = torch.norm(inp, p=1) / inp.numel()
        ctx.mark_non_differentiable(fac)
        return (inp >= 0) * fac, fac

    @staticmethod
    def backward(ctx, grad_output, grad_fac):
        inp = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[inp.abs() > 1] = 0
        return grad_input


class Sparser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        inpargmax = inp.argmax(-1)
        output = torch.zeros_like(inp)
        output[torch.arange(inp.shape[0]), inpargmax] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
