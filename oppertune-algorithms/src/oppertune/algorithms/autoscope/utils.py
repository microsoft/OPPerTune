from math import sqrt
from typing import Any, Optional, Tuple

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

    if initialization_mean.shape != (out_features,):
        raise ValueError(
            f"Shape of initialization_mean={initialization_mean.shape} should be (out_features,)={(out_features,)}"
        )

    k = 1 / sqrt(in_features)
    low = initialization_mean - k
    high = initialization_mean + k
    init_val = torch.distributions.uniform.Uniform(low, high).sample()
    return nn.Parameter(init_val, requires_grad=True)


class XLinear(nn.Module):
    """Provides more options to nn.Linear.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self._bias

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self._bias


class ScaleBinarizer(torch.autograd.Function):
    """Apply deterministic binarization in the forward pass as stated in https://arxiv.org/pdf/1602.02830.

    Use a smooth activation in the backward pass.
    """

    @staticmethod
    def forward(ctx: Any, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(inp)
        fac = torch.linalg.vector_norm(inp, ord=1) / inp.numel()
        ctx.mark_non_differentiable(fac)
        return (inp >= 0) * fac, fac

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, grad_fac: torch.Tensor) -> torch.Tensor:
        inp = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[inp.abs() > 1] = 0
        return grad_input


class Sparser(torch.autograd.Function):
    """Returns a binary vector with the selected leaf set to 1 and the rest to 0."""

    @staticmethod
    def forward(ctx: Any, inp: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inp)
        inpargmax = inp.argmax(-1)
        output = torch.zeros_like(inp)
        output[torch.arange(inp.shape[0]), inpargmax] = 1
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        grad_input = grad_output.clone()
        return grad_input
