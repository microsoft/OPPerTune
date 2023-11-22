from typing import Union

import numpy as np
import torch

Vector = Union[np.ndarray, torch.Tensor]


def normalize(v: Vector, lb: Vector, ub: Vector, inplace: bool = False):
    if inplace:
        v -= lb
    else:
        v = v - lb

    v /= ub - lb
    return v


def normalize_(v: Vector, lb: Vector, ub: Vector):
    return normalize(v, lb, ub, inplace=True)


def denormalize(v: Vector, lb: Vector, ub: Vector, inplace: bool = False):
    if inplace:
        v *= ub - lb
    else:
        v = v * (ub - lb)

    v += lb
    return v


def denormalize_(v: Vector, lb: Vector, ub: Vector):
    return denormalize(v, lb, ub, inplace=True)
