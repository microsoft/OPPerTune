import torch
from numpy import typing as npt
from typing_extensions import TypeVar

T = TypeVar("T", float, npt.NDArray, torch.Tensor)

__all__ = (
    "normalize",
    "normalize_",
    "denormalize",
    "denormalize_",
)


def normalize(v: T, v_min: T, v_max: T, inplace: bool = False) -> T:
    if inplace:
        v -= v_min
    else:
        v = v - v_min

    v /= v_max - v_min
    return v


def normalize_(v: T, v_min: T, v_max: T) -> T:
    return normalize(v, v_min, v_max, inplace=True)


def denormalize(v: T, v_min: T, v_max: T, inplace: bool = False) -> T:
    if inplace:
        v *= v_max - v_min
    else:
        v = v * (v_max - v_min)

    v += v_min
    return v


def denormalize_(v: T, v_min: T, v_max: T) -> T:
    return denormalize(v, v_min, v_max, inplace=True)
