from math import sqrt
from typing import Iterable, Union

import numpy as np

__all__ = (
    "absolute_error",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "squared_sum",
)


def absolute_error(a: Iterable[Union[int, float]], b: Iterable[Union[int, float]]) -> float:
    _a = a if isinstance(a, np.ndarray) else np.array(list(a))
    _b = b if isinstance(b, np.ndarray) else np.array(list(b))
    assert _a.shape == _b.shape
    return float(np.sum(np.abs(_a - _b)))


def mean_absolute_error(a: Iterable[Union[int, float]], b: Iterable[Union[int, float]]) -> float:
    _a = a if isinstance(a, np.ndarray) else np.array(list(a))
    _b = b if isinstance(b, np.ndarray) else np.array(list(b))
    assert _a.shape == _b.shape
    return float(np.mean(np.abs(_a - _b)))


def mean_squared_error(a: Iterable[Union[int, float]], b: Iterable[Union[int, float]]) -> float:
    _a = a if isinstance(a, np.ndarray) else np.array(list(a))
    _b = b if isinstance(b, np.ndarray) else np.array(list(b))
    assert _a.shape == _b.shape
    return float(np.mean((_a - _b) ** 2))


def root_mean_squared_error(a: Iterable[Union[int, float]], b: Iterable[Union[int, float]]) -> float:
    return sqrt(mean_squared_error(a, b))


def squared_sum(a: Iterable[Union[int, float]], b: Iterable[Union[int, float]]) -> float:
    _a = a if isinstance(a, np.ndarray) else np.array(list(a))
    _b = b if isinstance(b, np.ndarray) else np.array(list(b))
    assert _a.shape == _b.shape
    return float(np.sum(np.square(_a - _b)))
