from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import typing as npt


def unit_normal_vector(
    shape: Union[None, int, Tuple[int, ...], List[int]],
    generator: np.random.Generator,
    dtype: Optional[np.dtype] = None,
) -> npt.NDArray:
    v = np.asarray(generator.normal(size=shape), dtype=dtype)
    norm = np.linalg.norm(v)

    if norm != 0:
        v /= norm
    else:  # norm == 0 is very unlikely, nevertheless handling it
        v[0] = 1

    return v


def unit_normal_vector_like(v: npt.NDArray, generator: np.random.Generator) -> npt.NDArray:
    return unit_normal_vector(v.shape, generator=generator, dtype=v.dtype)
