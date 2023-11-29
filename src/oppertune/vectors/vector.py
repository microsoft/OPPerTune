from typing import List, Optional, Tuple, Union

import numpy as np


def unit_normal_vector(
    shape: Union[None, int, Tuple[int, ...], List[int]],
    generator: np.random.Generator,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    v = np.asarray(generator.normal(size=shape), dtype=dtype)
    norm = np.linalg.norm(v)

    if norm != 0:
        v /= norm
    else:  # norm == 0 is very unlikely, nevertheless handling it
        v[0] = 1

    return v


def unit_normal_vector_like(v: np.ndarray, generator: np.random.Generator):
    return unit_normal_vector(v.shape, generator=generator, dtype=v.dtype)
