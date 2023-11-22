import numpy as np
import torch

__all__ = ("torch_to_numpy_dtype",)

_torch_to_numpy_dtype_dict = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def torch_to_numpy_dtype(torch_dtype: torch.dtype):
    return _torch_to_numpy_dtype_dict[torch_dtype]
