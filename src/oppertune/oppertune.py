import io
from copy import deepcopy
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Optional, Union

import joblib

from .backends.backend import get_algorithm_backend_class
from .backends.base import PredictResponse
from .values import ContinuousValue, ValueDictType, ValueType, to_value

__all__ = (
    "OPPerTune",
    "dump",
    "dumps",
    "load",
    "loads",
)

_DEFAULT_ALGORITHM = "hybrid_solver"
_DEFAULT_ALGORITHM_CONFIG = {
    "numerical_solver": "bluefin",
    "numerical_solver_args": {
        "feedback": 1,
        "eta": 0.01,
        "delta": 0.1,
        "normalize": True,
        "optimizer": "sgd",
    },
    "categorical_solver": "exponential_weights",
}


class OPPerTune:
    def __init__(
        self,
        parameters: Iterable[Union[ValueDictType, ValueType]],
        algorithm: Optional[str] = None,
        algorithm_args: Optional[dict] = None,
    ):
        """
        Finds the correct backend corresponding to the :code:`algorithm`, and initializes it using
        the :code:`parameters` and :code:`algorithm_args`.

        Args:
            parameters: The list of parameters to tune
            algorithm: The algorithm OPPerTune should use
            algorithm_args: The arguments required by the chosen algorithm
        """
        if algorithm is None:
            assert algorithm_args is None, "Must specify the algorithm with the algorithm arguments"
            self.algorithm = _DEFAULT_ALGORITHM
            algorithm_args = _DEFAULT_ALGORITHM_CONFIG
        else:
            self.algorithm = algorithm
            algorithm_args = algorithm_args or {}

        parameters = tuple(to_value(deepcopy(p)) for p in parameters)
        assert len(parameters) >= 1, "Must provide at least 1 parameter"

        parameter_names = tuple(param.name for param in parameters)
        assert len(parameter_names) == len(set(parameter_names)), "Duplicate parameter names"

        backend_class = get_algorithm_backend_class(self.algorithm)
        self.backend = backend_class(parameters=parameters, **algorithm_args)

    def predict(self, **kwargs) -> PredictResponse:
        """
        Calls the backend's predict method

        Args:
            **kwargs: Any keyword arguments required by the backend

        Returns:
            A NamedTuple, with 2 elements:
                parameters: A dictionary with keys as the parameter names and the values as the parameter values
                metadata: Any extra data that an algorithm needs in the `set_reward` call
        """
        return self.backend.predict(**kwargs)

    def set_reward(self, reward: Union[float, int], metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Calls the backend's set_reward method

        Args:
            reward: The reward needed by the backend to update
            metadata: Any extra data that an algorithm needs, which is sent back in the `predict` call
            **kwargs: Any keyword arguments required by the backend

        Returns:
            None
        """
        return self.backend.set_reward(reward, metadata=metadata, **kwargs)

    def __str__(self):
        return f'OPPerTune(algorithm="{self.algorithm}")'


def dump(obj: OPPerTune, file: Union[str, Path, BinaryIO, io.BytesIO], **kwargs):
    state = {
        "algorithm": obj.algorithm,
        "backend": obj.backend.dumps(),
    }

    compression_kwargs = {} if "compress" in kwargs else {"compress": ("zlib", 3)}

    if isinstance(file, (BinaryIO, io.BytesIO)):
        joblib.dump(state, file, **compression_kwargs, **kwargs)
    else:
        with open(file, "wb") as f:
            joblib.dump(state, f, **compression_kwargs, **kwargs)


def dumps(obj: OPPerTune, **kwargs):
    file = io.BytesIO()
    dump(obj, file, **kwargs)
    return file.getvalue()


def load(file: Union[str, Path, BinaryIO, io.BytesIO], **kwargs) -> OPPerTune:
    if isinstance(file, (BinaryIO, io.BytesIO)):
        state = joblib.load(file, **kwargs)
    else:
        with open(file, "rb") as f:
            state = joblib.load(f, **kwargs)

    # Create an object with dummy parameters
    obj = OPPerTune([ContinuousValue(name="_", initial_value=0, lb=0, ub=1)], algorithm="identity")
    obj.algorithm = state["algorithm"]
    backend_class = get_algorithm_backend_class(state["algorithm"])
    obj.backend = backend_class.loads(state["backend"])
    return obj


def loads(buffer: bytes, **kwargs) -> OPPerTune:
    file = io.BytesIO(buffer)
    return load(file, **kwargs)
