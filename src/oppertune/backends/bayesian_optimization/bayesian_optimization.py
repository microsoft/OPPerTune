from contextlib import contextmanager
from typing import List, Optional, Union

import numpy as np
from skopt import Optimizer
from skopt.space import Categorical, Integer, Real
from typing_extensions import TypedDict

from ...values import CategoricalValue, ContinuousValue, DiscreteValue
from ..base import AlgorithmBackend, PredictResponse

__all__ = ("BayesianOptimization",)


class Metadata(TypedDict):
    x: dict


@contextmanager
def substitute_numpy_int():
    """
    If `np.int` is not defined, it temporarily assigns it to `int` within its scope.
    """
    if hasattr(np, "int"):
        yield
    else:
        # `np.int` was removed in NumPy 1.24 - https://numpy.org/devdocs/release/1.24.0-notes.html#expired-deprecations
        # `np.int` can be replaced with either `int` or `np.int_`.
        # Could have done this either in `BayesianOptimization`'s `__init__` or at the module level.
        # But this is to limit the scope of this change.
        np.int = int

        try:
            yield
        finally:
            del np.int


def parameter_to_skopt(
    parameter: Union[CategoricalValue, ContinuousValue, DiscreteValue]
) -> Union[Categorical, Integer, Real]:
    """
    Note:
        Will have to use it with the `substitute_numpy_int` context manager if `np.int` is not defined
    """
    if isinstance(parameter, CategoricalValue):
        return Categorical(categories=parameter.categories, name=parameter.name)

    if isinstance(parameter, DiscreteValue):
        return Integer(low=parameter.lb, high=parameter.ub, name=parameter.name)

    if isinstance(parameter, ContinuousValue):
        return Real(low=parameter.lb, high=parameter.ub, name=parameter.name)

    raise Exception(f"Invalid parameter type: {type(parameter)}")


class BayesianOptimization(AlgorithmBackend):
    def __init__(
        self,
        parameters: List[Union[CategoricalValue, ContinuousValue, DiscreteValue]],
        optimizer_kwargs: Optional[dict] = None,
        random_seed: Optional[int] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        assert "dimensions" not in optimizer_kwargs, "Use parameters to specify bounds"

        with substitute_numpy_int():
            dimensions = tuple(parameter_to_skopt(p) for p in parameters)

        self.optimizer = Optimizer(dimensions, random_state=random_seed, **optimizer_kwargs)
        self.__parameters = tuple(parameters)

    def predict(self):
        with substitute_numpy_int():
            x = self.optimizer.ask()

        # Convert from NumPy to Python type
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            for i in range(len(x)):
                if isinstance(x[i], np.generic):
                    x[i] = x[i].item()

        parameters = self.__get_prediction(x)
        return PredictResponse(parameters=parameters, metadata=Metadata(x=parameters))

    def set_reward(self, reward: float, metadata: Metadata):
        # BO solves cost minimization, but the reward is passed assuming reward maximization.
        # So we negate the reward to make it compatible with BO (so that the reward acts as cost).
        reward = -reward
        x = metadata["x"]

        if isinstance(x, dict):
            assert len(x) == len(self.__parameters)
            x = tuple(x[p.name] for p in self.__parameters)

        with substitute_numpy_int():
            self.optimizer.tell(x, reward)

    def __get_prediction(self, x: Union[list, tuple]) -> dict:
        assert len(self.__parameters) == len(x), "Lengths of parameters ({}) and x ({}) must be the same".format(
            len(self.__parameters), len(x)
        )

        prediction = {}
        for i, p in enumerate(self.__parameters):
            value = x[i]

            if isinstance(p, ContinuousValue):
                ss = p.step_size
                if ss is not None:
                    lb = p.lb
                    n = round(1 + (x[i] - lb) / ss)
                    value = float(lb + (n - 1) * ss)

            elif isinstance(p, DiscreteValue):
                ss = p.step_size or 1
                if ss != 1:
                    lb = p.lb
                    n = round(1 + (x[i] - lb) / ss)
                    value = round(lb + (n - 1) * ss)

            prediction[p.name] = value

        return prediction
