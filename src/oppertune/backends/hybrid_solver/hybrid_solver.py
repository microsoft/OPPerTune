from typing import Any, Dict, Iterable, Optional, Tuple, Union

from typing_extensions import TypedDict

from ...values import CategoricalValue, ContinuousValue, DiscreteValue
from ..base import AlgorithmBackend, PredictResponse

__all__ = ("HybridSolver",)


class Metadata(TypedDict):
    numerical: Union[Dict[str, Any], None]
    categorical: Union[Dict[str, Any], None]


class HybridSolver(AlgorithmBackend):
    def __init__(
        self,
        parameters: Iterable[Union[ContinuousValue, DiscreteValue, CategoricalValue]],
        numerical_solver: Union[str, Any] = "bluefin",
        categorical_solver: Union[str, Any] = "exponential_weights",
        numerical_solver_args: Optional[dict] = None,
        categorical_solver_args: Optional[dict] = None,
    ):
        """
        Args:
            numerical_solver: Either the name of numerical backend or the instantiated backend object
            categorical_solver: Either the name of categorical backend or the instantiated backend object
        """
        if numerical_solver_args is None:
            numerical_solver_args = {}

        if categorical_solver_args is None:
            categorical_solver_args = {}

        self.params = tuple(parameters)
        self.numerical_params, self.categorical_params = HybridSolver._get_param_split(self.params)

        self.numerical_solver = None
        self.categorical_solver = None

        from ..backend import get_algorithm_backend_class  # To avoid circular import

        if self.numerical_params:
            if isinstance(numerical_solver, str):
                numerical_solver_class = get_algorithm_backend_class(numerical_solver)
                self.numerical_solver = numerical_solver_class(self.numerical_params, **numerical_solver_args)
            else:
                assert numerical_solver_args is None
                self.numerical_solver = numerical_solver

        if self.categorical_params:
            if isinstance(categorical_solver, str):
                categorical_solver_class = get_algorithm_backend_class(categorical_solver)
                self.categorical_solver = categorical_solver_class(self.categorical_params, **categorical_solver_args)
            else:
                assert categorical_solver_args is None
                self.categorical_solver = categorical_solver

    def predict(self):
        numerical_pred, numerical_metadata = {}, None
        categorical_pred, categorical_metadata = {}, None

        if self.numerical_solver:
            numerical_pred, numerical_metadata = self.numerical_solver.predict()

        if self.categorical_solver:
            categorical_pred, categorical_metadata = self.categorical_solver.predict()

        param_dict = {
            p.name: numerical_pred[p.name] if p.name in numerical_pred else categorical_pred[p.name]
            for p in self.params
        }
        _metadata = Metadata(numerical=numerical_metadata, categorical=categorical_metadata)

        return PredictResponse(parameters=param_dict, metadata=_metadata)

    def set_reward(self, reward: float, metadata: Optional[Metadata] = None):
        """
        The solvers get the passed reward
        """
        if metadata is None:
            metadata = {}

        if self.numerical_solver:
            self.numerical_solver.set_reward(reward, metadata=metadata.get("numerical"))

        if self.categorical_solver:
            self.categorical_solver.set_reward(reward, metadata=metadata.get("categorical"))

    @staticmethod
    def _get_categorical_representation(pred, params: Iterable[CategoricalValue]) -> str:
        return ",".join(str(param.value(pred[param.name])) for param in params)

    @staticmethod
    def _get_param_split(
        parameters: Iterable[Union[ContinuousValue, DiscreteValue, CategoricalValue]]
    ) -> Tuple[Iterable[Union[ContinuousValue, DiscreteValue]], Iterable[CategoricalValue]]:
        numerical_params = []
        categorical_params = []

        for param in parameters:
            if isinstance(param, (ContinuousValue, DiscreteValue)):
                numerical_params.append(param)
            elif isinstance(param, CategoricalValue):
                categorical_params.append(param)
            else:
                raise ValueError(f"Unknown parameter type={param}")

        return numerical_params, categorical_params
