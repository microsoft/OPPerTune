from typing import Any, Dict, Iterable, Mapping, Optional, Type, Union

from oppertune.core.values import ValueDictType, ValueType

from .autoscope import AutoScope
from .base import Algorithm
from .bayesian_optimization import BayesianOptimization
from .bluefin import BlueFin
from .ddpg import DDPG
from .exponential_weights import ExponentialWeights
from .exponential_weights_slates import ExponentialWeightsSlates
from .hopt import HOpt
from .hybrid_solver import HybridSolver
from .identity import Identity
from .slates import Slates
from .uniform_random import UniformRandom

__all__ = (
    "create_tuning_instance",
    "get_algorithm_class",
)

ALGORITHMS: Dict[str, Type[Algorithm]] = {
    "autoscope": AutoScope,
    "bayesian_optimization": BayesianOptimization,
    "bluefin": BlueFin,
    "ddpg": DDPG,
    "exponential_weights": ExponentialWeights,
    "exponential_weights_slates": ExponentialWeightsSlates,
    "hopt": HOpt,
    "hybrid_solver": HybridSolver,
    "identity": Identity,
    "slates": Slates,
    "uniform_random": UniformRandom,
}


def create_tuning_instance(
    parameters: Iterable[Union[ValueType, ValueDictType]],
    algorithm: str = "hybrid_solver",
    algorithm_args: Optional[Mapping[str, Any]] = None,
) -> Algorithm:
    """Create a tuning instance using the given parameters and algorithm."""
    return ALGORITHMS[algorithm](parameters, **(algorithm_args or {}))


def get_algorithm_class(algorithm: str) -> Type[Algorithm]:
    """Return the class corresponding to `algorithm`."""
    return ALGORITHMS[algorithm]
