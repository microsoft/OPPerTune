from typing import Dict, Type

from .base import AlgorithmBackend
from .bayesian_optimization import BayesianOptimization
from .bluefin import BlueFin
from .ddpg import DDPG
from .dgt import DGT
from .exponential_weights import ExponentialWeights
from .exponential_weights_slates import ExponentialWeightsSlates
from .hopt import HOpt
from .hybrid_solver import HybridSolver
from .identity import Identity
from .slates import Slates

_BACKENDS: Dict[str, Type[AlgorithmBackend]] = {
    "bayesian_optimization": BayesianOptimization,
    "bluefin": BlueFin,
    "ddpg": DDPG,
    "dgt": DGT,
    "exponential_weights": ExponentialWeights,
    "exponential_weights_slates": ExponentialWeightsSlates,
    "hopt": HOpt,
    "hybrid_solver": HybridSolver,
    "identity": Identity,
    "slates": Slates,
}


def get_algorithm_backend_class(name: str):
    return _BACKENDS[name]
