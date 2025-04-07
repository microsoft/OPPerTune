# Tuning Algorithms

This module contains all the tuning algorithms that OPPerTune provides.

## Creating a custom algorithm

An algorithm must inherit the `Algorithm` class and define a basic set of functionality.
An example is shown below:

```python
# File: src/oppertune/algorithms/new_algorithm/new_algorithm.py
from typing import Any, Dict, Iterable, Optional, Union

from typing_extensions import override

from oppertune.core.values import Categorical, Integer, Real
from ..base import Algorithm
from oppertune.core.types import PredictResponse, _PredictResponse, _TuningRequest

__all__ = ("NewAlgorithm",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class NewAlgorithm(Algorithm):
    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        required_untransformed_parameters = False
        supports_context = False
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        random_seed: Optional[int] = None,
        # Other constructor parameters
    ):
        super().__init__(parameters, random_seed=random_seed)
        ...

    @override
    def _predict(self, context: None = None, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        ...

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        ...

    @property
    @override
    def iteration(self) -> int:
        ...
```

```python
# File: src/oppertune/backends/new_algorithm/__init__.py
from .new_algorithm import NewAlgorithm
```

## Registering your algorithm with OPPerTune

After creating your algorithm, register it in `all.py` as follows:

```python
from .new_algorithm import NewAlgorithm

ALGORITHMS = {
    "new_algorithm": NewAlgorithm,
    # Existing algorithms...
}
```
