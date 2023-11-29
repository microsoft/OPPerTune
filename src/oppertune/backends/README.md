# Backend Interface

A backend must implement the following interface

```python
# File: src/oppertune/backends/new_backend/new_backend.py
from typing import Any, Dict, Iterable, Union

from ...values import CategoricalValue, ContinuousValue, DiscreteValue
from ..base import AlgorithmBackend, PredictResponse


class NewBackend(AlgorithmBackend):
    def __init__(
        self,
        parameters: Iterable[Union[CategoricalValue, ContinuousValue, DiscreteValue]],
        # Other function parameters
    ):
        ...

    def predict(
        self,
        # Other function parameters
    ) -> PredictResponse:
        ...

    def set_reward(
        self,
        reward: Union[float, int],
        metadata,
        # Other function parameters
    ):
        ...
```

```python
# File: src/oppertune/backends/new_backend/__init__.py
from .new_backend import NewBackend
```

## Registering backend with OPPerTune

After creating your backend, register it in `src/oppertune/backends/backend.py` as follows:

```python
from .new_backend import NewBackend

_BACKENDS = {
    "new_backend": NewBackend,
    # Existing backends...
}
```