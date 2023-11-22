# OPPerTune

OPPerTune is an RL framework that enables systems and service developers to automatically tune various configuration
parameters and other heuristics in their codebase, rather than manually-tweaking, over time in deployment. It provides
easy-to-use API and is driven by bandit-style RL & online gradient-descent algorithms.

## Prerequisites

- Python 3 (>= 3.8)

## Installation

Install the latest version of pip

```bash
python3 -m pip install --upgrade pip
```

To setup the package locally, run

```bash
pip install .
```

## Usage

```python
# File: src/algorithms/python/examples/bluefin/example.py
import numpy as np
from oppertune import OPPerTune


def get_reward(pred) -> float:
    """Negative squared loss."""
    target = np.array([1, 700])
    return -np.square(pred - target).sum() / (1000 ** 2)


def main():
    parameters = (
        {
            'type': 'discrete',
            'name': 'p1',
            'initial_value': 5,
            'lb': 0,
            'ub': 10,
        },
        {
            'type': 'continuous',
            'name': 'p2',
            'initial_value': 100.0,
            'lb': 100.0,
            'ub': 900.0,
            'step_size': 100.0,
        },
    )

    # Initialize an instance of OPPerTune
    tuner = OPPerTune(
        algorithm='bluefin',
        parameters=parameters,
        algorithm_args=dict(
            feedback=2,
            eta=0.01,
            delta=0.1,
            random_seed=4
        )
    )

    num_iterations = 100

    for i in range(num_iterations):
        # Predict the next set of perturbed parameters
        pred, _metadata = tuner.predict()

        # Receive feedback
        reward = get_reward(np.asarray([pred['p1'], pred['p2']]))

        # Send the feedback to OPPerTune for the gradient update
        tuner.set_reward(reward, metadata=_metadata)

        if i % 25 == 0:
            print(f'Round={i}, Reward={reward}, Pred=({pred["p1"]:d}, {pred["p2"]:.4f}),'
                  f' Best=({tuner.backend.w_center[0]:.4f}, {tuner.backend.w_center[1]:.4f})')


if __name__ == '__main__':
    main()
```

## Contributing

### Setup

```bash
pip install -e "./[dev]"
```

### Style guide

To ensure your code follows the style guidelines, install `black>=23.1` and `isort>=5.12`

```shell
pip install black>=23.1
pip install isort>=5.12
```

then run,

```shell
isort . --sp=pyproject.toml
black . --config=pyproject.toml
```
