# oppertune-core

This package consists of the core classes and methods (e.g., `Categorical`, `Integer`, `Real`, `Context`, `TuningRequest`, `PredictResponse`, `to_value`) necessary for using the `oppertune-algorithms` package.

## Prerequisites

- Python 3 (>= 3.8)

1. Install the latest version of `pip`, `setuptools` and `wheel`.

    ```bash
    python -m pip install --upgrade pip setuptools wheel
    ```

## Installation

```shell
pip install .
```

## Contributing

### Setup

```bash
pip install -e .
```

### Style guide

To ensure your code follows the style guidelines, install `ruff`

```shell
pip install ruff --upgrade
```

then run,

```shell
ruff check
ruff format
```

and make sure that all warnings are addressed.
