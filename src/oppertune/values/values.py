import abc
from copy import deepcopy
from typing import Any, Dict, Hashable, Optional, Sequence, SupportsRound, Union

from typing_extensions import Literal, TypedDict

__all__ = (
    "CategoricalValueDict",
    "ContinuousValueDict",
    "DiscreteValueDict",
    "ValueDictType",
    "Value",
    "CategoricalValue",
    "ContinuousValue",
    "DiscreteValue",
    "ValueType",
    "to_value",
)


class CategoricalValueDict(TypedDict):
    """
    Use CategoricalValue to create the value, which also provides validation checks.
    This is mainly meant for type checking.
    """

    name: str
    initial_value: Hashable
    categories: Dict[Hashable, int]
    type: Literal["categorical"]


class ContinuousValueDict(TypedDict):
    """
    Use ContinuousValue to create the value, which also provides validation checks.
    This is mainly meant for type checking.
    """

    name: str
    initial_value: float
    lb: float
    ub: float
    step_size: Union[float, None]
    type: Literal["continuous"]


class DiscreteValueDict(TypedDict):
    """
    Use DiscreteValue to create the value, which also provides validation checks.
    This is mainly meant for type checking.
    """

    name: str
    initial_value: int
    lb: int
    ub: int
    step_size: int
    type: Literal["discrete"]


ValueDictType = Union[CategoricalValueDict, ContinuousValueDict, DiscreteValueDict]


class Value(abc.ABC):
    __slots__ = (
        "name",
        "initial_value",
    )

    @abc.abstractmethod
    def cast(self, value: Any) -> Any:
        pass

    @abc.abstractmethod
    def to_dict(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        pass

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(f"{s}={getattr(self, s)}" for s in self.__slots__))


class CategoricalValue(Value):
    __slots__ = (
        "name",
        "initial_value",
        "_categories",
        "_values",
    )

    def __init__(
        self,
        name: str,
        initial_value: Hashable,
        categories: Union[Sequence[Hashable], Dict[Hashable, int]],
        type: Literal["categorical"] = "categorical",  # For ease of deserialization
    ):
        assert initial_value in categories, "Initial value must be a valid category"
        assert len(categories) >= 2, "Categories should have at least 2 elements"
        assert len(categories) == len(set(categories)), "Categories should be unique"
        if isinstance(categories, dict):
            assert len(categories.values()) == len(
                set(categories.values())
            ), "Multiple categories cannot have the same mapped value"

        self.name = name
        self.initial_value = deepcopy(initial_value)
        self._categories = {value: category for value, category in enumerate(categories)}
        self._categories = deepcopy(self._categories)
        self._values = {category: value for value, category in self._categories.items()}
        self._values = deepcopy(self._values)

    @property
    def type(self):
        return "categorical"

    @property
    def categories(self):
        return tuple(self._categories.values())

    def value(self, category: Hashable, **kwargs):
        """
        Args:
            category: The category

        Keyword Args:
            default: The default value to return if the category is not found.
                If not specified and the category is not found, an error is raised.
        """
        if "default" in kwargs:
            return self._values.get(category, kwargs["default"])

        return self._values[category]

    def category(self, value: int):
        return self._categories[value]

    cast = category

    @property
    def n_categories(self):
        return len(self._categories)

    def to_dict(self):
        return CategoricalValueDict(
            type=self.type,
            name=self.name,
            initial_value=deepcopy(self.initial_value),
            categories=deepcopy(self._values),
        )


class ContinuousValue(Value):
    __slots__ = (
        "name",
        "initial_value",
        "lb",
        "ub",
        "step_size",
    )

    def __init__(
        self,
        name: str,
        initial_value: float,
        lb: float,
        ub: float,
        step_size: Union[float, None] = None,
        type: Literal["continuous"] = "continuous",  # For ease of deserialization
    ):
        assert lb < ub, "Lower bound must be strictly less than the upper bound"
        assert lb <= initial_value, "Initial value must be greater than or equal to the lower bound"
        assert initial_value <= ub, "Initial value must be less than or equal to the upper bound"

        if step_size is not None:
            assert step_size > 0, "Step size must be positive or None"
            assert (
                (ub - initial_value) / step_size
            ).is_integer(), "Lower bound must be reachable from initial value using step size"
            assert (
                (initial_value - lb) / step_size
            ).is_integer(), "Upper bound must be reachable from initial value using step size"

        self.name = name
        self.initial_value = initial_value
        self.lb = lb
        self.ub = ub
        self.step_size = step_size

    @property
    def type(self):
        return "continuous"

    @staticmethod
    def cast(value):
        return float(value)

    def to_dict(self):
        return ContinuousValueDict(
            type=self.type,
            name=self.name,
            initial_value=self.initial_value,
            lb=self.lb,
            ub=self.ub,
            step_size=self.step_size,
        )


class DiscreteValue(Value):
    __slots__ = (
        "name",
        "initial_value",
        "lb",
        "ub",
        "step_size",
    )

    def __init__(
        self,
        name: str,
        initial_value: int,
        lb: int,
        ub: int,
        step_size: int = 1,
        type: Literal["discrete"] = "discrete",  # For ease of deserialization
    ):
        assert lb < ub, "Lower bound must be strictly less than the upper bound"
        assert lb <= initial_value, "Initial value must be greater than or equal to the lower bound"
        assert initial_value <= ub, "Initial value must be less than or equal to the upper bound"

        assert step_size >= 1, "Step size must be at least 1"
        assert (initial_value - lb) % step_size == 0, "Lower bound must be reachable from initial value using step size"
        assert (ub - initial_value) % step_size == 0, "Upper bound must be reachable from initial value using step size"

        self.name = name
        self.initial_value = initial_value
        self.lb = lb
        self.ub = ub
        self.step_size = step_size

    @property
    def type(self):
        return "discrete"

    @staticmethod
    def cast(value: SupportsRound):
        return round(value)

    def to_dict(self):
        return DiscreteValueDict(
            type=self.type,
            name=self.name,
            initial_value=self.initial_value,
            lb=self.lb,
            ub=self.ub,
            step_size=self.step_size,
        )


ValueType = Union[CategoricalValue, ContinuousValue, DiscreteValue]

_VALUE_CLASS = {
    "categorical": CategoricalValue,
    "continuous": ContinuousValue,
    "discrete": DiscreteValue,
}


def to_value(
    x: Union[ValueDictType, ValueType],
    type: Optional[Literal["categorical", "continuous", "discrete"]] = None,
) -> ValueType:
    if isinstance(x, dict):
        t = x["type"] if type is None else type
        return _VALUE_CLASS[t](**x)

    return x
