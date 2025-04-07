"""Value objects for representing parameters and more."""

import abc
from dataclasses import asdict, dataclass, fields
from math import isinf, isnan
from typing import Any, ClassVar, Dict, Generic, Mapping, Optional, Sequence, Type, Union, overload

from typing_extensions import Literal, Self, TypedDict, TypeVar, override

__all__ = (
    "Value",
    "Categorical",
    "Integer",
    "Real",
    "CategoricalDict",
    "IntegerDict",
    "RealDict",
    "ValueDtype",
    "ValueType",
    "ValueDictType",
    "to_value",
    "to_dict",
    "transform",
    "inverse_transform",
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class Value(abc.ABC, Generic[_T]):
    """Base class for representing values."""

    type: ClassVar[str]
    dtype: ClassVar[Type]
    name: str
    val: _T

    @abc.abstractmethod
    def cast(self, value: Any) -> _T:
        """Convert the input to the `Value` object's `dtype`."""

    @classmethod
    def from_dict(cls: Type[Self], d: Mapping[str, Any]) -> Self:
        """Create a Value object from its dictionary-like representation."""
        field_value: Dict[str, Any] = {field.name: d[field.name] for field in fields(cls)}
        return cls(**field_value)

    def to_dict(self) -> Dict[str, Any]:
        """Generate the dictionary representation of the value."""
        return {"type": self.type, **asdict(self)}

    @property
    @abc.abstractmethod
    def is_constant(self) -> bool:
        """Returns true if the `Value` object can take only one value."""


@dataclass(frozen=True)
class Categorical(Value[str]):
    """Represents a `str`-based categorical value.

    Examples:
        >>> var = Categorical(name="cache_policy", val="LRU", categories=("FIFO", "FILO", "LFU", "LRU"))
        >>> print(var.encoded_value("FILO"))
        1
        >>> print(var.encoded_value("LRU"))
        3
        >>> print(var.encoded_value())
        3
        >>> print(var.category(1))
        'FILO'
        >>> print(var.n_categories)
        4
    """

    type: ClassVar[Literal["categorical"]] = "categorical"
    dtype: ClassVar[Type[str]] = str
    name: str
    val: str
    categories: Sequence[str]

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("name must be of type str")

        if not isinstance(self.val, str):
            raise TypeError("val must be of type str")

        if not isinstance(self.categories, Sequence):
            raise TypeError("categories must be of type Sequence")

        object.__setattr__(self, "categories", tuple(self.categories))
        for category in self.categories:
            if not isinstance(category, str):
                raise TypeError(f"Each category must be of type str ({category!r} is of type {type(category)})")

        if self.val not in self.categories:
            raise ValueError("val must also be in the categories list")

        if not (len(self.categories) >= 1):
            raise ValueError("Categories should have at least 1 elements")

        seen = set()
        for category in self.categories:
            if category in seen:
                raise ValueError(f"Duplicate category {category!r}")

            seen.add(category)

        self._category_to_encoded_value: Dict[str, int]
        object.__setattr__(
            self, "_category_to_encoded_value", {category: value for value, category in enumerate(self.categories)}
        )

    @override
    def cast(self, value: Union[str, int]) -> str:
        """Convert the input to its `str` equivalent category.

        If `value` is an `int`, the function returns the category corresponding to this encoded value.
        Else if `value` is a `str`, the function returns `value` itself after checking if it is a valid category.
        """
        if isinstance(value, str):
            if value not in self._category_to_encoded_value:
                raise ValueError(f"Unknown category {value!r}")

            return value

        return self.categories[value]

    def encoded_value(self, category: Optional[str] = None) -> int:
        """Return the encoded version of the category.

        Args:
            category: The category. If `None`, the encoded value of the current `val` is returned.

        Returns:
            The integer representation corresponding to the category.
        """
        if category is None:
            category = self.val

        return self._category_to_encoded_value[category]

    def category(self, value: int) -> str:
        """Convert the input to its `str` equivalent category.

        Args:
            value: The integer representation of the category

        Returns:
            The string representation corresponding to the value.
        """
        return self.categories[value]

    @override
    def to_dict(self) -> "CategoricalDict":
        return super().to_dict()  # type: ignore

    @property
    def n_categories(self) -> int:
        """Returns the total number of categories."""
        return len(self.categories)

    @property
    @override
    def is_constant(self) -> bool:
        return len(self.categories) == 1


@dataclass(frozen=True)
class Integer(Value[int]):
    """Represents a bounded integer value."""

    type: ClassVar[Literal["integer"]] = "integer"
    dtype: ClassVar[Type[int]] = int
    name: str
    val: int
    min: int
    max: int
    step: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("name must be of type str")

        if not (isinstance(self.val, int) or (isinstance(self.val, float) and self.val.is_integer())):
            raise TypeError("val must be of type int or integer-like (e.g., 1.0)")

        object.__setattr__(self, "val", int(self.val))

        if not (isinstance(self.min, int) or (isinstance(self.min, float) and self.min.is_integer())):
            raise TypeError("min must be of type int or integer-like (e.g., 1.0)")

        object.__setattr__(self, "min", int(self.min))

        if not (isinstance(self.max, int) or (isinstance(self.max, float) and self.max.is_integer())):
            raise TypeError("max must be of type int or integer-like (e.g., 1.0)")

        object.__setattr__(self, "max", int(self.max))

        if not (isinstance(self.step, int) or (isinstance(self.step, float) and self.step.is_integer())):
            raise TypeError("step must be of type int or integer-like (e.g., 1.0)")

        object.__setattr__(self, "step", int(self.step))

        if not (self.val >= self.min):
            raise ValueError("val must be greater than or equal to the min")

        if not (self.val <= self.max):
            raise ValueError("val must be less than or equal to the max")

        if not (self.step >= 1):
            raise ValueError("Step size must be at least 1")

        if not ((self.val - self.min) % self.step == 0):
            raise ValueError("Min must be reachable from initial value using step")

        if not ((self.max - self.val) % self.step == 0):
            raise ValueError("Max must be reachable from val using step")

    @override
    @staticmethod
    def cast(value: Union[int, float]) -> int:
        """Rounds the input to an `int`."""
        return round(value)

    @override
    def to_dict(self) -> "IntegerDict":
        return super().to_dict()  # type: ignore

    @property
    @override
    def is_constant(self) -> bool:
        return self.min == self.max


@dataclass(frozen=True)
class Real(Value[float]):
    """Represents a bounded floating point (real) value."""

    type: ClassVar[Literal["real"]] = "real"
    dtype: ClassVar[Type[float]] = float
    name: str
    val: float
    min: float
    max: float
    step: Union[float, None] = None

    def __post_init__(self):  # noqa: C901
        if not isinstance(self.name, str):
            raise TypeError("name must be of type str")

        if not isinstance(self.val, (float, int)):
            raise TypeError("val must be of type float or int")

        object.__setattr__(self, "val", float(self.val))

        if not isinstance(self.min, (float, int)):
            raise TypeError("min must be of type float or int")

        object.__setattr__(self, "min", float(self.min))

        if not isinstance(self.max, (float, int)):
            raise TypeError("max must be of type float or int")

        object.__setattr__(self, "max", float(self.max))

        if not (self.val >= self.min):
            raise ValueError(f"val {self.val} must be greater than or equal to the min {self.min}")

        if not (self.val <= self.max):
            raise ValueError("val must be less than or equal to the max")

        if isinf(self.val):
            raise ValueError("val must be finite")

        if isinf(self.min):
            raise ValueError("min must be finite")

        if isinf(self.max):
            raise ValueError("max must be finite")

        if isnan(self.val):
            raise ValueError("val cannot be NaN")

        if isnan(self.min):
            raise ValueError("min cannot be NaN")

        if isnan(self.max):
            raise ValueError("max cannot be NaN")

        if self.step is None:
            return

        if not isinstance(self.step, (float, int)):
            raise TypeError("step must be of type float, int or None")

        object.__setattr__(self, "step", float(self.step))

        if isinf(self.step):
            raise ValueError("step must be finite")

        if not (self.step > 0):
            raise ValueError("Step size must be positive or None")

        if not ((self.max - self.val) / self.step).is_integer():
            raise ValueError("min must be reachable from val using step")

        if not ((self.val - self.min) / self.step).is_integer():
            raise ValueError("max must be reachable from val using step")

    @override
    @staticmethod
    def cast(value: Union[int, float]) -> float:
        """Casts the input to a `float`."""
        return float(value)

    @override
    def to_dict(self) -> "RealDict":
        return super().to_dict()  # type: ignore

    @property
    @override
    def is_constant(self) -> bool:
        return self.min == self.max


class CategoricalDict(TypedDict):
    """Helper type for representing output of `Categorical`'s `to_dict` method.

    This is mainly meant for type checking. Use `Categorical` to create the value,
    which also provides validation checks.
    """

    type: Literal["categorical"]
    name: str
    val: str
    categories: Sequence[str]


class IntegerDict(TypedDict):
    """Helper type for representing output of `Integer`'s `to_dict` method.

    This is mainly meant for type checking. Use `Integer` to create the value,
    which also provides validation checks.
    """

    type: Literal["integer"]
    name: str
    val: int
    min: int
    max: int
    step: int


class RealDict(TypedDict):
    """Helper type for representing output of `Real`'s `to_dict` method.

    This is mainly meant for type checking. Use `Real` to create the value,
    which also provides validation checks.
    """

    type: Literal["real"]
    name: str
    val: float
    min: float
    max: float
    step: Union[float, None]


ValueDtype = Union[Categorical.dtype, Integer.dtype, Real.dtype]
ValueType = Union[Categorical, Integer, Real]
ValueDictType = Union[CategoricalDict, IntegerDict, RealDict]


_VALUE_CLASS: Dict[str, Type[ValueType]] = {
    "categorical": Categorical,
    "integer": Integer,
    "real": Real,
}


@overload
def to_value(x: Union[Categorical, CategoricalDict]) -> Categorical:
    ...


@overload
def to_value(x: Union[Integer, IntegerDict]) -> Integer:
    ...


@overload
def to_value(x: Union[Real, RealDict]) -> Real:
    ...


def to_value(x: Union[ValueType, ValueDictType]) -> ValueType:
    """Convert the input to its `Value` type.

    The `type` field is using for determining which type of `Value` it is.
    If input is already a Value type, it is returned as is without validations.
    """
    return _VALUE_CLASS[x["type"]].from_dict(x) if isinstance(x, dict) else x


@overload
def to_dict(x: Union[Categorical, CategoricalDict]) -> CategoricalDict:
    ...


@overload
def to_dict(x: Union[Integer, IntegerDict]) -> IntegerDict:
    ...


@overload
def to_dict(x: Union[Real, RealDict]) -> RealDict:
    ...


def to_dict(x: Union[ValueType, ValueDictType]) -> ValueDictType:
    """Return the dictionary representation of the value.

    If input is already a dictionary, it is returned as is without validations.
    """
    return x.to_dict() if isinstance(x, ValueType) else x


@overload
def transform(x: Categorical) -> Categorical:
    ...


@overload
def transform(x: Integer) -> Integer:
    ...


@overload
def transform(x: Real) -> Union[Integer, Real]:
    ...


def transform(x: Union[Categorical, Integer, Real]) -> Union[Categorical, Integer, Real]:
    """Simplify the representation of value objects. Name is not changed."""
    if isinstance(x, Categorical):
        return x

    if x.is_constant:
        return Integer(name=x.name, val=0, min=0, max=0)

    if isinstance(x, Integer):
        return Integer(
            name=x.name,
            val=(x.val - x.min) // x.step,
            min=0,
            max=(x.max - x.min) // x.step,
        )

    assert isinstance(x, Real)
    if x.step is None:
        return Real(
            name=x.name,
            val=(x.val - x.min) / (x.max - x.min),
            min=0.0,
            max=1.0,
        )

    # (x.val - x.min) / x.step and (x.max - x.min) / x.step both must be integer-like (e.g, 5.0)
    # We are using `round` instead of `int` to prevent almost integer-like floats to be
    # converted to the wrong int (e.g., int(4.99999) -> 4 but it should have been 5).
    return Integer(
        name=x.name,
        val=round((x.val - x.min) / x.step),
        min=0,
        max=round((x.max - x.min) / x.step),
    )


@overload
def inverse_transform(x: Categorical.dtype, reference: Categorical) -> Categorical.dtype:
    ...


@overload
def inverse_transform(x: Union[Integer.dtype, Real.dtype], reference: Integer) -> Integer.dtype:
    ...


@overload
def inverse_transform(x: Union[Integer.dtype, Real.dtype], reference: Real) -> Real.dtype:
    ...


def inverse_transform(
    x: Union[Categorical.dtype, Integer.dtype, Real.dtype],
    reference: Union[Categorical, Integer, Real],
) -> Union[Categorical.dtype, Integer.dtype, Real.dtype]:
    """Transforms the input value as per the given `reference` object."""
    if isinstance(reference, Categorical):
        assert isinstance(x, Categorical.dtype)
        return x

    if not isinstance(x, (Integer.dtype, Real.dtype)):
        raise TypeError(f"Invalid value {x!r} for {type(reference)} reference type")

    if isinstance(reference, Integer):
        # x (int) in {0, 1, ..., N}
        # rounding to convert integer-like floats (e.g., 5.0) to integer
        return round(x) * reference.step + reference.min

    assert isinstance(reference, Real)
    if reference.step is None:
        # x (float) in [0, 1]
        return float(x * (reference.max - reference.min) + reference.min)

    # x (int) in {0, 1, ..., N}
    return float(x * reference.step + reference.min)
