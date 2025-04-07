import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Dict, Generic, Mapping, Type, Union
from uuid import uuid4

# Generic and typing.NamedTuple do not work together: https://github.com/python/typing_extensions/pull/44
from typing_extensions import NamedTuple, Self, TypeVar

__all__ = (
    "ContextIdType",
    "ContextDataType",
    "PredictionType",
    "PredictDataType",
    "RequestIdType",
    "RewardType",
    "RewardDataType",
    "NullContextId",
    "NullRequestId",
    "generate_request_id",
    "generate_context_id",
    "Context",
    "TuningRequest",
    "PredictResponse",
)

_ValueType = TypeVar("_ValueType")

ContextIdType = str
ContextDataType = TypeVar("ContextDataType", bound=Mapping[str, Any], default=Mapping[str, Any], covariant=True)
PredictionType = Mapping[str, _ValueType]
PredictDataType = Mapping[str, Any]
RequestIdType = str
RewardType = float
RewardDataType = Mapping[str, Any]
MetadataType = Any

NullContextId = ContextIdType()  # Default context ID if context data is None
NullRequestId = RequestIdType()


def generate_request_id() -> RequestIdType:
    """Generate a random request ID."""
    return RequestIdType(uuid4())


def generate_context_id(context_data: Union[ContextDataType, None]) -> ContextIdType:
    """Generate context ID based on the input.

    To generate the ID, `context_data` is converted to its JSON string representation with its
    keys sorted alphabetically (to ensure the generated ID is the same regardless of the order
    of the keys). Finally, this string is hashed using the SHA256 algorithm.

    Important:
        For many algorithms, `context_data` will be `None`. Therefore, for simplicity,
        if `context_data` is `None` the generated context ID is an empty string.
    """
    if context_data is None:
        return RequestIdType()

    serialized_data = json.dumps(context_data, sort_keys=True)
    return sha256(serialized_data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Context(Generic[ContextDataType]):
    data: Union[ContextDataType, None] = None
    id: ContextIdType = NullContextId

    def __post_init__(self):
        if self.id == NullContextId:
            object.__setattr__(self, "id", generate_context_id(self.data))

    def to_dict(self) -> Dict[str, Any]:
        """Returns the dictionary representation of the object."""
        return asdict(self)


@dataclass
class TuningRequest:
    id: RequestIdType
    reward: Union[RewardType, None]
    context: Context

    def to_dict(self) -> Dict[str, Any]:
        """Returns the dictionary representation of the object."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[Self], d: Dict[str, Any]) -> Self:
        """Constructs a `TuningRequest` object from its dictionary representation."""
        ctx = d.get("context")
        context = Context() if ctx is None else Context(data=ctx["data"], id=ctx["id"])
        return cls(id=d["id"], reward=d["reward"], context=context)


class PredictResponse(NamedTuple, Generic[_ValueType]):
    prediction: PredictionType[_ValueType]
    request_id: RequestIdType

    def to_dict(self) -> Dict[str, Any]:
        """Returns the dictionary representation of the object."""
        return asdict(self)
