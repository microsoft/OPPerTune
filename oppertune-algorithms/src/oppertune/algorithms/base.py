import abc
import io
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Generic, Iterable, List, Optional, Sequence, Tuple, Type, Union

import joblib
from typing_extensions import NamedTuple, Self, TypeVar

from oppertune.core.types import (
    Context,
    ContextIdType,
    MetadataType,
    NullContextId,
    NullRequestId,
    PredictDataType,
    PredictionType,
    PredictResponse,
    RequestIdType,
    RewardDataType,
    RewardType,
    TuningRequest,
    generate_request_id,
)
from oppertune.core.values import ValueDictType, ValueType, inverse_transform, to_value, transform

__all__ = ("Algorithm",)

_ParameterValueType = TypeVar("_ParameterValueType")
_RewardDataType = TypeVar("_RewardDataType", bound=RewardDataType, default=RewardDataType)


@dataclass
class _TuningRequest(Generic[_ParameterValueType, _RewardDataType]):
    id: RequestIdType = field(init=False, default_factory=generate_request_id)
    context: Context = field(default_factory=Context)
    predict_data: Union[PredictDataType, None] = None
    prediction: PredictionType[_ParameterValueType] = field(default_factory=dict)
    metadata: MetadataType = None
    reward: Union[RewardType, None] = None
    reward_data: Union[_RewardDataType, None] = None


class _PredictResponse(NamedTuple, Generic[_ParameterValueType]):
    prediction: PredictionType[_ParameterValueType]
    metadata: MetadataType = None


class Algorithm(abc.ABC, Generic[_ParameterValueType]):
    """Abstract base class for all tuning algorithms."""

    class Meta:
        """Class used for defining metadata about the algorithm."""

        # In initialization
        supported_parameter_types: Tuple[Type[ValueType], ...] = ()
        requires_untransformed_parameters: bool = False

        # In predict
        supports_context: bool = False

        # In set_reward
        supports_single_reward: bool = True
        supports_sequence_of_rewards: bool = False

    def __init__(
        self,
        parameters: Iterable[Union[ValueType, ValueDictType]],
        *args,  # noqa: ANN002
        random_seed: Optional[int] = None,
        **kwargs,  # noqa: ANN003
    ):
        super().__init__(*args, **kwargs)

        params = tuple(to_value(deepcopy(param)) for param in parameters)
        if len(params) == 0:
            raise ValueError("Parameter list cannot be empty")

        # Check if the passed parameter types are supported by the algorithm
        for param in params:
            if not isinstance(param, self.Meta.supported_parameter_types):
                raise TypeError(
                    f"{self.__class__.__name__} only supports the following "
                    f"parameter types - {self.Meta.supported_parameter_types}, "
                    f"but {param.name} is of type {type(param)}."
                )

        # Check for duplicate parameter names
        parameter_names = set()
        for param in params:
            if param.name in parameter_names:
                raise ValueError(f"Duplicate parameter name {param.name!r}")

            parameter_names.add(param.name)

        self._raw_params = params  # Read-only
        # self.params does not have any constant parameters
        if self.Meta.requires_untransformed_parameters:
            self.params = tuple(param for param in params if not param.is_constant)  # Read-only
        else:
            self.params = tuple(transform(param) for param in params if not param.is_constant)  # Read-only

        self._requests_manager = TuningRequestsManager()
        self._random_seed = random_seed
        self._iteration = 0

    def predict(
        self,
        context: Optional[Context] = None,
        predict_data: Optional[PredictDataType] = None,
    ) -> PredictResponse:
        """Get the predicted parameter values for the given context."""
        if context is None:
            context = Context()

        request = self._requests_manager.get_request_by_context_id(context.id)
        if request is None or self.Meta.supports_sequence_of_rewards:
            _prediction, metadata = self._predict(context, predict_data)

            # Denormalize parameters to their original range and type.
            # Since self.params has only non-constant parameters, we skip saving the constant parameters here.
            if self.Meta.requires_untransformed_parameters:
                # To ensure parameters are returned in their original order.
                prediction = {
                    param.name: _prediction[param.name] for param in self._raw_params if not param.is_constant
                }
            else:
                prediction = {
                    param.name: inverse_transform(_prediction[param.name], param)
                    for param in self._raw_params
                    if not param.is_constant
                }

            request = _TuningRequest(context, predict_data, prediction, metadata)
        else:
            # Algorithms which do not support sequence of rewards
            # and for which a request for the given context was found
            request = deepcopy(request)
            request.id = generate_request_id()

        self._requests_manager.append(request)

        # full_prediction contains the constant parameters as well
        full_prediction = {
            param.name: param.val if param.is_constant else request.prediction[param.name] for param in self._raw_params
        }
        return PredictResponse(full_prediction, request.id)

    @abc.abstractmethod
    def _predict(
        self,
        context: Optional[Context] = None,
        predict_data: Optional[PredictDataType] = None,
    ) -> _PredictResponse:
        pass

    def store_reward(
        self,
        request_id: RequestIdType,
        reward: RewardType,
        reward_data: Optional[RewardDataType] = None,
    ) -> None:
        """Store the `reward` and `reward_data` for the given `request_id`.

        This is useful in scenarios where you wish to call `predict` multiple times
        before calling `set_reward`. By storing the reward, you can access the list
        of tuning requests and the stored reward using `get_tuning_requests`, aggregate
        the reward and finally call `set_reward`.

        For algorithms that expect a sequence of rewards, you need not aggregate the reward.
        In this case, all stored rewards will be passed to the algorithm.

        Raises:
            If the `request_id` does not exists, a `ValueError` is raised.
        """
        self._requests_manager.store_reward(request_id, reward, reward_data)

    def get_tuning_requests(self, context_id: Optional[ContextIdType] = None) -> List[TuningRequest]:
        """Return tuning requests for the given `context_id`.

        If `context_id` is `None`, return all tuning requests.
        """
        if context_id is None:
            return [
                TuningRequest(id=req.id, context=deepcopy(req.context), reward=req.reward)
                for req in self._requests_manager._requests
            ]

        return [
            TuningRequest(id=req.id, context=deepcopy(req.context), reward=req.reward)
            for req in self._requests_manager._requests
            if req.context.id == context_id
        ]

    def set_reward(
        self,
        reward: Union[RewardType, None] = None,
        context_id: ContextIdType = NullContextId,
    ) -> None:
        """Provide reward to the algorithm so that it can learn from it.

        Typically, `reward` will be a finite value (generally between 0.0 and 1.0) which is
        computed from the rewards for the tuning requests for this iteration.

        A `None` reward is allowed only for those algorithms which support a sequence of rewards.
        In this case, all tuning requests for this iteration, with the corresponding rewards,
        are used by the algorithm to learn. The algorithm may decide to not use all of them
        (for e.g., it can skip requests without a reward).

        Once the reward is set, the tuning requests for the given context id are cleared.

        Important:
            In the single finite reward scenario, only the recent most tuning request is passed
            to the algorithm along with the given reward. This means that data from older
            tuning requests (such as `context.data`, `predict_data` and `reward_data`) will not
            be used.
        """
        if reward is not None:
            if not self.Meta.supports_single_reward:
                raise ValueError(f"{self.__class__.__name__} does not support a single reward")

            tuning_request = deepcopy(self._requests_manager.get_request_by_context_id(context_id))
            if tuning_request is None:
                raise ValueError(f"No tuning request found for context_id {context_id!r}")

            tuning_request.id = NullRequestId
            tuning_request.reward = reward
            # TODO Make sure predict_data and reward_data are the same for all requests
            self._set_reward(tuning_request)
        else:
            if not self.Meta.supports_sequence_of_rewards:
                raise ValueError("reward=None is only supported by algorithms which accept a sequence of rewards")

            tuning_requests = self._requests_manager.get_requests_by_context_id(context_id)
            if len(tuning_requests) == 0:
                raise ValueError(f"No tuning requests found for context_id {context_id!r}")

            if not any(req.reward for req in tuning_requests):
                raise ValueError("At least one tuning request should have a finite reward")

            self._set_reward_for_sequence(tuning_requests)

        self._requests_manager.delete_requests_by_context_id(context_id)

    def _set_reward(self, tuning_request: _TuningRequest) -> None:
        raise NotImplementedError()

    def _set_reward_for_sequence(self, tuning_requests: Sequence[_TuningRequest]) -> None:
        raise NotImplementedError()

    def get_current_prediction(self, context_id: ContextIdType = NullContextId) -> Union[PredictResponse, None]:
        """Get the current prediction for the given `context_id`.

        If `set_reward` was called with the `context_id`
        """
        request = self._requests_manager.get_request_by_context_id(context_id)
        return None if request is None else PredictResponse(request.prediction, request.id)

    def dump(
        self,
        file: Union[str, Path, BinaryIO, io.BytesIO],
        compress: Union[int, Tuple[str, int]] = ("zlib", 3),
        **kwargs,
    ) -> None:
        """Save the serialized representation of the tuning instance to a file or stream.

        Args:
            file: The file path or I/O object where the object will be written to.
            compress: The compress argument for :py:func:`joblib.dump`.
            **kwargs: Additional arguments to the :py:func:`joblib.dump` function.
        """
        if isinstance(file, (BinaryIO, io.BytesIO)):
            joblib.dump(self, file, compress=compress, **kwargs)  # type: ignore
        else:
            with open(file, "wb") as f:
                joblib.dump(self, f, compress=compress, **kwargs)  # type: ignore

    def dumps(self, **kwargs) -> bytes:
        """Return the serialized representation of the tuning instance.

        Args:
            **kwargs: Additional arguments to the :py:meth:`dump` method.
        """
        f = io.BytesIO()
        self.dump(f, **kwargs)
        return f.getvalue()

    @classmethod
    def load(cls: Type[Self], file: Union[str, Path, BinaryIO, io.BytesIO], **kwargs) -> Self:
        """Load the serialized tuning instance from the given file or stream.

        Args:
            file: The file path or I/O object from where the object will be loaded.
            **kwargs: Additional arguments to the :py:func:`joblib.load` function.
        """
        if isinstance(file, (BinaryIO, io.BytesIO)):
            obj: cls = joblib.load(file, **kwargs)
        else:
            with open(file, "rb") as f:
                obj: cls = joblib.load(f, **kwargs)

        return obj

    @classmethod
    def loads(cls: Type[Self], buffer: bytes, **kwargs) -> Self:
        """Load the serialized tuning instance from the given buffer.

        Args:
            buffer: The serialized representation of the algorithm object.
            **kwargs: Additional arguments to the :py:meth:`load` method.
        """
        f = io.BytesIO(buffer)
        return cls.load(f, **kwargs)

    @property
    @abc.abstractmethod
    def iteration(self) -> int:
        """Return the current tuning iteration number."""

    def __str__(self):
        return f"{self.__class__.__name__}([{len(self.params)} parameters])"


class TuningRequestsManager:
    def __init__(self):
        self._requests: List[_TuningRequest] = []
        # self.__predictions: Dict[ContextIdType, PredictResponse] = {}
        # TODO Use __predictions to store the prediction per context only once, instead of in each request

    def append(self, request: _TuningRequest) -> None:
        self._requests.append(request)

    def get_request_by_id(self, tuning_request_id: RequestIdType) -> Union[_TuningRequest, None]:
        for req in self._requests:
            if req.id == tuning_request_id:
                return req

        return None

    def get_request_by_context_id(self, context_id: ContextIdType, newest: bool = True) -> Union[_TuningRequest, None]:
        requests = reversed(self._requests) if newest else self._requests
        for req in requests:
            if req.context.id == context_id:
                return req

        return None

    def get_requests_by_context_id(self, context_id: ContextIdType) -> List[_TuningRequest]:
        return [req for req in self._requests if req.context.id == context_id]

    def store_reward(
        self,
        request_id: RequestIdType,
        reward: RewardType,
        reward_data: Union[RewardDataType, None],
    ) -> None:
        request = self.get_request_by_id(request_id)
        if request is None:
            raise ValueError(f"No request for id {request_id}")

        request.reward = reward
        request.reward_data = reward_data

    def delete_request_by_id(self, tuning_request_id: RequestIdType) -> None:
        self._requests = [req for req in self._requests if req.id != tuning_request_id]

    def delete_requests_by_context_id(self, context_id: ContextIdType) -> None:
        self._requests = [req for req in self._requests if req.context.id != context_id]
