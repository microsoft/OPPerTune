"""The Slates algorithm."""

import io
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import joblib
import numpy as np
from typing_extensions import Self, TypedDict, TypeVar, override

from oppertune.core.types import Context, PredictResponse
from oppertune.core.values import Categorical, Integer, Real

from ..base import Algorithm, _PredictResponse, _TuningRequest

__all__ = (
    "Slates",
    "SlatesContext",
)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class SlatesContext(TypedDict):
    context: str


class Slates(Algorithm[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        requires_untransformed_parameters = False
        supports_context = True
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        vw_args: Optional[Dict[str, Any]] = None,
        n_discretized_values: int = 20,
        random_seed: Optional[int] = None,
    ):
        """Initialize the Slates algorithm object.

        Args:
            parameters: The parameters to tune.
            vw_args: Arguments for the VowpalWabbit Workspace.
            n_discretized_values: For a continuous variable, the number of values to discretize into
            (including both the bounds). This must be at least 2.
            random_seed: The seed used by VowpalWabbit's Workspace.
        """
        super().__init__(parameters, random_seed=random_seed)

        if not (n_discretized_values >= 2):
            raise ValueError("n_discretized_values must be at least 2")

        if vw_args is None:
            vw_args = {
                "arg_str": "--slates --epsilon 0.1 --power_t 0",
                "quiet": True,
                "random_seed": self._random_seed,
            }

        # Lazy import since vowpalwabbit wheels are not available for Python versions >=3.11.
        # This allows using other algorithms on Python versions >=3.11.
        import vowpalwabbit

        self.vw = vowpalwabbit.Workspace(**vw_args)
        self.slots: Dict[str, Sequence[Union[str, int, float]]] = {}
        self.index_prob: Dict[str, Tuple[int, float]] = {}

        for param in self.params:
            if isinstance(param, Categorical):
                values = param.categories
            elif isinstance(param, Integer):
                values = tuple(range(param.min, param.max + param.step, param.step))
            else:
                assert isinstance(param, Real)
                values = tuple(np.linspace(param.min, param.max, num=n_discretized_values, dtype=float).tolist())

            self.slots[param.name] = values

    @override
    def predict(
        self, context: Optional[Context[SlatesContext]] = None, predict_data: None = None
    ) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(
        self,
        context: Optional[Context[SlatesContext]] = None,
        predict_data: None = None,
    ) -> _PredictResponse[_ParameterValueType]:
        _context = context.data["context"] if context and context.data else None

        example = Slates.to_vw_slates_format(self.slots, _context)
        example = self.vw.parse(example)
        predictions: List[List[Tuple[int, float]]] = self.vw.predict(example)
        self.vw.finish_example(example)

        prediction = {}
        self.index_prob = {}
        for i, (slot, values) in enumerate(self.slots.items()):
            best_pred = predictions[i][0]
            prediction[slot] = values[best_pred[0]]
            self.index_prob[slot] = best_pred

        return _PredictResponse(prediction)

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        reward = tuning_request.reward
        if reward is None:
            raise ValueError(f"reward cannot be None for {self.__class__.__name__}")

        reward = -reward
        context = tuning_request.context.data["context"] if tuning_request.context.data else None

        example = Slates.to_vw_slates_format(self.slots, context, reward, self.index_prob)
        example = self.vw.parse(example)
        self.vw.learn(example)
        self.vw.finish_example(example)

        self._iteration += 1

    def finish(self) -> None:
        self.vw.finish()

    @staticmethod
    def to_vw_slates_format(
        slots: Dict[str, Sequence[Union[float, int, str]]],
        context: Union[str, None],
        reward: Optional[Union[int, float]] = None,
        index_probs: Optional[Dict[str, Tuple[int, float]]] = None,
    ) -> List[str]:
        with_label = False
        if (reward is not None) or index_probs:
            assert (reward is not None) and index_probs
            with_label = True

        text = [f"slates shared {reward} |Context {context}"] if with_label else [f"slates shared |Context {context}"]

        for i, values in enumerate(slots.values()):
            for value in values:
                text.append(f"slates action {i} |Action {value}")

        if with_label:
            for slot in slots:
                idx, prob = index_probs[slot]
                text.append(f"slates slot {idx}:{prob} |Slot {slot}")
        else:
            for slot in slots:
                text.append(f"slates slot |Slot {slot}")

        return text

    @override
    def dump(
        self,
        file: Union[str, Path, BinaryIO, io.BytesIO],
        compress: Union[int, Tuple[str, int]] = ("zlib", 3),
        **kwargs,
    ) -> None:
        with tempfile.NamedTemporaryFile("wb", delete=False) as path_workspace:
            pass

        self.vw.save(path_workspace.name)
        with open(path_workspace.name, "rb") as f:
            workspace_bytes = f.read()

        os.remove(path_workspace.name)
        state = {
            "vw": workspace_bytes,
            "slots": self.slots,
            "index_prob": self.index_prob,
        }

        if isinstance(file, (BinaryIO, io.BytesIO)):
            joblib.dump(state, file, compress=compress, **kwargs)  # type: ignore
        else:
            with open(file, "wb") as f:
                joblib.dump(state, f, compress=compress, **kwargs)  # type: ignore

    @override
    @classmethod
    def load(cls: Type[Self], file: Union[str, Path, BinaryIO, io.BytesIO], **kwargs) -> Self:
        if isinstance(file, (BinaryIO, io.BytesIO)):
            state: dict = joblib.load(file, **kwargs)
        else:
            with open(file, "rb") as f:
                state: dict = joblib.load(f, **kwargs)

        obj = cls(parameters=[])
        with tempfile.NamedTemporaryFile("wb", delete=False) as path_workspace:
            path_workspace.write(state["vw"])

        import vowpalwabbit  # To allow Python version >=3.11

        obj.vw = vowpalwabbit.Workspace(f"-i {path_workspace.name}", quiet=True)
        os.remove(path_workspace.name)
        state.pop("vw")

        for key, value in state.items():
            if not hasattr(obj, key):
                raise ValueError(f"Could not find attribute {key} while loading")
            setattr(obj, key, value)

        return obj

    @property
    @override
    def iteration(self) -> int:
        return self._iteration
