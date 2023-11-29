import io
import os
import pickle
import tempfile
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import vowpalwabbit
from typing_extensions import TypedDict

from ...values import CategoricalValue, ContinuousValue, DiscreteValue
from ..base import AlgorithmBackend, PredictResponse

__all__ = ("Slates",)


class Metadata(TypedDict):
    context: Optional[str]


class Slates(AlgorithmBackend):
    def __init__(
        self,
        parameters: Iterable[Union[ContinuousValue, DiscreteValue, CategoricalValue]],
        vw_args: Optional[dict] = None,
        n_discretized_values: int = 20,
    ):
        """
        n_discretized_values: For a continuous variable, the number of values to discretize into
        (including both the bounds). Ideally, this should be >= 2.
        """
        if vw_args is None:
            vw_args = dict(
                arg_str="--slates --epsilon 0.1 --power_t 0",
                quiet=True,
            )

        self.vw = vowpalwabbit.Workspace(**vw_args)
        self.slots: Dict[str, Sequence[Union[str, float, int]]] = {}
        self.index_prob: Dict[str, Tuple[float, int]] = {}

        for param in parameters:
            if isinstance(param, CategoricalValue):
                values = param.categories
            elif isinstance(param, DiscreteValue):
                step_size = round(param.step_size or 1)
                lb, ub = round(param.lb), round(param.ub)
                values = tuple(range(lb, ub + step_size, step_size))
            elif isinstance(param, ContinuousValue):
                values = tuple(np.linspace(param.lb, param.ub, num=n_discretized_values, dtype=float).tolist())
            else:
                raise TypeError(f"[{param.name}] Unknown dtype={param.dtype}")

            self.slots[param.name] = values

    def predict(self, context: Optional[str] = None):
        example = Slates.to_vw_slates_format(self.slots, context)
        example = self.vw.parse(example)
        predictions = self.vw.predict(example)
        self.vw.finish_example(example)

        parameters = {}
        self.index_prob = {}
        for i, (slot, values) in enumerate(self.slots.items()):
            best_pred = predictions[i][0]
            parameters[slot] = values[best_pred[0]]
            self.index_prob[slot] = best_pred

        return PredictResponse(parameters=parameters, metadata=Metadata(context=context))

    def set_reward(self, reward: Union[float, int], metadata: Metadata):
        reward = -reward
        example = Slates.to_vw_slates_format(self.slots, metadata["context"], reward, self.index_prob)
        example = self.vw.parse(example)
        self.vw.learn(example)
        self.vw.finish_example(example)

    def finish(self):
        self.vw.finish()

    @staticmethod
    def to_vw_slates_format(
        slots: Dict[str, Sequence[Union[float, int, str]]],
        context: Union[str, None],
        reward: Optional[Union[float, int]] = None,
        index_probs: Optional[Dict[str, Tuple[float, int]]] = None,
    ) -> List[str]:
        with_label = False
        if (reward is not None) or index_probs:
            assert (reward is not None) and index_probs
            with_label = True

        if with_label:
            text = [f"slates shared {reward} |Context {context}"]
        else:
            text = [f"slates shared |Context {context}"]

        for i, values in enumerate(slots.values()):
            for value in values:
                text.append(f"slates action {i} |Action {value}")

        if with_label:
            for slot in slots.keys():
                idx, prob = index_probs[slot]
                text.append(f"slates slot {idx}:{prob} |Slot {slot}")
        else:
            for slot in slots.keys():
                text.append(f"slates slot |Slot {slot}")

        return text

    def dump(self, file: Union[str, Path, BinaryIO, io.BytesIO], **pickle_kwargs):
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
            pickle.dump(state, file, **pickle_kwargs)
        else:
            with open(file, "wb") as f:
                pickle.dump(state, f, **pickle_kwargs)

    @classmethod
    def load(cls, file: Union[str, Path, BinaryIO, io.BytesIO], **pickle_kwargs):
        if isinstance(file, (BinaryIO, io.BytesIO)):
            state: dict = pickle.load(file, **pickle_kwargs)
        else:
            with open(file, "rb") as f:
                state: dict = pickle.load(f, **pickle_kwargs)

        obj = cls(parameters=[])
        with tempfile.NamedTemporaryFile("wb", delete=False) as path_workspace:
            path_workspace.write(state["vw"])

        obj.vw = vowpalwabbit.Workspace(f"-i {path_workspace.name}", quiet=True)
        os.remove(path_workspace.name)
        state.pop("vw")

        for key, value in state.items():
            assert hasattr(obj, key), f"Could not find attribute {key} while loading"
            setattr(obj, key, value)

        return obj

    def __str__(self):
        return "Slates"
