import abc
import io
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Dict, NamedTuple, Optional, Union


class PredictResponse(NamedTuple):
    parameters: Dict[str, Any]
    metadata: Union[Dict[str, Any], None] = None


class AlgorithmBackend(abc.ABC):
    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> PredictResponse:
        pass

    @abc.abstractmethod
    def set_reward(self, reward: Union[float, int], *args, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        pass

    def dump(self, file: Union[str, Path, BinaryIO, io.BytesIO], **pickle_kwargs):
        if isinstance(file, (BinaryIO, io.BytesIO)):
            pickle.dump(self, file, **pickle_kwargs)
        else:
            with open(file, "wb") as f:
                pickle.dump(self, f, **pickle_kwargs)

    def dumps(self, **pickle_kwargs):
        f = io.BytesIO()
        self.dump(f, **pickle_kwargs)
        return f.getvalue()

    @classmethod
    def load(cls, file: Union[str, Path, BinaryIO, io.BytesIO], **pickle_kwargs):
        if isinstance(file, (BinaryIO, io.BytesIO)):
            obj: cls = pickle.load(file, **pickle_kwargs)
        else:
            with open(file, "rb") as f:
                obj: cls = pickle.load(f, **pickle_kwargs)

        return obj

    @classmethod
    def loads(cls, buffer: bytes, **pickle_kwargs):
        f = io.BytesIO(buffer)
        return cls.load(f, **pickle_kwargs)
