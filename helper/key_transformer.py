from dataclasses import dataclass, field
from typing import List, Iterable, Union

import numpy as np
import torch


@dataclass()
class KeyTransformer:
    recording_keys: List[str]
    __ordinal_multiplier: np.ndarray = field(init=False)

    def __post_init__(self):
        self.__ordinal_multiplier = np.fromiter((2 ** i for i in range(len(self.recording_keys))), dtype=np.int8)

    def keys_to_directions(self, keys: List[str]) -> np.ndarray:
        return np.fromiter(map(lambda k: k in keys, self.recording_keys), dtype=float)

    def directions_to_keys(self, directions: Iterable[Union[float, bool]]) -> List[str]:
        return [self.recording_keys[i] for i, press in enumerate(directions) if press]

    def directions_to_ordinal(self, direction: np.ndarray) -> int:
        return sum(np.multiply(direction, self.__ordinal_multiplier))

    def ordinal_to_directions(self, order: int) -> np.ndarray:
        return np.fromiter((1 if (order % (2 ** (i + 1))) >= 2 ** i else 0 for i in range(len(self.recording_keys))),
                           float)

    def pred_to_keys(self, predictions: List[torch.Tensor]) -> List[str]:
        return self.directions_to_keys(map(lambda x: x > 0.5, predictions))
