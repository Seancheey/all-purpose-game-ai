from typing import List, Iterable, Union

import numpy as np
import torch
from torchvision.transforms import ToTensor
from helper.data_format import recording_keys

image_to_tensor = ToTensor()


def keys_to_directions(keys: List[str]) -> np.ndarray:
    return np.fromiter(map(lambda k: k in keys, recording_keys), dtype=float)


def directions_to_keys(directions: Iterable[Union[float, bool]]) -> List[str]:
    return [recording_keys[i] for i, press in enumerate(directions) if press]


__ordinal_multiplier = np.fromiter((2 ** i for i in range(len(recording_keys))), dtype=np.int8)


def directions_to_ordinal(direction: np.ndarray) -> int:
    """
    >>> directions_to_ordinal(np.array([1,0,0,0]))
    1
    >>> directions_to_ordinal(np.array([1,0,1,0]))
    5
    """
    return sum(np.multiply(direction, __ordinal_multiplier))


def ordinal_to_directions(order: int) -> np.ndarray:
    """
    >>> ordinal_to_directions(7)
    array([1., 1., 1., 0.])
    >>> ordinal_to_directions(11)
    array([1., 1., 0., 1.])
    """
    return np.fromiter((1 if (order % (2 ** (i + 1))) >= 2 ** i else 0 for i in range(len(recording_keys))), float)


def pred_to_keys(predictions: List[torch.Tensor]) -> List[str]:
    return directions_to_keys(map(lambda x: x > 0.5, predictions))
