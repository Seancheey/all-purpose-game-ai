# data format configs
from typing import List, Union

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as functional

img_size = (192, 108)
recording_keys = ['w', 'a', 's', 'd']
np_keys_filename = 'keys.npy'
np_screens_filename = 'screens.npy'
avi_video_filename = 'video.avi'

key_map = np.array(['', 'w', 'wd', 'd', 'sd', 's', 'sa', 'a', 'wa'])
key_to_ind = {key: i for i, key in enumerate(key_map)}

key_encoder = OneHotEncoder(sparse=False)
key_encoder.fit(key_map.reshape(-1, 1))


def normalize_keys(keys: List[str]) -> str:
    """
    Converts list of directions keys to normalized keys by:
    1. remove opposite direction keys: e.g. wad -> w
    2. swap order so vertical key always comes first: e.g. aw -> wa
    """
    x = 0
    y = 0
    for key in keys:
        if key == 'w':
            y -= 1
        elif key == 's':
            y += 1
        elif key == 'a':
            x -= 1
        elif key == 'd':
            x += 1
    vertical = ['', 's', 'w']
    horizontal = ['', 'd', 'a']
    return vertical[y] + horizontal[x]


def keys_to_directions(keys: Union[List[str], str]) -> np.ndarray:
    return key_encoder.transform(np.array(normalize_keys(keys)).reshape(-1, 1))[0]


def directions_to_keys(directions: torch.Tensor) -> str:
    if type(directions) != torch.Tensor:
        directions = torch.from_numpy(np.array(directions, dtype=int))
    return key_encoder.inverse_transform(
        functional.one_hot(torch.argmax(directions, keepdim=True), num_classes=key_map.shape[0]))[0]


if __name__ == '__main__':
    print(keys_to_directions('w'))
    print(keys_to_directions('wa'))
