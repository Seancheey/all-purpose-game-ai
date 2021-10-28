# data format configs
import numpy as np
import torch

img_size = (192, 108)
recording_keys = ['w', 'a', 's', 'd']
np_keys_filename = 'keys.npy'
np_screens_filename = 'screens.npy'
avi_video_filename = 'video.avi'

key_map = np.array([['a', '', 'd'],
                    ['w', '', 's']])


def keys_to_directions(keys):
    x = 1
    y = 1
    for key in keys:
        if key == 'w':
            y -= 1
        elif key == 's':
            y += 1
        elif key == 'a':
            x -= 1
        elif key == 'd':
            x += 1
    out = np.zeros((2, 3), dtype=bool)
    out[0, x] = True
    out[1, y] = True
    return out


def directions_to_keys(directions):
    max_indices = torch.argmax(directions, dim=1)
    return [key_map[row][ind] for row, ind in enumerate(max_indices) if
            key_map[row][ind] != '']
