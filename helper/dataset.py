import os
from collections import Callable

import numpy as np
import torch
import torchvision.transforms
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from helper.data_format import np_keys_filename, np_screens_filename
from helper.key_transformer import KeyTransformer


def over_sample_to_balance_labels(screen_dataset, key_dataset, key_transformer, seed):
    # sampler doesn't support multi-label input, so transform keys to ordinal encoding first
    key_dataset = np.fromiter((key_transformer.directions_to_ordinal(d) for d in key_dataset), np.int8)
    # sampler doesn't support multi-dimension input, so reshape image first
    screen_shape = screen_dataset.shape[1:]
    screen_dataset = screen_dataset.reshape((len(screen_dataset), -1))

    sm = RandomOverSampler(random_state=seed)
    screen_dataset, key_dataset = sm.fit_resample(screen_dataset, key_dataset)

    screen_dataset = screen_dataset.reshape((-1,) + screen_shape)
    key_dataset = np.stack([key_transformer.ordinal_to_directions(o) for o in key_dataset])
    return screen_dataset, key_dataset


class VideoKeyboardDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            key_transformer: KeyTransformer,
            screen_to_tensor_func: Callable[np.ndarray, torch.Tensor] = torchvision.transforms.ToTensor(),
            seed=0
    ):
        self.key_transformer = key_transformer
        self.screen_transformer = screen_to_tensor_func
        record_dirs = list(map(lambda path: os.path.join(data_dir, path), os.listdir(data_dir)))
        keys_list = [np.load(os.path.join(record_dir, np_keys_filename)) for record_dir in record_dirs]
        screens_list = [np.load(os.path.join(record_dir, np_screens_filename)) for record_dir in record_dirs]
        self.screens = np.concatenate(screens_list)
        self.keys = np.concatenate(keys_list)
        assert len(self.screens) == len(self.keys), "screen size and key size is not matching"

        self.screens, self.keys = over_sample_to_balance_labels(self.screens, self.keys, self.key_transformer, seed)

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index) -> T_co:
        return self.screen_transformer(self.screens[index]).cuda(), torch.tensor(self.keys[index], device='cuda',
                                                                                 dtype=torch.float)
