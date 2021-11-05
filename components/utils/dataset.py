import os
from collections import Callable, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision.transforms
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from components.utils.image_format import np_keys_filename, np_screens_filename
from components.utils.key_transformer import KeyTransformer


@dataclass
class VideoKeyboardDataset(Dataset):
    """
    Dataset class that loads N folders of screen and keyboard data, then convert them to tensors for pytorch trainer.
    """
    data_dir: str
    key_transformer: KeyTransformer
    device: str
    screen_to_tensor_func: Callable[[np.ndarray], torch.Tensor] = torchvision.transforms.ToTensor()
    screen_augmentation_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    load_to_device_at_init: bool = False
    oversample_to_balance_labels: bool = False
    seed: int = 0

    def __post_init__(self):
        record_dirs = list(map(lambda path: os.path.join(self.data_dir, path), os.listdir(self.data_dir)))
        print("loading screens")
        screens_list = list(tqdm((np.load(os.path.join(record_dir, np_screens_filename)) for record_dir in record_dirs),
                                 total=len(record_dirs)))
        self.screens = np.concatenate(screens_list)
        print("loading keys")
        keys_list = list(tqdm((np.load(os.path.join(record_dir, np_keys_filename)) for record_dir in record_dirs),
                              total=len(record_dirs)))
        self.keys = np.concatenate(keys_list, dtype=float)
        assert len(self.screens) == len(self.keys), "screen size and key size is not matching"

        if self.oversample_to_balance_labels:
            self.screens, self.keys = self.__over_sample_to_balance_labels(self.screens, self.keys)
        else:
            print('data distribution:')
            self.summarize_keys_distribution(self.keys)

        print("convert to tensors")
        self.screens = list(
            tqdm((self.screen_to_tensor_func(screen) for screen in self.screens), total=len(self.screens)))
        self.screens = torch.stack(self.screens)
        self.keys = torch.tensor(self.keys, dtype=torch.float)

        if self.load_to_device_at_init:
            print(f"loading data to {self.device}")
            self.screens = self.screens.to(device=self.device)
            self.keys = self.keys.to(device=self.device)

    def __over_sample_to_balance_labels(self, screen_dataset, key_dataset):
        print("before resample:")
        self.summarize_keys_distribution(key_dataset)
        print(f"over sample unbalanced dataset. original size: {len(screen_dataset)} ", end='')
        # sampler doesn't support multi-label input, so transform keys to ordinal encoding first
        key_dataset = np.fromiter((self.key_transformer.directions_to_ordinal(d) for d in key_dataset), np.int8)
        # sampler doesn't support multi-dimension input, so reshape image first
        screen_shape = screen_dataset.shape[1:]
        screen_dataset = screen_dataset.reshape((len(screen_dataset), -1))

        sm = RandomOverSampler(random_state=self.seed)
        screen_dataset, key_dataset = sm.fit_resample(screen_dataset, key_dataset)

        screen_dataset = screen_dataset.reshape((-1,) + screen_shape)
        key_dataset = np.stack([self.key_transformer.ordinal_to_directions(o) for o in key_dataset])
        print(f"new size: {len(screen_dataset)}")
        print("after resample:")
        self.summarize_keys_distribution(key_dataset)
        return screen_dataset, key_dataset

    def summarize_keys_distribution(self, keys: np.ndarray):
        counts = defaultdict(int)
        for directions in keys:
            counts['+'.join(self.key_transformer.directions_to_keys(directions))] += 1
        print('\n'.join(f'\'{key:>3}\': {count}' for key, count in counts.items()))

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index) -> T_co:
        screen = self.screens[index]
        key = self.keys[index]
        if self.screen_augmentation_func:
            screen = self.screen_augmentation_func(screen)
        if self.load_to_device_at_init:
            return screen, key
        else:
            return screen.to(device=self.device), key.to(device=self.device)
