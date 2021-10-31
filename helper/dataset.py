import os

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from helper.data_format import np_keys_filename, np_screens_filename, img_size
from helper.transforms import image_to_tensor, KeyTransformer


class LineaDataset(Dataset):
    def __init__(self, data_dir, key_transformer: KeyTransformer, seed=0):
        self.key_transformer = key_transformer
        record_dirs = list(map(lambda path: os.path.join(data_dir, path), os.listdir(data_dir)))
        keys_list = [np.load(os.path.join(record_dir, np_keys_filename)) for record_dir in record_dirs]
        screens_list = [np.load(os.path.join(record_dir, np_screens_filename)) for record_dir in record_dirs]
        self.screens = np.concatenate(screens_list)
        self.keys = np.concatenate(keys_list)
        assert len(self.screens) == len(self.keys), "screen size and key size is not matching"

        self.__over_sample(seed)

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index) -> T_co:
        return image_to_tensor(self.screens[index]).cuda(), torch.tensor(self.keys[index], device='cuda',
                                                                         dtype=torch.float)

    def __over_sample(self, seed):
        # sampler doesn't support multi-label input, so transform keys to ordinal encoding first
        self.keys = np.fromiter((self.key_transformer.directions_to_ordinal(d) for d in self.keys), np.int8)
        # sampler doesn't support multi-dimension input, so reshape image first
        self.screens = self.screens.reshape((len(self.screens), -1))

        sm = RandomOverSampler(random_state=seed)
        self.screens, self.keys = sm.fit_resample(self.screens, self.keys)

        self.screens = self.screens.reshape((-1,) + img_size.np_shape())
        self.keys = np.stack([self.key_transformer.ordinal_to_directions(o) for o in self.keys])
