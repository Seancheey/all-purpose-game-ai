import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from helper.data_format import np_keys_filename, np_screens_filename, img_size, key_map
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


class LineaDataset(Dataset):
    def __init__(self, data_dir, train=True, batch_size=64, device='cpu', seed=0):
        self.batch_size = batch_size
        record_dirs = list(map(lambda path: os.path.join(data_dir, path), os.listdir(data_dir)))
        keys_list = [np.load(os.path.join(record_dir, np_keys_filename)) for record_dir in record_dirs]
        screens_list = [np.load(os.path.join(record_dir, np_screens_filename)) for record_dir in record_dirs]
        self.screens = np.concatenate(screens_list)
        self.keys = np.concatenate(keys_list)
        assert len(self.screens) == len(self.keys), "screen size and key size is not matching"

        # split to train/test dataset
        if train:
            self.screens, _, self.keys, _ = train_test_split(self.screens, self.keys, random_state=seed)
        else:
            _, self.screens, _, self.keys = train_test_split(self.screens, self.keys, random_state=seed)

        # bootstrap to resample minority data
        sm = RandomOverSampler(random_state=seed)
        self.screens = self.screens.reshape((len(self.screens), -1))
        self.screens, self.keys = sm.fit_resample(self.screens, self.keys)
        self.screens = self.screens.reshape((-1, img_size[1], img_size[0], 3))

        # random shuffle
        rng = np.random.default_rng()
        rand_order = rng.permuted(np.arange(len(self.screens)))
        self.screens = self.screens[rand_order]
        self.keys = self.keys[rand_order]

        # group to batches
        self.screens = self.to_batch(self.screens)
        self.keys = self.to_batch(self.keys)

        # to pytorch tensors
        self.screens = torch.tensor(self.screens, device=device)
        self.keys = torch.tensor(self.keys, device=device, dtype=torch.float)

    def to_batch(self, array):
        batched_upper_index = len(array) - len(array) % self.batch_size
        return np.reshape(array[:batched_upper_index], (-1, self.batch_size) + array.shape[1:])

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index) -> T_co:
        return self.screens[index], self.keys[index]


if __name__ == '__main__':
    dataset = LineaDataset(os.path.join(os.getcwd(), '../data'))
    print(dataset[0])
