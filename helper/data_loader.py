import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from helper.data_format import np_keys_filename, np_screens_filename, img_size, key_map
import numpy as np


class LineaDataset(Dataset):
    def __init__(self, data_dir, train=True, batch_size=64, device='cpu'):
        record_dirs = list(map(lambda path: os.path.join(data_dir, path), os.listdir(data_dir)))
        keys_list = [np.load(os.path.join(record_dir, np_keys_filename)) for record_dir in record_dirs]
        screens_list = [np.load(os.path.join(record_dir, np_screens_filename)) for record_dir in record_dirs]
        self.screens = np.concatenate(screens_list)
        self.keys = np.concatenate(keys_list)
        assert len(self.screens) == len(self.keys), "screen size and key size is not matching"
        train_dataset_cut_index = round(0.75 * len(self.screens))
        if train:
            self.screens = self.screens[:train_dataset_cut_index]
            self.keys = self.keys[:train_dataset_cut_index]
        else:
            self.screens = self.screens[train_dataset_cut_index:]
            self.keys = self.keys[train_dataset_cut_index:]

        batched_upper_index = len(self.screens) - len(self.screens) % batch_size
        self.screens = np.reshape(self.screens[:batched_upper_index], (-1, batch_size, img_size[0], img_size[1], 3))
        self.keys = np.reshape(self.keys[:batched_upper_index], (-1, batch_size) + key_map.shape)

        self.screens = torch.tensor(self.screens, device=device)
        self.keys = torch.tensor(self.keys, device=device, dtype=torch.float)

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index) -> T_co:
        return self.screens[index], self.keys[index]


if __name__ == '__main__':
    dataset = LineaDataset(os.path.join(os.getcwd(), '../data'))
    print(dataset[0])
