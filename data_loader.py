import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from data_format import np_keys_filename, np_screens_filename
import numpy as np


class LineaDataset(Dataset):
    def __init__(self, data_dir):
        record_dirs = list(map(lambda path: os.path.join(data_dir, path), os.listdir(data_dir)))
        keys_list = [np.load(os.path.join(record_dir, np_keys_filename)) for record_dir in record_dirs]
        screens_list = [np.load(os.path.join(record_dir, np_screens_filename)) for record_dir in record_dirs]
        self.screens = np.concatenate(screens_list)
        self.keys = np.concatenate(keys_list)
        assert len(self.screens) == len(self.keys), "screen size and key size is not matching"

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index) -> T_co:
        return self.screens[index], self.keys[index]


if __name__ == '__main__':
    dataset = LineaDataset(os.path.join(os.getcwd(), 'data'))
    print(dataset[0])
