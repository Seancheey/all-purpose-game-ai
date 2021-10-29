import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from helper.data_format import np_keys_filename, np_screens_filename, img_size
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from torchvision.transforms import ToTensor


class LineaDataset(Dataset):
    def __init__(self, data_dir, seed=0):
        self.transform = ToTensor()
        record_dirs = list(map(lambda path: os.path.join(data_dir, path), os.listdir(data_dir)))
        keys_list = [np.load(os.path.join(record_dir, np_keys_filename)) for record_dir in record_dirs]
        screens_list = [np.load(os.path.join(record_dir, np_screens_filename)) for record_dir in record_dirs]
        self.screens = np.concatenate(screens_list)
        self.keys = np.concatenate(keys_list)
        assert len(self.screens) == len(self.keys), "screen size and key size is not matching"

        # bootstrap to resample minority data
        sm = RandomOverSampler(random_state=seed)
        self.screens = self.screens.reshape((len(self.screens), -1))
        self.screens, self.keys = sm.fit_resample(self.screens, self.keys)
        self.screens = self.screens.reshape((-1, img_size[1], img_size[0], 3))

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index) -> T_co:
        return self.transform(self.screens[index]).cuda(), torch.tensor(self.keys[index], device='cuda',
                                                                        dtype=torch.float)


if __name__ == '__main__':
    dataset = LineaDataset(os.path.join(os.getcwd(), '../data'))
    print(dataset[0])
