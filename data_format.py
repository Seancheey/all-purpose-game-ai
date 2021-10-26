import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np

img_size = (192, 108)
recording_keys = ['w', 'a', 's', 'd']
key_to_index = {key: ind for key, ind in enumerate(recording_keys)}
index_to_key = {ind: key for key, ind in enumerate(recording_keys)}


def to_key_array(key_list):
    return [key in key_list for key in recording_keys]


class LineaDataset(Dataset):
    def __init__(self, data_dir):
        a = os.listdir(data_dir)
        record_dirs = list(map(lambda path: os.path.join(data_dir, path), os.listdir(data_dir)))
        keys_list = [np.load(os.path.join(record_dir, 'keys.npy')) for record_dir in record_dirs]
        screens_list = [np.load(os.path.join(record_dir, 'screen.npy')) for record_dir in record_dirs]
        self.screens = np.concatenate(screens_list)
        self.keys = np.concatenate(keys_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index) -> T_co:
        pass

# dataset = LineaDataset(os.path.join(os.getcwd(), 'data'))
# print(dataset)
