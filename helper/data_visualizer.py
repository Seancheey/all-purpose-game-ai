import os.path
from dataclasses import dataclass

import numpy as np
from cv2 import cv2
from ratelimit import rate_limited, sleep_and_retry

from helper.data_format import np_keys_filename, np_screens_filename
from helper.transforms import KeyTransformer


@dataclass
class DataVisualizer:
    data_dir: str
    key_transformer: KeyTransformer
    fps: int

    def visualize_all(self):
        record_dirs = [os.path.join(self.data_dir, record_dir) for record_dir in os.listdir(self.data_dir)]
        for i, record_dir in enumerate(record_dirs):
            print(f'data {i}: {record_dir}')
            self.visualize_single(record_dir)

    def visualize_single(self, record_dir):
        keys = np.load(os.path.join(record_dir, np_keys_filename))
        screens = np.load(os.path.join(record_dir, np_screens_filename))

        @sleep_and_retry
        @rate_limited(1, 1 / self.fps)
        def show_img(img: np.ndarray):
            cv2.imshow('video', img)

        for direction, screen in zip(keys, screens):
            screen = cv2.resize(screen, (960, 540))
            keys = "keys:" + ",".join(self.key_transformer.directions_to_keys(direction))
            cv2.putText(screen, keys, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0, 255))
            show_img(screen)
            cv2.waitKey(1)
