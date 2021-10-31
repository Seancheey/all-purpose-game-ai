import os.path
from dataclasses import dataclass

import numpy as np
from cv2 import cv2
from ratelimit import rate_limited, sleep_and_retry

from helper.data_format import np_keys_filename, np_screens_filename
from helper.transforms import directions_to_keys


@dataclass
class DataVisualizer:
    fps: int

    def visualize(self, record_dir: str):
        keys = np.load(os.path.join(record_dir, np_keys_filename))
        screens = np.load(os.path.join(record_dir, np_screens_filename))

        @sleep_and_retry
        @rate_limited(1, 1 / self.fps)
        def show_img(img: np.ndarray):
            cv2.imshow('video', img)

        for direction, screen in zip(keys, screens):
            screen = cv2.resize(screen, (960, 540))
            keys = "keys:" + ",".join(directions_to_keys(direction))
            cv2.putText(screen, keys, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0, 255))
            show_img(screen)
            cv2.waitKey(1)


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'data')
    record_dirs = [os.path.join(data_dir, record_dir) for record_dir in os.listdir(data_dir)]
    for i, record_dir in enumerate(record_dirs):
        print(f'data {i}: {record_dir}')
        DataVisualizer(fps=20).visualize(record_dir)
