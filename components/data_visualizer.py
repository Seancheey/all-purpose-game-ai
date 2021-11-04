import os.path
from dataclasses import dataclass

import numpy as np
from cv2 import cv2
from ratelimit import rate_limited, sleep_and_retry

from components.utils.image_format import np_keys_filename, np_screens_filename, ImageFormat
from components.utils.key_transformer import KeyTransformer


@dataclass
class DataVisualizer:
    data_dir: str
    key_transformer: KeyTransformer
    img_format: ImageFormat
    fps: int
    video_window_scale: int = 2
    key_top_left_pos_scale: float = 0.2

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

        native_resolution = self.img_format.resolution_shape()
        output_res = (native_resolution[0] * self.video_window_scale, native_resolution[1] * self.video_window_scale)
        text_pos = (
            round(output_res[0] * self.key_top_left_pos_scale), round(output_res[1] * self.key_top_left_pos_scale))
        for direction, screen in zip(keys, screens):
            screen = cv2.resize(screen, output_res)
            keys = "keys:" + ",".join(self.key_transformer.directions_to_keys(direction))
            cv2.putText(screen, keys, text_pos, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0, 255))
            show_img(screen)
            cv2.waitKey(1)
