from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List

import numpy as np
from cv2 import cv2
from ratelimit import sleep_and_retry, rate_limited
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from screeninfo import screeninfo

from helper.data_format import img_size
from mss import mss


@dataclass
class ScreenStreamer:
    max_fps: int = 30
    screen_res: Tuple[int] = img_size

    def stream(self, stop_event) -> List[np.array]:
        """
        starts streaming images, which stops only when stop_event is triggered.
        :return: stream of images as np array, with shape (screen_res[0], screen_res[1], 3)
        """
        monitors = screeninfo.get_monitors()
        assert len(monitors) > 0, OSError('No Monitor Detected.')
        monitor = monitors[0]

        down_sample_factor = min(monitor.width // self.screen_res[1], monitor.height // self.screen_res[1])
        width_diff = (monitor.width - down_sample_factor * self.screen_res[0])
        height_diff = (monitor.height - down_sample_factor * self.screen_res[1])
        w_start, w_end = width_diff // 2, monitor.width - width_diff // 2
        h_start, h_end = height_diff // 2, monitor.height - height_diff // 2

        bounding_box = {
            'left': monitor.x,
            'top': monitor.y,
            'width': monitor.width,
            'height': monitor.height
        }

        screen_grabber = mss()

        @sleep_and_retry
        @rate_limited(1, 1 / self.max_fps)
        def capture():
            # noinspection PyTypeChecker
            raw_img = np.array(screen_grabber.grab(bounding_box))
            # capture screen, then down-sample + trim the sides to meet specified resolution. output color is RGBA.
            return raw_img[h_start:h_end:down_sample_factor, w_start:w_end:down_sample_factor]

        last_timestamp = datetime.now().timestamp()

        with Progress(
                TextColumn("Video Recorder Stats:"),
                TimeElapsedColumn(),
                TextColumn("[progress.description]{task.fields[fps]}")
        ) as progress:
            task_id = progress.add_task('recording', fps=0)
            while not stop_event.is_set():
                img = capture()
                yield cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                timestamp = datetime.now().timestamp()
                progress.update(task_id, fps=f'{round(1 / (timestamp - last_timestamp), 1)} frame per sec')
                last_timestamp = timestamp