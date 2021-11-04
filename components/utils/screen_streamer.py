from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Callable

from cv2.cv2 import cvtColor, COLOR_RGBA2RGB
from mss import mss
from numpy import array, ndarray
from ratelimit import sleep_and_retry, rate_limited
from rich.progress import Progress

from components.utils.image_format import ImageFormat
from components.utils.window_region import WindowRegion


@dataclass
class ScreenStreamer:
    output_img_format: ImageFormat
    max_fps: int = 30
    recording_img_transform_func: Callable[[ndarray], ndarray] = None
    record_window_region: WindowRegion = field(default_factory=lambda: WindowRegion.from_first_monitor())

    def stream(self, stop_event, progress_bar: Progress = None) -> List[ndarray]:
        """
        starts streaming images, which stops only when stop_event is triggered.
        :return: stream of images as np array, with shape (screen_res[1], screen_res[0], 3)
        """
        down_sample_factor = min(self.record_window_region.width // self.output_img_format.width,
                                 self.record_window_region.height // self.output_img_format.height)
        width_diff = (self.record_window_region.width - down_sample_factor * self.output_img_format.width)
        height_diff = (self.record_window_region.height - down_sample_factor * self.output_img_format.height)
        w_start, w_end = width_diff // 2, width_diff // 2 + down_sample_factor * self.output_img_format.width
        h_start, h_end = height_diff // 2, height_diff // 2 + down_sample_factor * self.output_img_format.height

        screen_grabber = mss()
        bounding_box = self.record_window_region.to_mss_bounding_box()

        @sleep_and_retry
        @rate_limited(1, 1 / self.max_fps)
        def capture():
            # noinspection PyTypeChecker
            raw_img = array(screen_grabber.grab(bounding_box))
            # capture screen, then down-sample + trim the sides to meet specified resolution. output color is RGBA.
            zoomed_img = raw_img[h_start:h_end:down_sample_factor, w_start:w_end:down_sample_factor]
            if self.recording_img_transform_func:
                return self.recording_img_transform_func(zoomed_img)
            else:
                return zoomed_img

        last_timestamp = datetime.now().timestamp()

        task_id = progress_bar.add_task('recording', fps=0) if progress_bar else None
        while not stop_event.is_set():
            img = capture()
            yield cvtColor(img, COLOR_RGBA2RGB)
            if progress_bar is not None:
                timestamp = datetime.now().timestamp()
                progress_bar.update(task_id, fps=f'{round(1 / (timestamp - last_timestamp), 1)} frame per sec')
                last_timestamp = timestamp
