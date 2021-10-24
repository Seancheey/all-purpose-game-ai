from concurrent.futures import ThreadPoolExecutor
import os.path
from cv2 import cv2
import numpy as np
from datetime import datetime
from threading import Event
import keyboard
from mss import mss

from dataclasses import dataclass, field
from screeninfo import screeninfo
from typing import List, Tuple, Set
from rich import print
from ratelimit import rate_limited, sleep_and_retry


@dataclass
class KeyEvent:
    key_code: str
    timestamp: float
    press_down: bool


@dataclass
class Recorder:
    save_dir: str
    fps: int = 60
    screen_res: Tuple[int] = (192, 108)
    recording_keys: Set[str] = field(default_factory=lambda: {'w', 'a', 's', 'd'})
    exit_key: str = 'q'

    def record(self):
        start_time = datetime.now().timestamp()
        stop_event = Event()
        with ThreadPoolExecutor(4) as pool:
            screen_future = pool.submit(self.__record_screen, stop_event)
            keyboard_future = pool.submit(self.__record_keyboard, stop_event)
            pool.submit(self.__listen_to_stop_event, stop_event).result()

            screen_future.result()
            keyboard_future.result()

    def __listen_to_stop_event(self, stop_event):
        keyboard.wait(self.exit_key)
        stop_event.set()

    def __record_keyboard(self, stop_event: Event) -> List[KeyEvent]:
        key_sequence = []

        def handle_press(event: keyboard.KeyboardEvent):
            if event.name in self.recording_keys:
                key_sequence.append(KeyEvent(event.name, datetime.now().timestamp(), True))
                print(event.name)

        def handle_release(event: keyboard.KeyboardEvent):
            if event.name in self.recording_keys:
                key_sequence.append(KeyEvent(event.name, datetime.now().timestamp(), False))

        keyboard.on_press(handle_press)
        keyboard.on_release(handle_release)

        stop_event.wait()
        return key_sequence

    def __record_screen(self, stop_event: Event):
        sct = mss()
        monitors = screeninfo.get_monitors()
        assert len(monitors) > 0, OSError('No Monitor Detected.')
        monitor = monitors[0]
        bounding_box = {
            'left': monitor.x,
            'top': monitor.y,
            'width': monitor.width,
            'height': monitor.height
        }
        down_sample_factor = monitor.width // 192
        assert monitor.height // 108 == down_sample_factor, 'monitor is not 16:9 screen'

        @sleep_and_retry
        @rate_limited(1, 1 / self.fps)
        def capture():
            sct_img = np.array(sct.grab(bounding_box))[::down_sample_factor, ::down_sample_factor]
            cv2.imshow('screen', sct_img)

        while not stop_event.is_set():
            capture()
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def __save_records(self, key_sequence, recording):
        np.save(os.path.join(self.save_dir, f"{self.start_time}"), np.array(key_sequence))


def main():
    data_dir = os.path.join(os.getcwd(), 'data')
    recorder = Recorder(save_dir=data_dir)
    recorder.record()


if __name__ == "__main__":
    main()
