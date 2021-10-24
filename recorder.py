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
from ratelimit import rate_limited, sleep_and_retry
from data_format import keys, img_size


@dataclass
class KeyEvent:
    key_code: str
    timestamp: float
    down: bool


@dataclass
class ScreenEvent:
    screen: np.array
    timestamp: float


@dataclass
class DatasetItem:
    screen: np.array
    key_codes: List[str]


@dataclass
class Recorder:
    save_dir: str
    fps: int = 60
    screen_res: Tuple[int] = img_size
    recording_keys: Set[str] = field(default_factory=lambda: keys.copy())
    exit_key: str = 'q'

    def record(self):
        start_time = datetime.now()
        stop_event = Event()
        with ThreadPoolExecutor(3) as pool:
            screen_future = pool.submit(self.__record_screen, stop_event)
            keyboard_future = pool.submit(self.__record_keyboard, stop_event)
            pool.submit(self.__listen_to_stop_event, stop_event).result()

            screen_data = screen_future.result()
            keyboard_data = keyboard_future.result()
        self.__save_records(keyboard_data, screen_data, start_time.strftime('%Y%m%d-%H%M%S'))

    def __listen_to_stop_event(self, stop_event):
        keyboard.wait(self.exit_key)
        stop_event.set()

    def __record_keyboard(self, stop_event: Event) -> List[KeyEvent]:
        key_sequence = []

        def handle_event(event: keyboard.KeyboardEvent):
            key_sequence.append(
                KeyEvent(event.name, datetime.now().timestamp(), event.event_type == 'down'))

        for key in self.recording_keys:
            keyboard.hook_key(key, handle_event)

        stop_event.wait()
        return key_sequence

    def __record_screen(self, stop_event: Event) -> List[ScreenEvent]:
        sct = mss()
        monitors = screeninfo.get_monitors()
        assert len(monitors) > 0, OSError('No Monitor Detected.')
        monitor = monitors[0]
        down_sample_factor = monitor.width // self.screen_res[0]
        assert monitor.height // self.screen_res[1] == down_sample_factor, f'screen mismatches ratio {self.screen_res}'

        bounding_box = {
            'left': monitor.x,
            'top': monitor.y,
            'width': monitor.width,
            'height': monitor.height
        }

        @sleep_and_retry
        @rate_limited(1, 1 / self.fps)
        def capture():
            # noinspection PyTypeChecker
            return np.array(sct.grab(bounding_box))[::down_sample_factor, ::down_sample_factor]

        screens = []

        while not stop_event.is_set():
            img = capture()
            cv2.imshow('screen', img)
            screens.append(ScreenEvent(img, datetime.now().timestamp()))
            cv2.waitKey(1)

        cv2.destroyAllWindows()
        return screens

    def __save_records(self, key_sequence: List[KeyEvent], screen_sequence: List[ScreenEvent], filename_suffix):
        ki, si = 0, 0
        cur_keys = set()
        data_out = []
        while ki < len(key_sequence) and si < len(screen_sequence):
            key_event = key_sequence[ki]
            screen_event = screen_sequence[si]
            if key_event.timestamp < screen_event.timestamp:
                if key_event.down:
                    cur_keys.add(key_event.key_code)
                else:
                    cur_keys.remove(key_event.key_code)
                ki += 1
            else:
                data_out.append(DatasetItem(
                    screen=screen_event.screen,
                    key_codes=list(cur_keys)
                ))
                si += 1
        np.save(os.path.join(self.save_dir, f'screen-{filename_suffix}'),
                np.array(list(map(lambda x: x.screen, data_out))))
        np.save(os.path.join(self.save_dir, f'key-{filename_suffix}'),
                np.array(list(map(lambda x: x.key_codes, data_out)), dtype=object))


def main():
    data_dir = os.path.join(os.getcwd(), 'data')
    recorder = Recorder(save_dir=data_dir)
    recorder.record()


if __name__ == "__main__":
    main()
