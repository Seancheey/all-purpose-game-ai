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
from rich.progress import Progress, TextColumn, TimeElapsedColumn


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
    timestamp: float = 0


@dataclass
class Recorder:
    save_dir: str
    max_fps: int = 30
    screen_res: Tuple[int] = img_size
    recording_keys: Set[str] = field(default_factory=lambda: keys.copy())
    exit_key: str = 'q'

    def record(self):
        stop_event = Event()
        with ThreadPoolExecutor(3) as pool:
            screen_future = pool.submit(self.__record_screen, stop_event)
            keyboard_future = pool.submit(self.__record_keyboard, stop_event)
            pool.submit(self.__listen_to_stop_event, stop_event).result()

            screen_data = screen_future.result()
            keyboard_data = keyboard_future.result()

        folder_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        os.mkdir(os.path.join(self.save_dir, folder_name))
        dataset = self.__to_training_data(keyboard_data, screen_data)
        self.__save_np_keys(dataset, folder_name)
        self.__save_np_screens(dataset, folder_name)
        self.__save_avi_video(dataset, folder_name)

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
        @rate_limited(1, 1 / self.max_fps)
        def capture():
            # noinspection PyTypeChecker
            return np.array(sct.grab(bounding_box))[::down_sample_factor, ::down_sample_factor]

        last_timestamp = datetime.now().timestamp()
        screens = []

        with Progress(
                TextColumn("Video Recorder Stats:"),
                TimeElapsedColumn(),
                TextColumn("[progress.description]{task.fields[fps]}")
        ) as progress:
            task_id = progress.add_task('recording', fps=0)
            while not stop_event.is_set():
                img = capture()
                timestamp = datetime.now().timestamp()
                screens.append(ScreenEvent(img, timestamp))
                progress.update(task_id, fps=f'{round(1 / (timestamp - last_timestamp), 1)} frame per sec')
                last_timestamp = timestamp

        return screens

    def __to_training_data(self, key_sequence: List[KeyEvent], screen_sequence: List[ScreenEvent]):
        ki, si = 0, 0
        cur_keys = set()
        data_out = []
        while si < len(screen_sequence):
            key_event = key_sequence[ki] if ki < len(key_sequence) else None
            screen_event = screen_sequence[si]
            if key_event is not None and key_event.timestamp < screen_event.timestamp:
                if key_event.down:
                    cur_keys.add(key_event.key_code)
                else:
                    cur_keys.remove(key_event.key_code)
                ki += 1
            else:
                data_out.append(DatasetItem(
                    screen=screen_event.screen,
                    key_codes=list(cur_keys),
                    timestamp=screen_event.timestamp
                ))
                si += 1
        return data_out

    def __save_np_screens(self, dataset: List[DatasetItem], folder: str):
        np.save(os.path.join(self.save_dir, folder, 'screen'),
                np.array(list(map(lambda x: x.screen, dataset))))

    def __save_np_keys(self, dataset: List[DatasetItem], folder: str):
        np.save(os.path.join(self.save_dir, folder, 'keys'),
                np.array(list(map(lambda x: x.key_codes, dataset)), dtype=object))

    def __save_avi_video(self, dataset: List[DatasetItem], folder: str):
        avg_fps = len(dataset) / (dataset[-1].timestamp - dataset[0].timestamp)
        print('average fps = ', avg_fps)
        video_writer = cv2.VideoWriter(os.path.join(self.save_dir, folder, f'video.avi'),
                                       cv2.VideoWriter_fourcc(*"XVID"), avg_fps, self.screen_res)
        for item in dataset:
            video_writer.write(cv2.cvtColor(item.screen, cv2.COLOR_BGR2RGB))
        video_writer.release()


def main():
    data_dir = os.path.join(os.getcwd(), 'data')
    recorder = Recorder(save_dir=data_dir)
    recorder.record()


if __name__ == "__main__":
    main()
