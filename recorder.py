from concurrent.futures import ThreadPoolExecutor
from threading import Event
import os.path
from cv2 import cv2
import numpy as np
from datetime import datetime
import keyboard
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from helper.data_format import np_keys_filename, avi_video_filename, np_screens_filename, recording_keys, img_size, \
    to_key_array
import psutil
from helper.screen_streamer import ScreenStreamer


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
    recording_keys: Set[str] = field(default_factory=lambda: set(recording_keys))
    finish_record_key: str = 'space'
    discard_tail_sec: int = 3  # discard last N seconds of content, so that failing movement won't be learnt by model.

    def record(self):
        stop_event = Event()
        with ThreadPoolExecutor(3) as pool:
            screen_future = pool.submit(self.__record_screen, stop_event)
            keyboard_future = pool.submit(self.__record_keyboard, stop_event)
            pool.submit(self.__listen_to_finish_record_event, stop_event).result()

            screen_data = screen_future.result()
            keyboard_data = keyboard_future.result()

        dataset = self.__to_training_data(keyboard_data, screen_data)

        if len(dataset) == 0:
            print('skipping saving dataset due to empty content\n')
            return

        folder_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        os.mkdir(os.path.join(self.save_dir, folder_name))
        self.__save_np_keys(dataset, folder_name)
        self.__save_np_screens(dataset, folder_name)
        self.__save_avi_video(dataset, folder_name)
        print(f'saved data to {folder_name}\n')

    def __listen_to_finish_record_event(self, stop_event):
        keyboard.wait(self.finish_record_key)
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
        streamer = ScreenStreamer(max_fps=self.max_fps, screen_res=self.screen_res)
        screens = []
        for img in streamer.stream(stop_event):
            screens.append(ScreenEvent(img, datetime.now().timestamp()))
        return screens

    def __to_training_data(self, key_sequence: List[KeyEvent], screen_sequence: List[ScreenEvent]):
        ki, si = 0, 0
        cur_keys = set()
        data_out = []
        end_timestamp = screen_sequence[-1].timestamp - self.discard_tail_sec
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
                if screen_event.timestamp > end_timestamp:
                    break
                data_out.append(DatasetItem(
                    screen=screen_event.screen,
                    key_codes=list(cur_keys),
                    timestamp=screen_event.timestamp
                ))
                si += 1
        return data_out

    def __save_np_screens(self, dataset: List[DatasetItem], folder: str):
        np.save(os.path.join(self.save_dir, folder, np_screens_filename),
                np.array(list(map(lambda x: x.screen, dataset))))

    def __save_np_keys(self, dataset: List[DatasetItem], folder: str):
        np.save(os.path.join(self.save_dir, folder, np_keys_filename),
                np.array(list(map(lambda x: to_key_array(x.key_codes), dataset)), dtype=bool))

    def __save_avi_video(self, dataset: List[DatasetItem], folder: str):
        avg_fps = len(dataset) / (dataset[-1].timestamp - dataset[0].timestamp)
        print('average fps =', round(avg_fps, 2))
        video_writer = cv2.VideoWriter(os.path.join(self.save_dir, folder, avi_video_filename),
                                       cv2.VideoWriter_fourcc(*"XVID"), avg_fps, self.screen_res)
        for item in dataset:
            video_writer.write(item.screen)
        video_writer.release()


def main():
    keyboard.add_hotkey('q', lambda: psutil.Process(os.getpid()).terminate())
    print('start recording... (press "q" to exit, press "space" to save and start next recording)')
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    while True:
        recorder = Recorder(save_dir=data_dir)
        recorder.record()


if __name__ == "__main__":
    main()
