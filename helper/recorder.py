import os.path
import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event
from typing import List, Set

import keyboard
import numpy as np
import psutil
from cv2 import cv2
from rich.progress import Progress, TextColumn, TimeElapsedColumn

from helper.data_format import np_keys_filename, avi_video_filename, np_screens_filename
from helper.screen_streamer import ScreenStreamer
from helper.transforms import KeyTransformer


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
    """
    Recorder that records both screen video and keyboard events at the same time.
    Upon calling stop_and_save(), key inputs and video will be saved to save_dir in npy format, along with an avi video.
    """
    save_dir: str
    recording_keys: Set[str]
    key_transformer: KeyTransformer
    screen_streamer: ScreenStreamer
    discard_tail_sec: float = 3  # discard last N seconds of content, so that failing movement won't be learnt by model.
    key_recording_delay_sec: float = -0.005  # record key events N sec earlier to compensate for delay
    __finish_record_event: Event = field(default_factory=lambda: Event())

    def record(self):
        """
        Make a blocking call to start recording.
        """
        self.__finish_record_event.clear()
        with ThreadPoolExecutor(3) as pool:
            screen_future = pool.submit(self.__record_screen)
            keyboard_future = pool.submit(self.__record_keyboard)
            pool.submit(self.__listen_to_finish_record_event).result()

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

    def stop_and_save(self):
        self.__finish_record_event.set()

    def __listen_to_finish_record_event(self):
        self.__finish_record_event.wait()

    def __record_keyboard(self) -> List[KeyEvent]:
        key_sequence = []

        def handle_event(event: keyboard.KeyboardEvent):
            key_sequence.append(
                KeyEvent(event.name, datetime.now().timestamp() + self.key_recording_delay_sec,
                         event.event_type == 'down'))

        for key in self.recording_keys:
            keyboard.hook_key(key, handle_event)

        self.__finish_record_event.wait()
        return key_sequence

    def __record_screen(self) -> List[ScreenEvent]:
        with Progress(
                TextColumn("Video Recorder Stats:"),
                TimeElapsedColumn(),
                TextColumn("[progress.description]{task.fields[fps]}")
        ) as progress:
            screens = []
            for img in self.screen_streamer.stream(self.__finish_record_event, progress):
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
                np.stack(list(map(lambda x: x.screen, dataset))))

    def __save_np_keys(self, dataset: List[DatasetItem], folder: str):
        np.save(os.path.join(self.save_dir, folder, np_keys_filename),
                np.stack(list(map(lambda x: self.key_transformer.keys_to_directions(x.key_codes), dataset))))

    def __save_avi_video(self, dataset: List[DatasetItem], folder: str):
        avg_fps = len(dataset) / (dataset[-1].timestamp - dataset[0].timestamp)
        print('average fps =', round(avg_fps, 2))
        video_writer = cv2.VideoWriter(os.path.join(self.save_dir, folder, avi_video_filename),
                                       cv2.VideoWriter_fourcc(*"XVID"), avg_fps,
                                       self.screen_streamer.output_img_format.resolution_shape())
        for item in dataset:
            video_writer.write(item.screen)
        video_writer.release()


@dataclass
class RepeatingRecorder:
    recorder: Recorder
    start_key: str
    stop_key: str
    save_key: str

    def start_recording(self):
        keyboard.add_hotkey(self.stop_key, RepeatingRecorder.terminate_everything)
        keyboard.add_hotkey(self.save_key, lambda: self.recorder.stop_and_save())
        print(f'press "{self.start_key}" to start recording.')
        keyboard.wait(self.start_key)
        print(f'start recording...')
        print(f'press "{self.stop_key}" to exit, press "{self.save_key}" to save and start next recording')
        pathlib.Path(self.recorder.save_dir).mkdir(parents=True, exist_ok=True)
        while True:
            self.recorder.record()

    @staticmethod
    def terminate_everything():
        psutil.Process(os.getpid()).terminate()
