import os.path

import numpy as np
from screen_recorder_sdk import screen_recorder
from datetime import datetime
import keyboard

from dataclasses import dataclass


@dataclass
class KeyEvent:
    key_code: str
    timestamp: float
    press_down: bool


class Recorder:
    recording_keys = {'w', 'a', 's', 'd'}

    def __init__(self, save_dir, fps=30, bitrate=8000000):
        self.save_dir = save_dir
        self.fps = fps
        self.bitrate = bitrate
        self.key_sequence = []
        self.start_time = 0

    def start_record(self):
        screen_recorder.init_resources(screen_recorder.RecorderParams())
        self.start_time = datetime.now().timestamp()
        filename = os.path.join(self.save_dir, f"{self.start_time}.mp4")
        self.key_sequence = []
        screen_recorder.start_video_recording(filename, self.fps, self.bitrate, True)

        def handle_press(event: keyboard.KeyboardEvent):
            if event.name in Recorder.recording_keys:
                self.key_sequence.append(KeyEvent(event.name, datetime.now().timestamp(), True))

        def handle_release(event: keyboard.KeyboardEvent):
            if event.name in Recorder.recording_keys:
                self.key_sequence.append(KeyEvent(event.name, datetime.now().timestamp(), False))

        keyboard.on_press(handle_press)
        keyboard.on_release(handle_release)

    def end_record(self):
        screen_recorder.stop_video_recording()
        screen_recorder.free_resources()
        np.save(os.path.join(self.save_dir, f"{self.start_time}"), np.array(self.key_sequence))


def main():
    recorder = Recorder(os.getcwd())

    keyboard.wait('s')
    recorder.start_record()

    keyboard.wait('q')
    recorder.end_record()


if __name__ == "__main__":
    main()
