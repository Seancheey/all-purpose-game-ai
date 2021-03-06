from collections import Callable
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event

import keyboard
import numpy as np
import torch
from keyboard import press, release
from rich.progress import Progress
from torch import reshape

from components.utils.key_transformer import KeyTransformer
from components.utils.screen_streamer import ScreenStreamer


@dataclass
class GameAiApplier:
    trained_model: torch.nn.Module
    screen_streamer: ScreenStreamer
    key_transformer: KeyTransformer
    screen_to_tensor_func: Callable[np.ndarray, torch.Tensor]
    start_apply_hotkey: str
    stop_apply_hotkey: str
    __stop_event: Event = field(default_factory=lambda: Event())

    def start_apply_keyboard_events(self):
        self.__stop_event.clear()

        keyboard.add_hotkey(self.stop_apply_hotkey, self.stop)
        print(f'Press {self.start_apply_hotkey} to start applying AI.')
        keyboard.wait(self.start_apply_hotkey)
        print(f'Press {self.stop_apply_hotkey} to stop.')

        with Progress('Prediction: fps: {task.fields[fps]} keys: {task.fields[keys]}') as progress:
            with torch.no_grad():
                self.trained_model.eval()
                task = progress.add_task('', fps=None, keys=[])
                cur_keys = set()
                last_timestamp = datetime.now().timestamp()
                for img in self.screen_streamer.stream(self.__stop_event):
                    img = self.screen_to_tensor_func(img)
                    pred = self.trained_model(reshape(img, (1,) + img.shape))[0]
                    keys = set(self.key_transformer.pred_to_keys(pred))
                    to_press = keys - cur_keys
                    to_release = cur_keys - keys
                    for key in to_press:
                        press(key)
                    for key in to_release:
                        release(key)
                    cur_keys = keys
                    timestamp = datetime.now().timestamp()
                    fps = round(1 / (timestamp - last_timestamp), 1)
                    last_timestamp = timestamp
                    progress.update(task, fps=fps, keys=keys)

            for key in cur_keys:
                keyboard.release(key)

    def stop(self):
        self.__stop_event.set()
