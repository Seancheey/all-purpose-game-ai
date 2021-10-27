import keyboard
import torch
from helper.screen_streamer import ScreenStreamer
from threading import Event
from rich.progress import Progress
from helper.data_format import recording_keys
from helper.model import ANN

model = ANN()
model.load_state_dict(torch.load('model.pth'))

stop_event = Event()

keyboard.add_hotkey('q', lambda: stop_event.set())

streamer = ScreenStreamer()

with Progress('Prediction: keys: {task.fields[keys]}') as progress:
    task = progress.add_task('', keys=[])
    cur_keys = set()
    for img in streamer.stream(stop_event):
        pred = model(torch.tensor([img]))[0]
        keys = {recording_keys[i]: bool(val > 0.5) for i, val in enumerate(pred)}
        for key, down in keys.items():
            if down and key not in cur_keys:
                keyboard.press(key)
            elif not down and key in cur_keys:
                keyboard.release(key)
        progress.update(task, keys=keys)

    for key in cur_keys:
        keyboard.release(key)
