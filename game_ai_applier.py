import keyboard
import torch
from helper.screen_streamer import ScreenStreamer
from threading import Event, Thread
from rich.progress import Progress
from helper.transforms import directions_to_keys, image_to_tensor
from helper.model import PlayModel

model = PlayModel()
model.load_state_dict(torch.load('model.pth'))

stop_event = Event()

streamer = ScreenStreamer()


def start_apply_keyboard_events():
    with Progress('Prediction: keys: {task.fields[keys]}') as progress:
        task = progress.add_task('', keys=[])
        cur_keys = set()
        for img in streamer.stream(stop_event):
            img = image_to_tensor(img)
            pred = model(torch.reshape(img, (1,) + img.shape))[0]
            keys = set(list(str(directions_to_keys(pred)[0])))
            to_press = keys - cur_keys
            to_release = cur_keys - keys
            for key in to_press:
                keyboard.press(key)
            for key in to_release:
                keyboard.release(key)
            cur_keys = keys
            progress.update(task, keys=keys)

        for key in cur_keys:
            keyboard.release(key)


def stop():
    stop_event.set()


keyboard.add_hotkey('e', lambda: Thread(target=start_apply_keyboard_events).start())
keyboard.add_hotkey('space', stop)

stop_event.wait()
