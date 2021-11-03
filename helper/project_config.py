from dataclasses import dataclass
from typing import List, Callable

import numpy as np
import torch
import torchvision

from helper.data_format import ImageFormat
from helper.data_visualizer import DataVisualizer
from helper.dataset import VideoKeyboardDataset
from helper.game_ai_applier import GameAiApplier
from helper.key_transformer import KeyTransformer
from helper.recorder import RepeatingRecorder, Recorder
from helper.screen_streamer import ScreenStreamer
from helper.train import Trainer
from helper.window_region import WindowRegion


@dataclass
class ProjectConfig:
    """
    ProjectConfig that resolves all dependencies for recording, training, applying AI.
    I should have used a DI framework though :(... but it could be messy in python... idk...
    """
    recording_keys: List[str]
    img_format: ImageFormat
    data_dir: str
    train_log_dir: str
    model_class: Callable[[], torch.nn.Module]
    model_path: str
    record_window_region_func: Callable[[], WindowRegion]
    start_record_key: str
    stop_record_key: str
    save_record_key: str
    start_apply_key: str
    stop_apply_key: str
    stop_train_key: str
    train_name: str
    data_visualize_fps: int
    max_record_fps: int
    recording_image_filter_func: Callable[[np.ndarray], np.ndarray] = None
    screen_to_tensor_func: Callable[[np.ndarray], torch.Tensor] = torchvision.transforms.ToTensor()

    def provide_recorder(self) -> RepeatingRecorder:
        return RepeatingRecorder(
            recorder=Recorder(
                save_dir=self.data_dir,
                recording_keys=set(self.recording_keys),
                screen_streamer=self._provide_screen_streamer(),
                key_transformer=self._provide_key_transformer()
            ),
            start_key=self.start_record_key,
            stop_key=self.stop_record_key,
            save_key=self.save_record_key
        )

    def provide_trainer(self) -> Trainer:
        return Trainer(
            train_name=self.train_name,
            dataset=self._provide_dataset(),
            model=self._provide_raw_model(),
            model_save_path=self.model_path,
            train_log_dir=self.train_log_dir,
        )

    def provide_ai_applier(self) -> GameAiApplier:
        return GameAiApplier(
            screen_streamer=self._provide_screen_streamer(),
            start_apply_hotkey=self.start_apply_key,
            stop_apply_hotkey=self.stop_apply_key,
            trained_model=self._provide_trained_model(),
            key_transformer=self._provide_key_transformer(),
            screen_to_tensor_func=self.screen_to_tensor_func
        )

    def provide_data_visualizer(self) -> DataVisualizer:
        return DataVisualizer(
            data_dir=self.data_dir,
            key_transformer=self._provide_key_transformer(),
            fps=self.data_visualize_fps
        )

    def _provide_screen_streamer(self) -> ScreenStreamer:
        return ScreenStreamer(
            max_fps=self.max_record_fps,
            record_window_region=self.record_window_region_func(),
            output_img_format=self.img_format,
            recording_img_transform_func=self.recording_image_filter_func
        )

    def _provide_dataset(self) -> VideoKeyboardDataset:
        return VideoKeyboardDataset(self.data_dir, self._provide_key_transformer(), self.screen_to_tensor_func)

    def _provide_raw_model(self) -> torch.nn.Module:
        return self.model_class()

    def _provide_trained_model(self) -> torch.nn.Module:
        model = self._provide_raw_model()
        model.load_state_dict(torch.load(self.model_path))
        return model

    def _provide_key_transformer(self) -> KeyTransformer:
        return KeyTransformer(self.recording_keys)
