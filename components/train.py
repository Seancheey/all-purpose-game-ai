import time
from dataclasses import dataclass, field
from threading import Event
from typing import Sized, Type

import keyboard
import torch
from tensorboard import program
from torch import nn
from torch.utils.data import DataLoader

from components.utils.tensor_board_summarizer import Summarizer


@dataclass
class Trainer:
    model: nn.Module
    dataset: torch.utils.data.Dataset
    train_name: str
    model_save_path: str
    train_log_dir: str
    tensor_board_summarizer: Summarizer
    learning_rate = 0.001
    batch_size = 200
    epochs = 500
    device = 'cuda'
    optimizer_func: Type[torch.optim.Optimizer] = torch.optim.Adam
    loss_fn = nn.BCELoss()
    train_test_split_ratio = 0.8
    stop_train_key = 'ctrl+q'
    auto_save_best: bool = True

    __stop_event: Event = field(default_factory=lambda: Event())

    def train_existing_and_save(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.train_and_save()

    def train_and_save(self):
        self.start_tensor_board()
        self.train()
        if not self.auto_save_best:
            self.save_model()

    def train(self):
        model = self.model.to(self.device)

        train_dataset, test_dataset = self.__split_dataset_to_train_and_test()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # tensorboard summary writer
        self.tensor_board_summarizer.add_graph(model, next(iter(train_loader))[0])

        # stop event listening setup
        keyboard.add_hotkey(self.stop_train_key, self.stop_training)
        self.__stop_event.clear()
        print(f'Training started. Press {self.stop_train_key} to stop.')

        optimizer = self.optimizer_func(model.parameters(), lr=self.learning_rate)

        cur_min_loss = 23 * 10 ** 8
        for epoch in range(self.epochs):
            time.sleep(0.1)  # leave a bit time for keyboard event loop to react
            if self.__stop_event.is_set():
                break

            model.train(mode=True)
            for X, y in train_loader:
                pred = model(X)
                loss = self.loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.tensor_board_summarizer.add_train_loss(loss)

            model.eval()
            with torch.no_grad():
                test_loss_sum = 0
                for X, y in DataLoader(test_dataset):
                    test_loss_sum += self.loss_fn(model(X), y)
                mean_loss = test_loss_sum / len(test_dataset)
                self.tensor_board_summarizer.add_test_loss(mean_loss)
                if self.auto_save_best and mean_loss < cur_min_loss:
                    self.save_model()

    def start_tensor_board(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.train_log_dir])
        url = tb.launch()
        print(f'TensorBoard starting at {url}')

    def stop_training(self):
        print("Stopping Training. Please allow time for the trainer to finish last epoch.")
        self.__stop_event.set()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)

    def __split_dataset_to_train_and_test(self):
        # split data to train&test
        if isinstance(self.dataset, Sized):
            data_len = len(self.dataset)
        else:
            data_len = sum((1 for _ in self.dataset))
        train_data_size = round(data_len * self.train_test_split_ratio)
        test_data_size = data_len - train_data_size
        return torch.utils.data.random_split(self.dataset, [train_data_size, test_data_size])
