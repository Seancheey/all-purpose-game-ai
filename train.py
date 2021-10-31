import os

import keyboard
import torch
from torch import nn

from helper.dataset import LineaDataset
from helper.model import PlayModel
from threading import Event
from torch.utils.data import DataLoader
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass, field
import time


@dataclass
class Summarizer:
    """
    Writes various summary information about model to tensor board log, so it could be read be tensor board.
    """
    writer: SummaryWriter
    train_loss_record_interval = 1
    __train_loss_step = 0
    __test_loss_step = 0
    __train_loss_loop_i = 1

    def add_graph(self, model, sample_data_input):
        self.writer.add_graph(model, sample_data_input)

    def add_train_loss(self, loss):
        if self.__train_loss_loop_i < self.train_loss_record_interval:
            self.__train_loss_loop_i += 1
        else:
            self.__train_loss_loop_i = 1
            self.writer.add_scalar('train loss', loss, self.__train_loss_step)
        self.__train_loss_step += 1

    def add_test_loss(self, loss):
        self.writer.add_scalar('test loss', loss, self.__test_loss_step)
        self.__test_loss_step += 1


@dataclass
class Trainer:
    learning_rate = 0.001
    batch_size = 200
    epochs = 500
    device = 'cuda'
    optimizer_func = torch.optim.Adam
    loss_fn = nn.BCELoss()
    log_dir = os.path.join(os.getcwd(), 'runs')
    train_test_split_ratio = 0.8
    stop_train_key = 'ctrl+q'

    __stop_event: Event = field(default_factory=lambda: Event())

    def train(self, model, dataset: torch.utils.data.Dataset, run_name: str = "train"):
        model = model.to(self.device)
        # split data to train&test
        train_data_size = round(len(dataset) * self.train_test_split_ratio)
        test_data_size = len(dataset) - train_data_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # tensorboard summary writer
        summarizer = Summarizer(
            SummaryWriter(os.path.join(self.log_dir, run_name + "-" + datetime.now().strftime('%Y%m%d-%H-%M-%S'))))
        summarizer.add_graph(model, next(iter(train_loader))[0])

        # stop event listening setup
        keyboard.add_hotkey(self.stop_train_key, self.stop_training)
        self.__stop_event.clear()
        print(f'Training started. Press {self.stop_train_key} to stop.')

        optimizer = self.optimizer_func(model.parameters(), lr=self.learning_rate)
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

                summarizer.add_train_loss(loss)

            model.eval()
            with torch.no_grad():
                test_loss_sum = 0
                for X, y in DataLoader(test_dataset):
                    test_loss_sum += self.loss_fn(model(X), y)
                summarizer.add_test_loss(test_loss_sum / len(test_dataset))

    def start_tensor_board(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir])
        url = tb.launch()
        print(f'TensorBoard starting at {url}')

    def stop_training(self):
        print("Stopping Training. Please allow time for the trainer to finish last epoch.")
        self.__stop_event.set()


def main():
    trainer = Trainer()
    trainer.start_tensor_board()

    dataset = LineaDataset(os.path.join(os.getcwd(), 'data'))
    model = PlayModel()

    trainer.train(model, dataset, 'multi-class-train')
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
