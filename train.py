import os

import keyboard
import torch
from torch import nn

from helper.dataset import LineaDataset
from helper.model import ANN
from threading import Event
from torch.utils.data import DataLoader
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass
import time


@dataclass
class Summarizer:
    writer: SummaryWriter
    train_loss_record_interval = 1
    __train_loss_step = 0
    __test_loss_step = 0
    __train_loss_loop_i = 1

    def add_graph(self, model, data_input):
        self.writer.add_graph(model, data_input)

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
    learning_rate = 1e-2
    batch_size = 200
    epochs = 500
    device = 'cuda'
    optimizer_func = torch.optim.Adam
    loss_fn = nn.CrossEntropyLoss()
    log_dir = os.path.join(os.getcwd(), 'runs')
    train_test_split_ratio = 0.8

    def train(self, model, dataset: torch.utils.data.Dataset, stop_event: Event):
        model = model.to(self.device)
        train_data_size = round(len(dataset) * self.train_test_split_ratio)
        test_data_size = len(dataset) - train_data_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        summarizer = Summarizer(SummaryWriter(os.path.join(self.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))))

        optimizer = self.optimizer_func(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            time.sleep(0.1)  # leave a bit time for keyboard event loop to react
            if stop_event.is_set():
                break
            for X, y in train_loader:
                pred = model(X)
                loss = self.loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                summarizer.add_train_loss(loss)
            with torch.no_grad():
                test_loss_sum = 0
                for X, y in DataLoader(test_dataset):
                    test_loss_sum += self.loss_fn(model(X), y)
                summarizer.add_test_loss(test_loss_sum / len(test_dataset))

    def start_tensor_board(self):
        """
        This board will contain following figures:
        1. Network structure
        2. Train Loss
        3. Test Loss
        4. PR curve
        5. sampled image classification
        :return:
        """
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir])
        url = tb.launch()
        print(f'TensorBoard starting at {url}')


def main():
    print('press q to stop training')
    keyboard.add_hotkey('q', lambda: stop_event.set())

    trainer = Trainer()
    trainer.start_tensor_board()

    dataset = LineaDataset(os.path.join(os.getcwd(), 'data'))
    model = ANN()
    stop_event = Event()

    trainer.train(model, dataset, stop_event)
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
