import os

import keyboard
import torch
from torch import nn

from helper.dataset import LineaDataset
from helper.model import ANN
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from threading import Event
from torch.utils.data import DataLoader

learning_rate = 1e-3
batch_size = 200
epochs = 500
device = 'cuda'

if __name__ == '__main__':
    print('press q to stop training')
    data_dir = os.path.join(os.getcwd(), 'data')
    train_dataset = DataLoader(LineaDataset(data_dir), batch_size=batch_size, shuffle=True)

    model = ANN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    stop_event = Event()
    keyboard.add_hotkey('q', lambda: stop_event.set())

    with Progress(
            TimeElapsedColumn(),
            TextColumn('Loss: {task.fields[loss]}'),
            TextColumn('Epoch: {task.fields[epoch]}')
    ) as progress:
        task = progress.add_task('', loss='None', epoch=0)
        for epoch in range(epochs):
            if stop_event.is_set():
                break
            for X, y in train_dataset:
                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.update(task, loss=loss, epoch=epoch)
    torch.save(model.state_dict(), 'model.pth')
