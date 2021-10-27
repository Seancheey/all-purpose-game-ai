import os

import torch
from torch import nn
from helper.data_loader import LineaDataset
from helper.model import ANN
from rich.progress import Progress, TextColumn, TimeElapsedColumn

learning_rate = 1e-3
batch_size = 200
epochs = 500
device = 'cuda'

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'data')
    train_dataset = LineaDataset(data_dir, train=True, batch_size=batch_size, device=device)
    test_dataset = LineaDataset(data_dir, train=False, batch_size=batch_size, device=device)

    model = ANN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with Progress(
            TimeElapsedColumn(),
            TextColumn('Loss: {task.fields[loss]}'),
            TextColumn('Epoch: {task.fields[epoch]}')
    ) as progress:
        task = progress.add_task('', loss='None', epoch=0)
        for epoch in range(epochs):
            for X, y in train_dataset:
                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.update(task, loss=loss, epoch=epoch)
    torch.save(model.state_dict(), 'model.pth')
