import torch
from torch import nn
from torchsummary import summary

from helper.data_format import key_map


class PlayModel(nn.Module):
    def __init__(self):
        super(PlayModel, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(8, 8), stride=(2, 2), padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.ann_stack = nn.Sequential(
            nn.Linear(4224, 1024),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.3),
            nn.Sigmoid(),
            nn.Linear(1024, key_map.shape[0]),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ann_stack(x)
        return x


if __name__ == '__main__':
    model = PlayModel().to('cuda')
    summary(model, (3, 108, 192), batch_size=64)
    test = torch.rand(100, 3, 108, 192, device='cuda')
    print(model(test))
