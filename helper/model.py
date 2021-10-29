import torch
from torch import nn

from helper.data_format import key_map


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(20, 20), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(10, 10), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(5, 5), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.ann_stack = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, key_map.shape[0]),
            nn.Sigmoid()
        )
        self.prediction = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = torch.div(x, 255)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ann_stack(x)
        x = torch.reshape(x, (-1,) + key_map.shape)
        x = self.prediction(x)
        return x


if __name__ == '__main__':
    model = ANN().to('cuda')
    test = torch.rand(100, 108, 192, 3, device='cuda')
    print(model(test))
