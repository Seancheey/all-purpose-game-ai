import torch
from torch import nn
from torchsummary import summary

from helper.data_format import key_map


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.ann_stack = nn.Sequential(
            nn.Linear(62208, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, key_map.shape[0]),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = self.ann_stack(x)
        return x


if __name__ == '__main__':
    model = ANN().to('cuda')
    summary(model, (1, 3, 108, 192))
    test = torch.rand(100, 3, 108, 192, device='cuda')
    print(model(test))
