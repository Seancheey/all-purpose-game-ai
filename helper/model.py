import torch
from torch import nn
from helper.data_format import img_size, recording_keys


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.ann_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, len(recording_keys)),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = torch.div(x, 255)
        x = self.flatten(x)
        return self.ann_stack(x)


if __name__ == '__main__':
    model = ANN().to('cuda')
    test = torch.rand(1, 192, 108, 3, device='cuda')
    model(test)
