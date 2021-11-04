import torch
from torch import nn


class SuperHexagonModel(nn.Module):

    @staticmethod
    def basic_fc(num_input, num_output):
        return nn.Sequential(
            nn.Linear(num_input, num_output),
            nn.BatchNorm1d(num_output),
            nn.GELU(),
            nn.Dropout(p=0.5),
        )

    def __init__(self):
        super(SuperHexagonModel, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(8, 8), stride=(2, 2), padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.ann_stack = nn.Sequential(
            SuperHexagonModel.basic_fc(1152, 1024),
            SuperHexagonModel.basic_fc(1024, 1024),
            SuperHexagonModel.basic_fc(1024, 1024)
        )
        self.output_stacks = nn.ModuleList([nn.Sequential(
            SuperHexagonModel.basic_fc(1024, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
            for _ in range(2)])

    def forward(self, x: torch.Tensor):
        x = self.cnn_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ann_stack(x)
        return torch.stack([torch.reshape(output_stack(x), shape=(-1,)) for output_stack in self.output_stacks], dim=-1)
