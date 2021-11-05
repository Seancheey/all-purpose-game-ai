import torch
from torch import nn


class SuperHexagonModel(nn.Module):

    @staticmethod
    def fc_block(num_input, num_output):
        return nn.Sequential(
            nn.Linear(num_input, num_output),
            nn.BatchNorm1d(num_output),
            nn.GELU(),
            nn.Dropout(p=0.4),
        )

    @staticmethod
    def conv_block(channel_in, channel_out, kernel_size, stride=(1, 1), padding=1):
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channel_out),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def __init__(self):
        super(SuperHexagonModel, self).__init__()
        self.cnn_stack = nn.Sequential(
            SuperHexagonModel.conv_block(1, 8, kernel_size=(9, 9), stride=(2, 2), padding=4),
            SuperHexagonModel.conv_block(8, 16, kernel_size=(7, 7), padding=3),
            SuperHexagonModel.conv_block(16, 32, kernel_size=(5, 5), padding=2),
            SuperHexagonModel.conv_block(32, 64, kernel_size=(3, 3), padding=1),
            SuperHexagonModel.conv_block(64, 128, kernel_size=(3, 3), padding=1),
        )
        self.ann_stack = nn.Sequential(
            SuperHexagonModel.fc_block(2048, 1024),
            SuperHexagonModel.fc_block(1024, 1024),
            SuperHexagonModel.fc_block(1024, 1024)
        )
        self.output_stacks = nn.ModuleList([nn.Sequential(
            SuperHexagonModel.fc_block(1024, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
            for _ in range(2)])

    def forward(self, x: torch.Tensor):
        x = self.cnn_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ann_stack(x)
        return torch.stack([torch.reshape(output_stack(x), shape=(-1,)) for output_stack in self.output_stacks], dim=-1)
