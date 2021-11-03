import torch
from torch import nn
from torchsummary import summary

from helper.data_format import ImageFormat


class PlayModel(nn.Module):
    def __init__(self, num_outputs: int):
        super(PlayModel, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(8, 8), stride=(2, 2), padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.ann_stack = nn.Sequential(
            nn.Linear(1920, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
        )
        self.output_stacks = nn.ModuleList([nn.Sequential(
            nn.Linear(1024, 128),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ) for _ in range(num_outputs)])

    def forward(self, x: torch.Tensor):
        x = self.cnn_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ann_stack(x)
        return torch.stack([torch.reshape(output_stack(x), shape=(-1,)) for output_stack in self.output_stacks], dim=-1)


if __name__ == '__main__':
    model = PlayModel(num_outputs=4).to('cuda')
    img_size = ImageFormat(width=192, height=108, channel=3)
    summary(model, img_size.tensor_shape(), batch_size=64)
    test = torch.rand(100, *img_size.tensor_shape(), device='cuda')
    print(model(test)[0].shape)
