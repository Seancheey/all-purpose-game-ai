import torch
from torch import nn
from helper.data_format import img_size, key_map


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
            nn.Linear(16, key_map.shape[0] * key_map.shape[1]),
            nn.Sigmoid()
        )
        self.prediction = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x = torch.div(x, 255)
        x = self.flatten(x)
        x = self.ann_stack(x)
        x = torch.reshape(x, (-1, key_map.shape[0], key_map.shape[1]))
        x = self.prediction(x)
        return x


if __name__ == '__main__':
    model = ANN().to('cuda')
    test = torch.rand(1, 108, 192, 3, device='cuda')
    print(model(test))
