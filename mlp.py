import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP"""
    def __init__(self):
        super().__init__()
        # (B, 512)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )  # (B, 256)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 52),
            nn.ReLU()
        )
        # (B, 52)
        self.fc3 = nn.Sequential(
            nn.Linear(52, 4),
            nn.ReLU()
        )
        # (B, 4)
        # nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        # x = [B, 512]
        x = self.fc1(x)
        # x = [B, 256]
        x = self.fc2(x)
        # x = [B, 54]
        x = self.fc3(x)
        # x = [B, 4]
        return x
