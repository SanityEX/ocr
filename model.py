import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # [B, 32, 16, 60]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # [B, 64, 8, 30]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # [B, 128, 4, 30]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),   # [B, 256, 1, 30]
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)             # [B, 256, 1, W]
        x = x.squeeze(2)            # [B, 256, W]
        x = x.permute(2, 0, 1)      # [W, B, 256]
        x = self.classifier(x)      # [W, B, num_classes]
        return x