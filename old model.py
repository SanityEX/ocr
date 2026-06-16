import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # [B, 64, 24, 60]

            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # [B, 128, 12, 30]

            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # [B, 256, 6, 30]

            # block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # [B, 512, 3, 30]

            # block 5
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [B, 512, 1, 30]
        )

        self.dropout = nn.Dropout(0.2)

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)           # [B, 512, 1, W]
        x = x.squeeze(2)          # [B, 512, W]
        x = self.dropout(x)
        x = x.permute(2, 0, 1)    # [W, B, 512]

        x, _ = self.rnn(x)        # [W, B, 512]
        x = self.fc(x)            # [W, B, C]
        return x