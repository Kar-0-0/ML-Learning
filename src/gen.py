import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 256 * 8 * 8)   # 256x8x8
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 128x16x16
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 64x32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)         # 3x32x32
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        return x