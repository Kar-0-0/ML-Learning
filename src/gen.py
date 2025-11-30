import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 3136) # (64x7x7)
        self.conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 7, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        return x