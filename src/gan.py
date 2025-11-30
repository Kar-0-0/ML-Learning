import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)


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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)        
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, 3, 1, 0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn1(self.conv2(x)))
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x.view(-1)

epochs = 50
disc = Discriminator().to(device)
gen = Generator().to(device)
optimizerG = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(9, 100, device=device) 
plt.ion()
for epoch in range(epochs):
    # Generate Images Every Epoch
    print(f"Epoch: {epoch}")
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu() 
        grid = make_grid(fake, nrow=3, padding=2,
                        normalize=True, value_range=(-1, 1))

        npimg = grid.numpy().transpose(1, 2, 0) 

        plt.clf()                 
        plt.imshow(npimg, cmap="gray")
        plt.axis("off")
        plt.draw()         
        plt.pause(0.001) 

    for i, (real_img, _) in enumerate(train_loader):
        real_img = real_img.to(device)
        # Check Discriminator Loss on Real Images First (Training Discriminator)
        bs = real_img.size(0)
        real_out = disc(real_img).view(-1)
        lossD_real = F.binary_cross_entropy(real_out, torch.ones((bs,), device=device))

        # Now check it on Fake Images (Training Discriminator)
        noise = torch.randn((bs, 100), device=device)
        fake_img = gen(noise).detach()
        fake_out = disc(fake_img).view(-1)
        lossD_fake = F.binary_cross_entropy(fake_out, torch.zeros((bs,), device=device))

        real_loss = lossD_fake + lossD_real
        disc.zero_grad()
        real_loss.backward()
        optimizerD.step()
        
        # Train Generator
        noise = torch.randn((bs, 100), device=device)
        fake_imgs = gen(noise)
        fake_out = disc(fake_imgs)
        lossG = F.binary_cross_entropy(fake_out, torch.ones((bs,), device=device))

        gen.zero_grad()
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0 or i+1 == len(train_loader):
            print(f"{i}/{len(train_loader)}")

model = Generator()
torch.save(gen.state_dict(), 'final_model.pth')