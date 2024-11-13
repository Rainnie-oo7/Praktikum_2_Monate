from Mydataseteee2 import *
import torch.nn as nn


# for X, y in dataloader_test:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y (y is a dict):{len(y)}{type(y)}")
#     break



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        # fully connected layer
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, img, target):
        img = self.conv1(img)
        img = self.conv2(img)
        target = target
        # flatten the output of conv2
        # img = img.view(img.size(0), -1)
        img = img.view(-1, 32 * 7 * 7)
        output = self.out(img)
        return output, img, target