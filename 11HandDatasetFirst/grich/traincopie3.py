from torch import nn, optim
from model import *

import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from MydatasetEPO3 import *
from skimage.draw import polygon
from torchvision import transforms
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.ops.boxes import masks_to_boxes
import scipy
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn as nn
from torch.nn import functional as F
from evaluate import test
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

model.to(device)

criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simple training loop
num_epochs = 5
train_losses, val_losses = [], []

print("len(train_loader)",len(train_loader)) #Nicht 5

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0                      # Ich sollte: Masken bzw Boxes, Labels # Hier sd. Ground-Truth durch die Bilder gegeben. Meine müssen erstmal durch Boxen aus Maskenbilder interpretiert werden (CNN with Boundary Box)
    for img, target in train_loader: # von Training Loop Picture/s
        # print("target:", target)

        # img, target = img.to(device), target.to(device)
    # for images, labels in train_loader:  # von Training Loop Picture/s
    # #     print("labels:", labels)
    #     images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.bacward()
        optimizer.step()
        running_loss += loss.item() * image.size(0)
    train_loss = running_loss/ len(train_loader.data_train)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for image, labels in tqdm(val_loader, desc='Validation loop'):
            # Move inputs and labels to the device
            image, labels = image.to(device), labels.to(device)

            outputs = model(image)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")


#Visualize Losses
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

test(model, test_loader)
