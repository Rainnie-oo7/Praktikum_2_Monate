from torch import nn, optim
from model import *
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from MydatasetEPO import *
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

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
print(f"Using {device} device")

model = CNN().to(device)

loss_func = nn.CrossEntropyLoss()

def train(n_epochs, model, data_train):
    n_batches = len(data_train)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        for batch, (img, labels) in enumerate(data_train):
            # Vorwärtsdurchlauf
            pred = model(img)
            loss = loss_func(pred, labels)

            # Backpropagation
            loss.backward()

            # Akkumulationsschritte überprüfen
            if (batch + 1) % accumulation_steps == 0:
                optimizer.step()  # Optimierungsschritt
                optimizer.zero_grad()  # Gradienten zurücksetzen

            # Ausgabe der Zwischenschritte
            if (batch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{n_epochs}], Step [{batch + 1}/{n_batches}], Loss: {losses.item():.4f}")

        # Am Ende jeder Epoche Gradienten zurücksetzen (optional, zur Sicherheit)
        optimizer.zero_grad()


