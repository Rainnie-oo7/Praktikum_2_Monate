import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from main import *

from skimage.draw import polygon
from torchvision import transforms
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.ops.boxes import masks_to_boxes
import scipy
import numpy as np


class PolygonModell(nn.Module):
    def __init__(self):
        super(PolygonModell, self).__init__()
        # Convolutional Layers für Bildmerkmale
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Lineare Schicht für Polygondaten (8 Werte für die 4 (x, y)-Punkte) in: 8 out: 64
        self.fc_polygon = nn.Linear(2, 921600)

        # Kombinierte Features
        self.fc_combined = nn.Linear(64 + 64, 128)
        self.fc_output = nn.Linear(128, 1)  # 1 für die Vorhersage (z. B. Objektklasse oder andere Ausgabe)

    def forward(self, image, polygon):
        image = image.float()
        # image = image.permute(0, 3, 1, 2)  # Ändere die Dimensionen von (Batch, Höhe, Breite, Kanäle) zu (Batch, Kanäle, Höhe, Breite)

        # Convolutional layers für Bildmerkmale
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        polygon = F.relu(self.fc_polygon(polygon))
        print(f"x shape: {x.shape}")
        print(f"polygon shape: {polygon.shape}")
        combined = torch.cat((x, polygon), dim=1)

        # Durch das Netzwerk führen
        combined = F.relu(self.fc_combined(combined))
        output = self.fc_output(combined)

        return output