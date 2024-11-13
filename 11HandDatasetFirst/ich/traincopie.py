from torch import nn, optim
from model import *

import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from Mydatasetsofortmaskelabeleinpfad import *
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



def train(n_epochs, dataloader_train):
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

    n_batches = len(dataloader_train)
    model.train()  # Modell in den Trainingsmodus setzen
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
      # #Alternativ:
      # for images, labels in dataloader_train:
      #     # Bilder und Labels auf das gleiche Gerät wie das Modell verschieben
      #     images, labels = images.to(device), labels.to(device)
      #   for sample in dataloader_train:
      #   for batch_idx, sample in enumerate(dataloader_train):
        for img, sample in dataloader_train:
            print("len(sample):", len(sample))      # Gibt die Anzahl der Elemente im Batch zurück
            print("sample:", sample)
            # print("samplesize:", sample.size)
            print("keys", sample.keys())
            # img = sample[0].cuda()
            # target = sample[1]
            # target = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in target.items()}
            # img, target = sample[0].cuda(), sample[1].cuda()  # or something similar

            img = img.to(device)


    # for img, target in dataloader_train:  # Hier wird das Dataloader-Objekt verwendet
            # print("images:", img)
            # print("target:", target)
            # img = img.cuda()
            # img, labels = dataloader_train[0].to(device), dataloader_train[1].to(device)

            # images, labels = images.to(device), labels.to(device)
            # Compute prediction error
            pred = model(img, target)

            loss = loss_func(pred, target)
            # optimizer.zero_grad()

            # Vorwärtsdurchlauf
            loss_dict = model(img, target)
            losses = sum(loss for loss in loss_dict.values())
            # Alternativ: # Vorwärtsdurchlauf
            # output = model(images)[0]

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Ausgabe der Zwischenschritte
            print(f"Epoch {epoch}, Loss: {losses.item()}")

            # if (batch + 1) % 100 == 0:
            #     loss, current = loss.item(), (batch + 1) * len(img)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{n_batches:>5d}]")

                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #       .format(epoch + 1, n_epochs, batch + 1, n_batches, loss.item()))
        # Am Ende jeder Epoche Gradienten zurücksetzen (optional, zur Sicherheit)
        optimizer.zero_grad()
