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
        for batch, (img, labels) in enumerate(data_train):
      # #Alternativ:
      # for images, labels in dataloader_train:
      #     # Bilder und Labels auf das gleiche Gerät wie das Modell verschieben
      #     images, labels = images.to(device), labels.to(device)
      #   for sample in dataloader_train:
      #   for batch_idx, sample in enumerate(dataloader_train):
      #   for img, target in dataloader_train:
            print("len(target):", len(target))      # Gibt die Anzahl der Elemente im Batch zurück
            print("target:", target)

            # img = sample[0].cuda()
            # target = sample[1]
            # target = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in target.items()}
            # img, target = sample[0].cuda(), sample[1].cuda()  # or something similar
            img = img.to(device)

            # img, labels = dataloader_train[0].to(device), dataloader_train[1].to(device)
            """
            # List Comprehension: Das bedeutet, dass eine neue Liste erstellt wird, indem über alle Elemente in der targets-Liste iteriert wird.
            # Dict Comprehension:
            Für jedes Dictionary t in der Liste targets, wird ein neues Dictionary erstellt, das die gleichen Schlüssel k und Werte v enthält.
            t.items() gibt alle Schlüssel-Wert-Paare des Dictionaries t zurück, und für jedes Paar k und v wird geprüft
            """

            # target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target]
            target = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                if isinstance(t, dict) else t  # Füge dies hinzu, um Strings oder andere Typen zu ignorieren
                for t in target
            ]

            for t in target:
                if not isinstance(t, dict):
                    print(f"Unexpected type in target: {type(t)}, value: {t}")

            # images, labels = images.to(device), labels.to(device)
            # Compute prediction error
            pred = model(img, target)

            if isinstance(pred, tuple): ## Wenn `pred` ein Tupel ist, nimm den ersten Wert (logits) für die Vorhersagen
                pred = pred[0]  # Extrahiere den relevanten Tensor

            for t in target:
                target_labels = torch.tensor(t[2], dtype=torch.long).to(
                    device)  # Zugriff auf 'labels', falls sie das erste Element sind
                loss = loss_func(pred, target_labels)


            # Sicherstellen, dass target eine Liste von Zahlen ist und keine Strings enthält
            # if isinstance(target, list):
            #     target_cleaned = []
            #     for t in target:
            #         if isinstance(t, (int, float)):  # Überprüfen, ob t eine Zahl ist
            #             target_cleaned.append(t)
            #         else:
            #             print(
            #                 f"Unerwarteter Typ in target: {t} (Typ: {type(t)})")  # Debugging-Ausgabe für fehlerhafte Elemente
            #
            #     # Konvertiere die bereinigte Liste in einen Tensor
            #     target = torch.tensor(target_cleaned, dtype=torch.long)

            # loss = loss_func(pred, target)
            loss = loss_func(pred, target['labels'].to(device))            # with torch.cuda.amp.autocast(enabled=scaler is not None):
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

        optimizer.zero_grad()


