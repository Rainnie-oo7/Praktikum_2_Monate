import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PolygonModell import *
from Mydatasetsofortmaskelabel import *
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
from model import CNN
from traind import train
from evaluate import test

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model():

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # Hintergrund + Handobjekte
    # Define a hidden layer size (commonly same as the in_features)
    hidden_layer = 256  # You can change this if needed
    # Anzahl der Ausgabeklassen anpassen (num_classes = Hintergrund + Anzahl der Objekte)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, hidden_layer, num_classes)

    return model



model = get_model()
print("dataset:", dataset)
# Datenset und Optimizer laden
optimizer = optim.Adam(model.parameters(), lr=0.001)
data_train = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
data_test = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))

dataset = Mydataset(path)
train(n_epochs, cnn, train_loader)
