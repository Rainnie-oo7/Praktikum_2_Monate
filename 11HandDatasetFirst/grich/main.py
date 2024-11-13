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
from model import *
from train import *
from evaluate import test
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor





# Datenset und Optimizer laden
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 2
# print("model train started")
model.train()
# print("model train finished")
if __name__ == '__main__':

    train= train()
