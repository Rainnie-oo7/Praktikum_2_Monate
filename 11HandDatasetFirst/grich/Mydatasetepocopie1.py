import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import os
from numpy.lib.function_base import extract, rot90
from skimage.draw import polygon
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import tv_tensors
import os.path as osp
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import masks_to_boxes
import scipy
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from evaluate import test
import torchvision
from torchvision.ops.boxes import masks_to_boxes
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class Mydataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
        self.imgs = []
        self.blackimglist = []
        self.unterordner_liste = []
        self.transform = transform
        for dir in os.listdir(path):
            self.full_dir_path = os.path.join(path, dir)
            if os.path.isdir(self.full_dir_path):
                # print(f"Checking directory: {self.full_dir_path}")
                for file in os.listdir(self.full_dir_path):
                    # print(f"Found file: {file}")
                    if file.startswith('frame'):
                        self.imgs.append(cv2.imread(os.path.join(self.full_dir_path, file)))
                    else:
                        if file.startswith('polygons'):
                            self.full_path = os.path.join(self.full_dir_path, file)
                            # print("full_path:", self.full_path)
                            self.boxes_labels_data = self.extract_polygons(self.full_path)

    def extract_polygons(self, full_path):
        labels = [0, 1, 2, 3]
        mat_data = scipy.io.loadmat(self.full_path)
        data = mat_data['polygons']
        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                 zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
        xy_ml = []
        xy_mr = []
        xy_yl = []
        xy_yr = []
        count1 = 0
        count = 0

        for idx, (label, array_mlmrylyr) in enumerate(zip(labels, maske)):
            for self.mask in array_mlmrylyr:
                # print("array_mlmrylyrshape", len(array_mlmrylyr))   #4
                print("maskshape", self.mask.shape)  # (256, 2)
                if self.mask.size > 0:  #[[692.62 487.84] /n ... [684.70 493.12]]       #label 0 fürml
                    print("label:", label)
                    print("count1", count1)
                    self.mask = np.array(self.mask, dtype=np.int32)  #will integer CV_32S #Reshape to (N, 1, 2)     (.)
                    self.mask = self.mask.reshape((-1, 1, 2))  #macht aus horizontalen Vektoren ( ., .) senkrechte# (.)
                    if label == 0:
                        xy_ml.append(self.mask)
                        print("xy_ml", xy_ml)
                        count += 1  # Zähler erhöhen, wenn Bedingung erfüllt
                        print("count", count)

                    elif label == 1:
                        xy_mr.append(self.mask)
                        count += 1  # Zähler erhöhen, wenn Bedingung erfüllt
                        print("count", count)

                    elif label == 2:
                        xy_yl.append(self.mask)
                        count += 1  # Zähler erhöhen, wenn Bedingung erfüllt
                        print("count", count)

                    elif label == 3:
                        xy_yr.append(self.mask)
                        count += 1  # Zähler erhöhen, wenn Bedingung erfüllt
                        print("count", count)

                    print("xy_coords", xy_ml)
                    print("xy_coords", xy_mr)
                    print("xy_coords", xy_yl)
                    print("xy_coords", xy_yr)

                    return xy_ml, xy_mr, xy_yl, xy_yr

    def extract_listgon(self, xy_ml, xy_mr, xy_yl, xy_yr):
        x_list = []
        y_list = []
        labels = [0, 1, 2, 3]
        boxes_labels_data = []  # Liste für die gesammelten Box- und Label-Daten
        # Für jede Koordinatenliste (ml, mr, yl, yr) durchgehen, Auftrennung von X und y
        for coords in [xy_ml, xy_mr, xy_yl, xy_yr]:
            if coords:
                for coord in coords:
                    if isinstance(coord, (list, tuple)) and len(coord) > 1:
                        x_list.append(coord[0])  # x-Koords. sammeln
                        y_list.append(coord[1])  # y-Koords. sammeln
                    else:
                        print(f"Unerwartetes Format von coord: {coord}")

            try:
                if len(x_list) != 0 and len(y_list) != 0:  # Wenn valide Koordinaten existieren
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)

                    # Beispiel Labels: Wenn du ein Label True/False möchtest, kannst du das so anpassen
                    for label_set in labels:
                        boxes_labels_data.append({
                            'Boxes': [x_min, x_max, y_min, y_max],
                            'Labels': label_set  # Entweder 0, 1 oder eine Liste [0, 1, 2, 3]
                        })
                        print("boxes_labels_data:", boxes_labels_data)
                        print("boxes_labels_datashape:", boxes_labels_data[0])
                        print("boxes_labels_datalen:", boxes_labels_data)
                    # Gibt die Liste mit Boxen und Labels zurück

            except Exception as e:
                print(f"Fehler beim Berechnen der Eckpunkte: {e}")
                return None
            return boxes_labels_data

    def __len__(self):
        return len(self.imgs)

    def takeoneimgfromlist(self):
        for i in self.blackimglist:
            if isinstance(i, np.ndarray):
                j = torch.from_numpy(i)
                # print("mask.shape j:", j.shape)         # 1280 x 720
                return j
            else:
                print("there is noinstance(i, np.ndarray")
                                    #Split the coordinates-Array in side dict xm, yr, mlm, yr


    def __getitem__(self, idx):
        img = self.imgs[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)      #damit es in Pillow-Format vorliegt, sonst schimpft 'transform'
        xy_ml, xy_mr, xy_yl, xy_yr = self.extract_polygons(self.full_dir_path)

        target = self.extract_listgon(xy_ml, xy_mr, xy_yl, xy_yr)

        if self.transform:
            img = self.transform(img)

        return img, target

black_img = np.zeros((1280, 720), dtype=np.uint8)

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
path2 = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat"))

# Instantiate your dataset
data_train = Mydataset(path, transform=transform)  # IST es nicht das Gleiche?
data_test = Mydataset(path, transform=transform)

dataloader_train = DataLoader(data_train, batch_size=4, shuffle=True, num_workers=0)
dataloader_test = DataLoader(data_test, batch_size=4, shuffle=False, num_workers=0)

# train_set, validate_set= torch.utils.PennFudanPed.random_split(dataset, [round(len(dataset)*0.7), (len(dataset) - round(len(dataset)*0.7))])

"""
Traceback (most recent call last):
  File "/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/main.py", line 43, in <module>
    train(n_epochs, dataloader_train)
  File "/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/testing_the_prototype.py", line 106, in train
    losses = loss1 + loss2  # oder eine andere Berechnung
             ~~~~~~^~~~~~~
RuntimeError: The size of tensor a (10) must match the size of tensor b (1568) at non-singleton dimension 1
"""