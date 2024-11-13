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
        count = 0
        for i in range(len(labels)):
            for idx, coords in enumerate(maske):
                for label, self.mask in enumerate(coords):
                    print(f"Processing label {label}, mask shape: {self.mask.shape}")

                    if self.mask.size > 0:
                        print(f"self.Mask {self.mask}")

                        self.mask = np.array(self.mask, dtype=np.int32).reshape((-1, 1, 2))
                        for coords in self.mask:
                            for coord in coords:    #coord [219 719]
                                # if isinstance(coord, (list, tuple)) and len(coord) > 1:
                                print(f"forcoordsforcoordincoordsself.mask[0] {coord[1]}")
                                a = coord[1]    # 719
                                print(f"forcoordsforcoordincoordsself.mask[0] {a}")


                        if self.mask[0] is None:
                            self.mask[0] = 0
                            print(f"if self.mask[0] is None:self.Mask {self.mask[1]}")

                        elif self.mask[1] is None:
                            self.mask[1] = 0
                        print(f"if self.mask[1] is None:self.Mask {self.mask[0]}")

                        # Zuweisung je nach Label
                        xy_dict = {0: xy_ml, 1: xy_mr, 2: xy_yl, 3: xy_yr}
                        if label in xy_dict:
                            xy_dict[label].append(self.mask)
                            xy_array = np.array(xy_dict[label])
                            print(f"xy_{label}shape:", xy_array.shape)
                            print(f"xy_{label}len:", len(xy_array))

                        count += 1
                        print("Total count:", count)
                    else:
                        self.mask[0] = 0
                        self.mask[1] = 0
                        print("Yo, das Zelle ist leer! Sie würde gefüllt mit (x=0,y=0)")

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

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)      #damit es in Pillow-Format vorliegt, sonst schimpft 'transform'
        xy_ml, xy_mr, xy_yl, xy_yr = self.extract_polygons(self.full_dir_path)

        target = self.extract_listgon(xy_ml, xy_mr, xy_yl, xy_yr)
        #xy_mr, xy_yl, xy_yr ist leer
        if self.transform:
            img = self.transform(img)
        print("lentarget", target)
        print("typetarget", type(target))


        return img, target

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))

# Instantiate your dataset
data_complete = Mydataset(path, transform=transform)  # IST es nicht das Gleiche?

dataset_size = len(data_complete)

# train_set, validate_set= torch.utils.PennFudanPed.random_split(dataset, [round(len(dataset)*0.7), (len(dataset) - round(len(dataset)*0.7))])
train_size = round(dataset_size * 0.7)
val_size = round(dataset_size * 0.15)
test_size = dataset_size - train_size - val_size  #Trai(70%), Val(15%), Tes(15%)
train_set, val_set, test_set = torch.utils.data.random_split(data_complete, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)


"""
Traceback (most recent call last):
  File "/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/main.py", line 43, in <module>
    train(n_epochs, dataloader_train)
  File "/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/testing_the_prototype.py", line 106, in train
    losses = loss1 + loss2  # oder eine andere Berechnung
             ~~~~~~^~~~~~~
RuntimeError: The size of tensor a (10) must match the size of tensor b (1568) at non-singleton dimension 1
"""