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
import torchvision
from torchvision.ops.boxes import masks_to_boxes
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
def bratwurst(path, transform=None):
    path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
    imgs = []
    blackimglist = []
    unterordner_liste = []
    transform = transform
    for dir in os.listdir(path):
        full_dir_path = os.path.join(path, dir)
        if os.path.isdir(full_dir_path):
            # print(f"Checking directory: {full_dir_path}")
            for file in os.listdir(full_dir_path):
                # print(f"Found file: {file}")
                if file.startswith('frame'):
                    imgs.append(cv2.imread(os.path.join(full_dir_path, file)))
                else:
                    if file.startswith('polygons'):
                        full_path = os.path.join(full_dir_path, file)
                        # print("full_path:", full_path)
                        ml, mr, yl, yr = extract_polygons(full_path)
                        stored_lists = extract_listgon(ml, mr, yl, yr)
                        print()
                        return full_path
def extract_polygons(full_path):
    labels = [0, 1, 2, 3]
    mat_data = scipy.io.loadmat(full_path)
    data = mat_data['polygons']
    maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
             zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
    ml = []
    mr = []
    yl = []
    yr = []
    count = 0

    for idx, coords in enumerate(maske):
        for label, mask in enumerate(coords):

            for idx, coords in enumerate(maske):  # Iteriere über die Masken
                # Du gehst davon aus, dass coords 4 Koordinatenpaare enthält (ml, mr, yl, yr)
                ml.append(coords[0])  # Füge das erste Paar in ml hinzu
                mr.append(coords[1])
                yl.append(coords[2])
                yr.append(coords[3])

            print("ML:", ml)
            print("MR:", mr)
            print("YL:", yl)
            print("YR:", yr)

            return ml, mr, yl, yr


def extract_listgon(ml, mr, yl, yr):
    # Beispiel: Falls ml, mr, yl, yr bereits gefüllt sind
    print("ML:", ml)
    print("MR:", mr)
    print("YL:", yl)
    print("YR:", yr)

    # Listen für die x- und y-Koordinaten
    x_listml = []
    y_listml = []
    x_listmr = []
    y_listmr = []
    x_listyl = []
    y_listyl = []
    x_listyr = []
    y_listyr = []
    x_list = []
    y_list = []

    # Listen für die vier Kategorien ml, mr, yl, yr
    coords_list = [ml, mr, yl, yr]
    print("Coords List (ml, mr, yl, yr):", coords_list)  # Debug-Ausgabe vor der Schleife

    # Die zugehörigen x- und y-Listen
    x_lists = [x_listml, x_listmr, x_listyl, x_listyr]
    y_lists = [y_listml, y_listmr, y_listyl, y_listyr]


    stored_lists = {}

    # Schleife, um durch die Koordinaten zu iterieren
    for coords in ml, mr, yl, yr:
        if coords:  # Überprüfen, ob coords nicht leer ist
            for coord in coords:
                if isinstance(coord, (list, tuple)) and len(coord) > 1:  # Überprüfen, ob coord ein Paar ist
                    x_list.append(coord[0])  # x-Koordinate in die entsprechende Liste einfügen
                    y_list.append(coord[1])  # y-Koordinate in die entsprechende Liste einfügen




    print("X_ListML:", x_listml)
    print("Y_ListML:", y_listml)
    print("X_ListMR:", x_listmr)
    print("Y_ListMR:", y_listmr)
    print("X_ListYL:", x_listyl)
    print("Y_ListYL:", y_listyl)
    print("X_ListYR:", x_listyr)
    print("Y_ListYR:", y_listyr)

    return stored_lists


if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
    full_path = bratwurst(path)
    ml, mr, yl, yr = extract_polygons(full_path)
    extract_listgon(ml, mr, yl, yr)

