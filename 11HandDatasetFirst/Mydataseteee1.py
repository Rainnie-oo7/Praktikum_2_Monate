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


class Mydataset(Dataset):
    def __init__(self, path, transform=None):
        self.imgs = []
        self.mymasks = []
        self.transforms = transform
        self.polygons_data = []  # Speichert alle Polygon-Koordinaten

        # Lade die Bilder und Masken-Pfade
        if isinstance(path, str):
            for dir in os.listdir(path):
                full_dir_path = os.path.join(path, dir)
                if os.path.isdir(full_dir_path):
                    files = sorted([file for file in os.listdir(full_dir_path) if file.startswith('frame')])
                    for file in files:
                        img_path = os.path.join(full_dir_path, file)
                        self.imgs.append(cv2.imread(img_path))
                    mask_path = os.path.join(full_dir_path, 'polygons.mat')  # Beispiel für Maskenpfad
                    self.polygons_data = self.extract_polygons(mask_path)  # Alle Polygone laden

    def extract_polygons(self, full_path):
        mat_data = scipy.io.loadmat(full_path)
        data = mat_data['polygons']
        polygons = {
            'myleft': data['myleft'][0],
            'myright': data['myright'][0],
            'yourleft': data['yourleft'][0],
            'yourright': data['yourright'][0]
        }
        return polygons

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        # Das bedeutet, dass label nacheinander die Werte der Schlüssel (z.B. myleft, myright, etc.) annimmt. #idx wird verwendet, um den spezifischen Eintrag (z.B. eine Liste von Polygon-Koordinaten) zu extrahieren, der dem aktuellen Index (idx) entspricht.

        # mask_data = {label: self.polygons_data[label][idx] for label in self.polygons_data}
        # print(f"Der aktuelle Index ist: {idx}")
        # for label in self.polygons_data:
        #     print(f"{label} hat eine Länge von: {len(self.polygons_data[label])}")
        #
        # # Erstelle schwarze Maske für das Bild
        # black_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # for label, polygon_coords in mask_data.items():
        #     if polygon_coords.size > 0:
        #         polygon_coords = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
        #         cv2.fillPoly(black_img, [polygon_coords], 255)  # Beispiel für weiß
        color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}

        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                 zip(self.polygons_data['myleft'][0], self.polygons_data['myright'][0], self.polygons_data['yourleft'][0], self.polygons_data['yourright'][0])]
        # maske = torch.tensor(maske)
        labels = ['ml', 'mr', 'yl', 'yr']
        mapped_data = {}
        black_img = np.zeros_like(img)
        for idx, (label, array_mlmrylyr) in enumerate(zip(labels, maske)):
            if array_mlmrylyr == [0, 0] or []:
                print(f"{label} ist leer.")
            else:
                print(f"{label} hat Daten.")
                for mask in array_mlmrylyr:
                    if mask.size > 0:
                        mask = np.array(mask, dtype=np.int32)  # CV will integer habenCV_32S
                        # Reshape to (N, 1, 2) for OpenCV
                        mask = mask.reshape((-1, 1, 2))
                        print("idx:", idx)
                        print("Erster Eintrag mask:", mask[0])
                        print("typemask:", type(mask))
                        # print("mask:", mask)
                        print("lenmask:", len(mask))
                        print("typemask:", mask.shape)
                        print("sizemask:", mask.size)
                        print("coloridx:", color[label])
                        cv2.fillPoly(black_img, [mask], color[label])

        mask = torch.tensor(black_img, dtype=torch.uint8)  # Convert to tensor
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        masks = torch.tensor(black_img, dtype=torch.uint8).unsqueeze(0)
        masks = masks.squeeze(0).permute(2, 0, 1)


        boxes = masks_to_boxes(masks)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "masks": masks,
            "labels": labels,
            "image_id": idx
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def run(self):
        print(f"Hallo!")


transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

path = osp.normpath(osp.join(osp.dirname(__file__), "data"))

# # Instantiate your dataset
# # data_complete = Mydataset(path, transform=transform)  # IST es nicht das Gleiche?
#
# dataset_size = len(data_complete)
#
# # train_set, validate_set= torch.utils.PennFudanPed.random_split(dataset, [round(len(dataset)*0.7), (len(dataset) - round(len(dataset)*0.7))])
# train_size = round(dataset_size * 0.7)
# val_size = round(dataset_size * 0.15)
# test_size = dataset_size - train_size - val_size  #Trai(70%), Val(15%), Tes(15%)
# train_set, val_set, test_set = torch.utils.data.random_split(data_complete, [train_size, val_size, test_size])
#
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

if __name__ == '__main__':
    object = Mydataset(Dataset)
    object.run()