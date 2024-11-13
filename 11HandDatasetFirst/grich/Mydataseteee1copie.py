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

    def __init__(self, path, root, transform=None):
        self.root = root
        self.imgs = []
        self.mymasks = []
        self.transforms = transforms
        if isinstance(path, str):
            for dir in os.listdir(path):
                self.full_dir_path = os.path.join(path, dir)
                if os.path.isdir(self.full_dir_path):
                    # print(f"Checking directory: {self.full_dir_path}")
                    for file in os.listdir(self.full_dir_path):
                        if file.startswith('frame'):
                            files = sorted([file for file in os.listdir(self.full_dir_path) if file.startswith('frame')])#Man sieht alle Files Bilder pol.mat Datei.
                            # Lade Bilder in der sortierten Reihenfolge
                            for frame_file in files:                            # Keine in-root Legung von Ordner img Ordner masken Ordner misc, daher sortiert im geititem oder in der init???
                                img_path = os.path.join(self.full_dir_path, frame_file)
                                self.imgs.append(cv2.imread(img_path))

                            # Prüfe, ob die Datei mit 'polygons' beginnt
                        elif file.startswith('polygons'):
                            self.full_path = os.path.join(self.full_dir_path, file)
                            self.boxes = self.extract_polygons(self.full_path)
        else:
            print("Der Pfad muss vom Datentyp(String) sein.!")

#Was macht?
        # for idx, img in enumerate(self.imgs):
        #     if idx < 0 or idx > len(self.mymasks):
        #         print(f"Index {idx} is out of range für len(self.maske): {len(self.mymasks)}")

        print("lenimg:", len(self.imgs))
    # def getimg(self):
    #     self.img = self.imgs[idx]

    def extract_polygons(self, idx):
        # try:
        #     idx = int(idx)
        # except ValueError:
        #     raise ValueError(f"Der Index muss ein Integer sein, erhalten: {type(idx)}")
        imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
        img = cv2.imread(imgsizepath)
        color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}
        mymasks = []
        mat_data = scipy.io.loadmat(self.full_path)
        data = mat_data['polygons']

        if isinstance(data['myleft'], np.ndarray):
            length = data['myleft'].shape[1]  # Wenn es ein 2D-Array ist
        else:
            length = len(data['myleft'])  # Wenn es eine Liste ist

        if idx < 0 or idx >= length:  # Beispiel: 2D-Array
            raise IndexError(f"Index {idx} ist außerhalb des gültigen Bereichs.")

        mask_data = {
            'myleft': data['myleft'][0][idx],
            'myright': data['myright'][0][idx],
            'yourleft': data['yourleft'][0][idx],
            'yourright': data['yourright'][0][idx]
        }
        mymasks = torch.tensor(mask_data)
        return mymasks

     def mkmaskslist(self, mask_data):
        # Für jedes Label und die zugehörige Maske
        for label, polygon_coords in mask_data.items():
            # melle = sorted([label for label, polygon_coords in mask_data.items() if
            #                 polygon_coords.size > 0])
            if polygon_coords.size > 0:  # Überprüfen, ob die Maske existiert
                polygon_coords = np.array(polygon_coords, dtype=np.int32)  # Koordinaten in Ganzzahl umwandeln
                polygon_coords = polygon_coords.reshape((-1, 1, 2))  # Umformen für OpenCV
        return polygon_coords

    def mkblackslist(self, polygon_coords):
        # Schwarzes Bild für Maske erstellen
        black_img = np.zeros_like(img)
        # Maske auf schwarzem Bild einzeichnen
        cv2.fillPoly(black_img, [polygon_coords], color[label])
        mymasks.append(black_img)
        print(f"mymasks für idx {idx}:", len(mymasks))
        return mymasks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
        # img = read_image(imgsizepath)
        img = self.imgs[idx]
        self.mymasks = self.extract_polygons(idx)
        boxes = masks_to_boxes(self.mymasks)
        masks = self.mkblacklist()[idx]


        for mask in self.mymasks:
            # # instances are encoded as different colors
            obj_ids = torch.unique(mask)
            # # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
            return masks
        labels = torch.ones(4, dtype=torch.int64)

        # Parameters needed for RCNN training
        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = labels

        # Return the image in PIL format or applied the transform
        if ((self.transforms is not None) and (self.get_image_PIL == False)):
            img = self.transforms(img)

        return img, target

    def run(self):
        print(f"Hallo!")


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

if __name__ == '__main__':
    object = Mydataset(Dataset)
    object.run()