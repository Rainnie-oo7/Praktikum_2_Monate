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
# min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
# It does not only pop up with Automatic1111 webui, ( whatever that is ), i epxerimented with a Pytorch Object Detection Finetuning Tutorial, as well as it's Writing Custom Dataset, Dataloaders, Transformers, and this error popped up all my Datasets. Cannot find details or solutio for it.
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

    def extract_polygons(self, mask_path):
        mat_data = scipy.io.loadmat(mask_path)
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
        black_img = np.zeros_like(img)
        color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}

        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                 zip(self.polygons_data['myleft'][0], self.polygons_data['myright'][0],
                     self.polygons_data['yourleft'][0], self.polygons_data['yourright'][0])]
        # maske = torch.tensor(maske)
        labels = ['ml', 'mr', 'yl', 'yr']
        # Construct masks from polygons
        for idx, (label, array_mlmrylyr) in enumerate(zip(labels, maske)):
            if array_mlmrylyr == [0, 0] or not array_mlmrylyr:
                print(f"{label} ist leer.")
            else:
                print(f"{label} hat Daten.")
                for mask in array_mlmrylyr:
                    if mask.size > 0:
                        mask = np.array(mask, dtype=np.int32)
                        mask = mask.reshape((-1, 1, 2))
                        cv2.fillPoly(black_img, [mask], color[label])

        # Convert to tensor after filling it
        if isinstance(black_img, np.ndarray):
            mask = torch.tensor(black_img, dtype=torch.uint8).permute(2, 0, 1)  # Ensure shape is (C, H, W)
            print("Mask shape:", mask.shape)  # Debug: Check mask shape

        # Check if there are any masks
        obj_ids = torch.unique(mask)
        if obj_ids.numel() == 0 and obj_ids is not None:
            print("No valid masks found for this image.")
            # Return a dummy target or handle as needed
            return img, {"boxes": torch.empty(0, 4), "masks": torch.empty(0, img.shape[0], img.shape[1]),
                         "labels": torch.empty(0), "image_id": idx}

        # First id is the background, so remove it
        obj_ids = obj_ids[1:]

        # Create masks from unique object IDs
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # Debug: Check masks shape and content
        print("Unique object IDs:", obj_ids)
        print("Masks shape:", masks.shape)  # Debug: Check masks shape

        # Create bounding boxes
        if masks.numel() > 0 and masks is not None:
            try:
                boxes = masks_to_boxes(masks)
                print("Boxes shape:", boxes.shape)  # Debug: Check boxes shape
            except Exception as e:
                print("Error during masks_to_boxes:", e)
                boxes = torch.empty(0, 4)  # Return empty if there are no boxes
        else:
            boxes = torch.empty(0, 4)  # Return empty if there are no boxes

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
#
# # Instantiate your dataset
# data_complete = Mydataset(path, transform=transform)  # IST es nicht das Gleiche?
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
