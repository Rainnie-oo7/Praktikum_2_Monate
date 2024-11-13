import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from numpy.lib.function_base import extract

# from main import *

from skimage.draw import polygon
from torchvision import transforms
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import tv_tensors
import scipy
import numpy as np


class Mydataset(Dataset):

    def __init__(self, path, transforms=None):
        self.path = osp.normpath(osp.join(osp.dirname(__file__), "data"))
        self.imgs = []
        self.masks = []

        # image_size = (720, 1280)
        # self.masks = self.create_mask_image(self.maske, image_size)
        self.transforms = transforms
        for idx, img in enumerate(self.imgs):
            if idx < 0 or idx > len(self.masks):
                print(f"Index {idx} is out of range für len(self.maske): {len(self.masks)}")
            else:
                for dir in os.listdir(path):
                    full_dir_path = os.path.join(path, dir)
                    if os.path.isdir(full_dir_path):
                        # Iteriere durch alle Dateien im Unterverzeichnis
                        for file in os.listdir(full_dir_path):
                            if file.startswith('frame'):
                                self.imgs.append(cv2.imread(os.path.join(full_dir_path, file)))

                            else:
                                self.masks = self.masks + self.extract_polygons(self.path)
                                # print(len(self.masks)) #// 19???
    # def getimg(self):
    #     self.img = self.imgs[idx]
    def extract_polygons(self):
        color = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 255)]
        mymasks = []
        black_img = []
        # print(self.path)
        # Iteriere durch alle Verzeichnisse im angegebenen Pfad
        for dir in os.listdir(self.path):
            full_dir_path = os.path.join(self.path, dir)

            # Überprüfen, ob es sich um ein Verzeichnis handelt
            if os.path.isdir(full_dir_path):
                # Iteriere durch alle Dateien im Unterverzeichnis
                for file in os.listdir(full_dir_path):
                    if file.startswith('polygons') and file.endswith('.mat'):
                        file_path = os.path.join(full_dir_path, file)
                        print(f"Verarbeite Datei: {file_path}")

                        # Lade die .mat-Datei
                        mat_data = scipy.io.loadmat(file_path)
                        data = mat_data['polygons']
                        print("data", data, "huhn")
                        # print(PennFudanPed)
                        # print(PennFudanPed.shape[0]) # // 1, 1, 1, ..
                        # print("Das ist PennFudanPed:", PennFudanPed)
                        # If 'PennFudanPed' is a structured array, you may need to iterate over its rows
                        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                                          zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
                        # maske = torch.tensor(maske)


                        for idx,elem in enumerate(maske):
                            for mask in elem:
                                black_img = np.zeros_like(self.img)
                                if mask.size > 0:
                                    cv2.fillPoly(black_img, mask, color[idx])
                            mymasks.append(black_img)
                            # mymasksIter = mymasks
                            print("mymasks:", mymasks)
                            # print("mymasks:", len(mymasks))

                        # for i in range(PennFudanPed.shape[0]):
                        for i in range(99):
                            self.myleft = data['myleft'][0][i]
                            # self.myleft = np.asarray(self.myleft, dtype="object")
                            self.myright = data['myright'][0][i]
                            # self.myright = np.asarray(self.myright, dtype="object")
                            self.yourleft = data['yourleft'][0][i]
                            # self.yourleft = np.asarray(self.yourleft, dtype="object")
                            self.yourright = data['yourright'][0][i]
                            # self.yourright = np.asarray(self.yourright, dtype="object")

                            # if PennFudanPed['myleft'][0][i].size == 0:
                            #     PennFudanPed['myleft'][0][i] = np.array([])
                            # if PennFudanPed['myright'][0][i].size == 0:
                            #     PennFudanPed['myright'][0][i] = np.array([])
                            # if PennFudanPed['yourleft'][0][i].size == 0:
                            #     PennFudanPed['yourleft'][0][i] = np.array([])
                            # if PennFudanPed['yourright'][0][i].size == 0:
                            #     PennFudanPed['yourright'][0][i] = np.array([])

                        print("Myleft values:", data['myleft'][0][i].size)  # Weil es sich nur um eine DImenstion handel 0
                                                                            # ohne 0 wäre es im zweiten Durchlauf zu keinem Wert gekommen.
                        print("Myright values:", data['myright'][0][i].size)
                        print("Yourleft values:", data['yourleft'][0][i].size)
                        print("Yourright values:", data['yourright'][0][i].size)
                        continue

                        # for i in range(PennFudanPed.shape[0]):
                        #     myleft, myright, yourleft, yourright = PennFudanPed[['myleft'], ['myright'], ['yourleft'], ['yourright']][i, 0]
                        #     print(f"Ml: {myleft}, mr: {myright}, yl: {yourleft}, yr: {yourright}")

                        # Erstelle Maske
                        # maske = {f"key_{i}": {"myleft": ml, "myright": mr, "yourleft": yl, "yourright": yr}
                        #          for i, (ml, mr, yl, yr) in
                        #          enumerate(zip(self.myleft, self.myright, self.yourleft, self.yourright))}

                        # if len(self.myleft) == len(self.myright) == len(self.yourleft) == len(self.yourright):

                        print("maske:", maske)
                        print(type(maske))
                        print("shapemaske:", maske.shape)
                        masks.extend(mymasks)     #makes no structured or ensted array, but an array

        # print(f"gesamtanzahl masks: {len(masks)}")  # //2485
        return mymasksIter, mymasks


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        self.img = self.imgs[idx]
        self.mymasks = self.extract_polygons()
        # instances are encoded as different colors
        obj_ids = torch.unique(self.masks)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        #obj_ids is a 1D tensor that contains the unique identifiers (IDs)
        # for each object class present in the color-encoded mask. The [:, None, None] operation
        # reshapes obj_ids from shape (N,) (where N is the number of object classes) to shape (N, 1, 1).
        # This allows for broadcasting when comparing against the mask.

        #result is a boolean tensor where:Each pixel in the mask is checked against each object ID. Resulting
        # Tensor has shape (N,H,W) where N indicateds boolean. vvv makes it to 0/1
        for mask in range(self.masks):
            masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        labels = torch.tensor(["myleft", "myright", "yourleft", "yourright"], dtype=torch.int64)

        image_id = idx
        # Wrap sample and targets into torchvision tv_tensors:
        # maske = tv_tensors.Mask(self.sieveoutbackground())
        target = {}
        target["masks"] = self.masks
        target["labels"] = labels
        target["image_id"] = image_id

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        #
        # return img, target
    def run(self):
        print("hello from run")

path = osp.normpath(osp.join(osp.dirname(__file__), "data"))

if __name__ == '__main__':
    object = Mydataset(Dataset)
    object.extract_polygons()
