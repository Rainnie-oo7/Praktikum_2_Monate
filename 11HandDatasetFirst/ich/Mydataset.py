import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from numpy.lib.function_base import extract

from main import *

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

    def __init__(self, path):
        self.path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
        self.imgs = []
        self.mymasks = []

        # image_size = (720, 1280)
        # self.masks = self.create_mask_image(self.maske, image_size)
        self.transforms = transforms

        for dir in os.listdir(path):
            full_dir_path = os.path.join(path, dir)
            if os.path.isdir(full_dir_path):
                # Iteriere durch alle Dateien im Unterverzeichnis
                for file in os.listdir(full_dir_path):
                    if file.startswith('frame'):
                        self.imgs.append(cv2.imread(os.path.join(full_dir_path, file)))

                    else:
                        self.mymasks = self.mymasks + self.extract_polygons()
                        # print(len(self.masks)) #// 19???

        for idx, img in enumerate(self.imgs):
            if idx < 0 or idx > len(self.mymasks):
                print(f"Index {idx} is out of range für len(self.maske): {len(self.mymasks)}")

        print("lenimg:", len(self.imgs))
    # def getimg(self):
    #     self.img = self.imgs[idx]
    def extract_polygons(self):
        imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
        img = cv2.imread(imgsizepath)
        color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}
        mymasks = []
        black_img = []
        # print(path)
        # Iteriere durch alle Verzeichnisse im angegebenen Pfad
        for dir in os.listdir(path):
            full_dir_path = os.path.join(path, dir)

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
                        # print(PennFudanPed)
                        # print(PennFudanPed.shape[0]) # // 1, 1, 1, ..
                        # print("Das ist PennFudanPed:", PennFudanPed)
                        # If 'PennFudanPed' is a structured array, you may need to iterate over its rows
                        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                                 zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
                        # maske = torch.tensor(maske)
                        labels = ['ml', 'mr', 'yl', 'yr']
                        mapped_data = {}
                        for idx, (label, array_mlmrylyr) in enumerate(zip(labels, maske)):
                            if array_mlmrylyr == [0, 0] or []:
                                print(f"{label} ist leer.")
                            else:
                                print(f"{label} hat Daten.")
                                for mask in array_mlmrylyr:
                                    black_img = np.zeros_like(img)
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
                                mymasks.append(black_img)

                            # print("mymasks:", mymasks)    # OK
                            print("mymasks:", len(mymasks))

                        # for i in range(PennFudanPed.shape[0]):
                        for i in range(99):
                            myleft = data['myleft'][0][i]
                            # myleft = np.asarray(myleft, dtype="object")
                            myright = data['myright'][0][i]
                            # myright = np.asarray(myright, dtype="object")
                            yourleft = data['yourleft'][0][i]
                            # yourleft = np.asarray(yourleft, dtype="object")
                            yourright = data['yourright'][0][i]
                            # yourright = np.asarray(yourright, dtype="object")

                            # if PennFudanPed['myleft'][0][i].size == 0:
                            #     PennFudanPed['myleft'][0][i] = np.array([])
                            # if PennFudanPed['myright'][0][i].size == 0:
                            #     PennFudanPed['myright'][0][i] = np.array([])
                            # if PennFudanPed['yourleft'][0][i].size == 0:
                            #     PennFudanPed['yourleft'][0][i] = np.array([])
                            # if PennFudanPed['yourright'][0][i].size == 0:
                            #     PennFudanPed['yourright'][0][i] = np.array([])

                        print("Myleft values:",
                              data['myleft'][0][i].size)  # Weil es sich nur um eine DImenstion handel 0
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
                        #          enumerate(zip(myleft, myright, yourleft, yourright))}

                        # if len(self.myleft) == len(self.myright) == len(self.yourleft) == len(self.yourright):

                        print("maske:", maske)
                        print(type(maske))
                        print("shapemaske:", maske.shape)
                        # masks.extend(mymasks)  # makes no structured or ensted array, but an array

        # print(f"gesamtanzahl masks: {len(masks)}")  # //2485
        return mymasks


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
        # img = read_image(imgsizepath)
        self.img = self.imgs[idx]
        self.mymasks = self.extract_polygons()
        drawn_masks = []
        # img = imgs[idx]
        for mask in self.mymasks:
            # # instances are encoded as different colors
            obj_ids = torch.unique(mask)
            # # first id is the background, so remove it
            obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        # obj_ids is a 1D tensor that contains the unique identifiers (IDs)
        # for each object class present in the color-encoded mask. The [:, None, None] operation
        # reshapes obj_ids from shape (N,) (where N is the number of object classes) to shape (N, 1, 1).
        # This allows for broadcasting when comparing against the mask.

        # result is a boolean tensor where:Each pixel in the mask is checked against each object ID. Resulting
        # Tensor has shape (N,H,W) where N indicateds boolean. vvv makes it to 0/1
        #     for obj_id in range(mask):
            masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
            # split the color-encoded mask into a set of boolean masks.
            # boxes = masks_to_boxes(masks)
            # drawn_masks.append(draw_segmentation_masks(img, mask, alpha=0.8, colors="blue"))
            # return drawn_masks
            return masks
        labels = torch.ones(4, dtype=torch.int64)

        # image_id = idx
        # Wrap sample and targets into torchvision tv_tensors:
        # maske = tv_tensors.Mask(sieveoutbackground())
        target = {}
        target["masks"] = masks  # Ein Arr oder ein Objekt
        target["labels"] = labels
        # target["image_id"] = image_id

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        #
        # return img, target


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Things use in the original script
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros(4, dtype=torch.int64)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Parameters needed for RCNN training
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area # ?
        target["iscrowd"] = iscrowd

        # Return the image in PIL format or applied the transform
        if ((self.transforms is not None) and (self.get_image_PIL == False)):
            img = self.transforms(img)

        return img, target

path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
