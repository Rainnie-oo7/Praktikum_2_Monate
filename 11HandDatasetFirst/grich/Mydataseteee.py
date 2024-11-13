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
        self.transforms = transforms
        if isinstance(path, str):
            for dir in os.listdir(path):
                self.full_dir_path = os.path.join(path, dir)
                if os.path.isdir(self.full_dir_path):
                    # print(f"Checking directory: {self.full_dir_path}")
                    for file in os.listdir(self.full_dir_path):
                        # print(f"Found file: {file}")
                        if file.startswith('frame'):
                            self.imgs.append(cv2.imread(os.path.join(self.full_dir_path, file)))
                            self.imgs = sorted(self.imgs)
                        else:
                            if file.startswith('polygons'):
                                self.full_path = os.path.join(self.full_dir_path, file)
                                # print("full_path:", self.full_path)
                                self.boxes = self.extract_polygons(self.full_path)
        else:
            print("Der Pfad muss ein String sein.")

#Was macht?
        # for idx, img in enumerate(self.imgs):
        #     if idx < 0 or idx > len(self.mymasks):
        #         print(f"Index {idx} is out of range für len(self.maske): {len(self.mymasks)}")

        print("lenimg:", len(self.imgs))
    # def getimg(self):
    #     self.img = self.imgs[idx]

    def extract_polygons(self, idx):
        imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
        img = cv2.imread(imgsizepath)
        black_img = np.zeros_like(img)
        color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}
        mymasks = []
        mat_data = scipy.io.loadmat(self.full_path)
        data = mat_data['polygons']


        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                 zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
        labels = ['ml', 'mr', 'yl', 'yr']
        for idx, (label, array_mlmrylyr) in enumerate(zip(labels, maske)):
            if array_mlmrylyr == [0, 0] or []:
                print(f"{label} ist leer.")
            else:
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
        return mymasks[idx]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
        # img = read_image(imgsizepath)
        img = self.imgs[idx]
        self.mymasks = self.extract_polygons(idx)
        masks = self.mymasks[idx]

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

        # target["image_id"] = image_id

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        #
        # return img, target




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

object = Mydataset(Dataset)
object.run()
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
