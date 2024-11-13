import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import os
from numpy.lib.function_base import extract
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
        self.mymasks = []
        self.unterordner_liste = []
        self.transform = transform
        for dir in os.listdir(path):
            self.full_dir_path = os.path.join(path, dir)
            if os.path.isdir(self.full_dir_path):
                # print(f"Checking directory: {self.full_dir_path}")
                # Iteriere durch alle Dateien im Unterverzeichnis
                for file in os.listdir(self.full_dir_path):
                    # print(f"Found file: {file}")

                    if file.startswith('frame'):
                        self.imgs.append(cv2.imread(os.path.join(self.full_dir_path, file)))
                    else:
                        if file.startswith('polygons'):
                            self.full_path = os.path.join(self.full_dir_path, file)
                            # print("full_path:", self.full_path)
                            # print("hallooo")
                            mymasks, _ = self.extract_polygons(self.full_path)
                            self.mymasks = self.mymasks + mymasks
                            # print(len(self.mymasks))    # 0 ?!?!?!?
                            continue
                self.unterordner_liste.append(self.full_dir_path)

        # for idx, img in enumerate(self.imgs):
        #     if idx < 0 or idx >= len(self.mymasks):
        #         print(f"Index {idx} is out of range für len(self.maske): {len(self.mymasks)}")
        # print("lenimg:", len(self.imgs))    # 4800 OK

    def extract_polygons(self, full_path):
        color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}
        mymasks = []
        mat_data = scipy.io.loadmat(self.full_path)
        data = mat_data['polygons']
        data_transponed = data.T
        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                 zip(data_transponed['myleft'][0], data_transponed['myright'][0], data_transponed['yourleft'][0], data_transponed['yourright'][0])]
        # print("maske:", maske)

        # maske = torch.tensor(maske)
        labels = ['ml', 'mr', 'yl', 'yr']
        mapped_data = {}

        for idx, (label, array_mlmrylyr) in enumerate(zip(labels, maske)):
            # if array_mlmrylyr == [0, 0] or []:
                # print(f"{label} ist leer.")
            # else:
                # print(f"{label} hat Daten.")
            for self.mask in array_mlmrylyr:                     # Hier checken ob was drin ist, aber sind nicht-WErte nicht super schnell durchgefegt ?
                black_img = np.zeros_like(img)
                if self.mask.size > 0:
                    self.mask = np.array(self.mask, dtype=np.int32)  # CV will integer habenCV_32S
                    # Reshape to (N, 1, 2) for OpenCV
                    self.mask = self.mask.reshape((-1, 1, 2))
                    cv2.fillPoly(black_img, [self.mask], color[label])
                    self.masks = torch.tensor(self.mask)  # m_to_b braucht Namen maskS
                    boxes_vektor = masks_to_boxes(self.masks)
                mymasks.append(black_img)
        return mymasks, boxes_vektor

    def __len__(self):
        return len(self.imgs)

    def takeoneimgfromlist(self):
        for i in self.mymasks:
            if isinstance(i, np.ndarray):
                j = torch.from_numpy(i)
                # for j in i:             # Weil es struktiert war, stattdessen nimmt er verschachtelte Schleife !!! des Bilds!!!
                #     j = torch.from_numpy(j)
                    # print(type(j)) #<class 'torch.Tensor'>
                # print("mask.shape j:", j.shape)         # sollte 1280 x 720, haben! stattdessen 1280 x 3
                return j
            else:
                print("there is noinstance(i, np.ndarray")
    #     for i in self.mymasks:
    #         self.mask = torch.from_numpy(i)       # Umwandlung Array zum Tensor 720 1280 3
    #         print("das ist mask:", self.mask)
    #         for j in i:
    #         if j.size == 0:
    #             # return torch.tensor([])
    #             return [0, 0, 0, 0]
    #         return i

    def comparemasktoimg(self):
        # self.mask = self.mask.to(torch.int)
        # # instances are encoded as different colors
        obj_ids = torch.unique(self.blkimg)
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
        # for obj_id in range(self.mask):

        # split the color-encoded mask into a set of boolean masks.
        # boxes = masks_to_boxes(masks)
        # drawn_masks.append(draw_segmentation_masks(img, mask, alpha=0.8, colors="blue"))
        # return drawn_masks

        objectid = (self.blkimg == obj_ids[:, None, None]).to(dtype=torch.uint8)
        return objectid

    def crectangle(self, boxes_vektor):
        x0 = boxes_vektor[:, 0]
        y0 = boxes_vektor[:, 1]
        x1 = boxes_vektor[:, 2]
        y1 = boxes_vektor[:, 3]
        # Erstelle den Boxen-Tensor (N, 4) für (x0, y0, x1, y1) = Bound Box für eine Hand
        boxes_tensor = torch.stack((x0, y0, x1, y1), dim=1)
        boxes_tensor = list(boxes_tensor)
        return boxes_tensor

    def __getitem__(self, idx):
        img = self.imgs[idx]
        # If 'img' is a numpy array, convert it to a PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        ## Wrap sample and targets into torchvision tv_tensors:
        # img = tv_tensors.Image(img)
        mymasks, boxes_vektor = self.extract_polygons(self.full_dir_path)
        # print("mymasks:", mymasks)
        boxes_tensor = self.crectangle(boxes_vektor)
        self.blkimg = self.takeoneimgfromlist()
        print(self.blkimg.shape)

        champ = self.comparemasktoimg()
        labels = torch.ones(4, dtype=torch.int64)
        # Wrap sample and targets into torchvision tv_tensors:
        # maske = tv_tensors.Mask(sieveoutbackground())

        # print(type(mymasks), mymasks)
        # print(type(self.objectid), self.objectid.shape)

        # split the color-encoded mask into a set of binary masks
        # masks = (mymasks == objectid[:, None, None]).to(torch.uint8)
        #masks = torch.tensor(mymasks == objectid[:, None, None], dtype=torch.uint8)

        area = (champ[:, 3] - champ[:, 1]) * (champ[:, 2] - champ[:, 0])
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros(4, dtype=torch.int64)
        # Parameters needed for RCNN training
        target = {}
        # target["boxes_tensor"] = tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(1280,720))
        target["boxes"] = boxes_tensor

        # target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))

        target["masks"] = tv_tensors.Mask(self.mask)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            img = self.transform(img)

        return img, target

imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
img = cv2.imread(imgsizepath)
print("img.shape):", img.shape)
black_img = np.zeros_like(img)  # HIER bitte 1280, 720

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
path2 = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat"))

data_train = Mydataset(path, transform=transform)
data_test = Mydataset(path, transform=transform)


dataloader_train = DataLoader(data_train, batch_size=4, shuffle=True, num_workers=8)
dataloader_test = DataLoader(data_test, batch_size=4, shuffle=False, num_workers=8)
