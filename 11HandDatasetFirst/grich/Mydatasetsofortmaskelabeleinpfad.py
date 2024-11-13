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
                            blackimglist, _ = self.extract_polygons(self.full_path)
                            self.blackimglist = self.blackimglist + blackimglist
                            # print(len(self.blackimglist))    # 0 ?!?!?!?
                            continue
                self.unterordner_liste.append(self.full_dir_path)

        # for idx, img in enumerate(self.imgs):
        #     if idx < 0 or idx >= len(self.blackimglist):
        #         print(f"Index {idx} is out of range für len(self.maske): {len(self.blackimglist)}")
        # print("lenimg:", len(self.imgs))    # 4800 OK

    def extract_polygons(self, full_path):
        color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}
        blackimglist = []
        boxes_vektorlist = []
        mat_data = scipy.io.loadmat(self.full_path)
        data = mat_data['polygons']
        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                 zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
        # print("maske:", maske)

        # maske = torch.tensor(maske)
        labels = ['ml', 'mr', 'yl', 'yr']
        result = 0, 0, 0, 0
        xy_coords = {label: [] for label in labels}  # Dictionary, um x,y-Koordinaten für jede Maske zu speichern
        for idx, (label, array_mlmrylyr) in enumerate(zip(labels, maske)):
            # if array_mlmrylyr == [0, 0] or []:
                # print(f"{label} ist leer.")
            # else:
                # print(f"{label} hat Daten.")
            for self.mask in array_mlmrylyr:                     # Hier checken ob was drin ist, aber sind nicht-WErte nicht super schnell durchgefegt ?
                if self.mask.size > 0:
                    self.mask = np.array(self.mask, dtype=np.int32)  # CV will integer habenCV_32S
                    # Reshape to (N, 1, 2) for OpenCV
                    self.mask = self.mask.reshape((-1, 1, 2))
                    cv2.fillPoly(black_img, [self.mask], color[label])
                    xy_coords[label].extend(self.mask.reshape(-1, 2).tolist())
                    for key, coords in xy_coords.items():
                        if len(coords) == 4:
                            del coords[2:]
                            pass
                        boxes_tensor = np.array(self.crectangle(coords))
                        # print("boxes_tensor:", boxes_tensor)
                    self.masks = torch.tensor(self.mask)  # m_to_b braucht Namen maskS
                    boxes_vektorlist.append(self.masks)
                blackimglist.append(black_img)
        return blackimglist, boxes_tensor

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

    def crectangle(self, coords):
        x_list = []
        y_list = []
        """
        for label, coords in xy_coords.items():
            if coords:  #überprüfen ob Koordinatn vorhanden sind
                coords_array = np.array(coords)
                x_min, y_min = coords_array.min(axis=0)
                x_max, y_max = coords_array.max(axis=0)
                # print(f"{label} - Min (x, y): ({x_min}, {y_min}), Max (x, y): ({x_max}, {y_max})")
            else:
                print(f"{label} hat keine Koordinaten.")
        """
        # # Erstelle den Boxen-Tensor (N, 4) für (x0, y0, x1, y1) = Bound Box für eine Hand
        # boxes_tensor = torch.stack((x0, y0, x1, y1), dim=1)
        # boxes_tensor = masks_to_boxes(self.masks)
            # boxes_tensor = list(boxes_tensor)
            # print("Das ist boxes_tensor:", boxes_tensor)
        if coords:
            # print("lencoords:", len(coords))
            for coord in coords:
                if isinstance(coord, (list, tuple)) and len(coord) > 1:
                    x_list.append(coord[0])
                    y_list.append(coord[1])
                else:
                    print(f"Unerwartetes Format von coord: {coord}")
            # print("coord[0]", x_list)
            # print("coord[1]", y_list)

            # print("coord[0]:", coord[0])
            # print("coord[0]:", coord[0])
            try:
                if len(x_list) != 0 or len(y_list) != 0:     #Nur die ml (oder mr, yl, yr), die durch die Zelle vertreten sind
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    return x_min, x_max, y_min, y_max
            # print("x_min:", x_min)
            # print("y_min:", y_min)
            # print("x_max:", x_max)
            # print("y_max:", y_max)
            except:

                print(f"Unerwartet. Eckpunkt ist leer: x_min, x_max, y_min, y_max")
                # return None



    def __getitem__(self, idx):
        img = self.imgs[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)      #damit es in Pillow-Format vorliegt, sonst schimpft 'transform'

        # print("img:", img)
        # print("imgtype:", type(img)) # imgtype: <class 'PIL.Image.Image'>
        blackimglist,  boxes_tensor = self.extract_polygons(self.full_dir_path)

        # if len(result) == 4:
        #     pass
        # boxes_tensor = np.array(self.crectangle(coords))
        self.blkimg = self.takeoneimgfromlist()
        # boxes_tensor = torch.tensor(boxes_vektor)
        champ = self.comparemasktoimg()
        labels = torch.ones(4, dtype=torch.int64)
        # print(type(blackimglist), blackimglist)
        # print(type(self.objectid), self.objectid.shape)
        # split the color-encoded mask into a set of binary masks
        # masks = (blackimglist == objectid[:, None, None]).to(torch.uint8)
        #masks = torch.tensor(blackimglist == objectid[:, None, None], dtype=torch.uint8)
        area = (champ[:, 3] - champ[:, 1]) * (champ[:, 2] - champ[:, 0])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros(4, dtype=torch.int64)
        target = {}
        # target["boxes_tensor"] = tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(1280,720))
        target["boxes"] = boxes_tensor
        target["masks"] = tv_tensors.Mask(self.mask)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        print("target:", target)
        # print(f"boxes_tensor: {boxes_tensor}")
        # print(f"labels: {labels}")
        # print(f"target: {target}")
        if self.transform:
            img = self.transform(img)
        # print("Das ist target:", target)
        # Extrahiere die Werte des Dictionaries
        target_values = list(target.values())
        # Wandle die Maske (Ziel) in einen LongTensor für die Segmentierung um
        target = torch.tensor(target_values, dtype=torch.long)

        return img, target

# imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
# img = cv2.imread(imgsizepath)
# print("img.shape):", img.shape)

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
