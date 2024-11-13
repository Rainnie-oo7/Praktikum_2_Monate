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
        mat_data = scipy.io.loadmat(self.full_path)
        data = mat_data['polygons']
        maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                 zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
        ml = []
        mr = []
        yl = []
        yr = []
        count = 0
        for idx, coords in enumerate(maske):
            for label, self.mask in enumerate(coords):

                for idx, coords in enumerate(maske):  # Iteriere über die Masken
                    # Du gehst davon aus, dass coords 4 Koordinatenpaare enthält (ml, mr, yl, yr)
                    ml.append(coords[0])  # Füge das erste Paar in ml hinzu
                    mr.append(coords[1])
                    yl.append(coords[2])
                    yr.append(coords[3])

                # print("ML:", ml)
                # print("MR:", mr)
                # print("YL:", yl)
                # print("YR:", yr)
                # print("mllen:", len(ml))

                return ml, mr, yl, yr

    def extract_listgon(self):
        ml, mr, yl, yr = self.extract_polygons(self.full_path)
        # print("ML:", ml)
        # print("MR:", mr)
        # print("YL:", yl)
        # print("YR:", yr)
        # print("mllen:", len(ml))
        x_listeml = []
        y_listeml = []
        x_listemr = []
        y_listemr = []
        x_listeyl = []
        y_listeyl = []
        x_listeyr = []
        y_listeyr = []
        # print("mlboris", ml)
        for array1 in ml: #100 Stk. array1
            for i in array1:
                # print("boris", i)
                if len(i) != 0:
                    x_listeml.append(i[0])
                    y_listeml.append(i[1])
                else:
                    break
        else:
            print("array1 hat nicht genug Elemente. Es hat:", len(array1))

        # print("x_listeml:", x_listeml)
        # print("y_listeml:", y_listeml)
        #
        # print("mrboris", mr)
        for array2 in mr:  # 100 Stk. array1
            for i in array2:
                # print("boris", i)
                if len(i) != 0:
                    x_listemr.append(i[0])
                    y_listemr.append(i[1])
                else:
                    break
        else:
            print("array1 hat nicht genug Elemente. Es hat:", len(array2))
        #
        # print("x_listemr:", x_listemr)
        # print("y_listemr:", y_listemr)

        # print("ylboris", yl)
        for array3 in yl:  # 100 Stk. array1
            for i in array3:
                # print("boris", i)
                if len(i) != 0:
                    x_listeyl.append(i[0])
                    y_listeyl.append(i[1])
                else:
                    break
        else:
            print("array1 hat nicht genug Elemente. Es hat:", len(array3))
        #
        # print("x_listeyl:", x_listeyl)
        # print("y_listeyl:", y_listeyl)
        #
        # print("mlboris", yr)
        for array4 in yr:  # 100 Stk. array1
            for i in array4:
                # print("boris", i)
                if len(i) != 0:
                    x_listeyr.append(i[0])
                    y_listeyr.append(i[1])
                else:
                    break
        else:
            print("array1 hat nicht genug Elemente. Es hat:", len(array4))
        #
        # print("x_listeyr:", x_listeyr)
        # print("y_listeyr:", y_listeyr)

        """
        # for array1 in ml:
        #     print("Aktuelles array1:", array1)  # Ausgabe des gesamten Arrays
        #     if isinstance(array1, list):
        #         for coords in array1:
        #             print("Aktuelles coords:", coords)  # Ausgabe jedes Elements (coords)
        #             if isinstance(coords, (list, tuple)) and len(coords) ==2:
        #                 print("coords", coords)
        #                 x_listeml.append(coords[0])  # x-Wert hinzufügen
        #                 y_listeml.append(coords[1])  # y-Wert hinzufügen
        #             else:
        #                 print("coords nicht gültig:", coords)
        #     else:
        #         print("array1 nicht gültig:", array1)
        # print("x_listeml:", x_listeml)
        # print("y_listeml:", y_listeml)

    
                        for array2 in mr:
                            if isinstance(array2, list):
                                for coords in array2:
                                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                                        print("coords", coords)
                                        x_listemr.append(coords[0])  # x-Wert hinzufügen
                                        y_listemr.append(coords[1])  # y-Wert hinzufügen
                        # print("x_listemr:", x_listemr)
                        # print("y_listemr:", y_listemr)
                
                        for array3 in yl:
                            if isinstance(array3, list):
                                for coords in array3:
                                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                                        print("coords", coords)
                                        x_listeyl.append(coords[0])  # x-Wert hinzufügen
                                        y_listeyl.append(coords[1])  # y-Wert hinzufügen
                        # print("x_listeyl:", x_listeyl)
                        # print("y_listeyl:", y_listeyl)
                
                        for array4 in yr:
                            if isinstance(array4, list):
                                for coords in array4:
                                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                                        print("coords", coords)
                                        x_listeyr.append(coords[0])  # x-Wert hinzufügen
                                        y_listeyr.append(coords[1])  # y-Wert hinzufügen
                        print("x_listeyr:", x_listeyr)
                        print("y_listeyr:", y_listeyr)
                
                        return x_listeml, y_listeml, x_listemr, y_listemr, x_listeyl, y_listeyl, x_listeyr, y_listeyr
                        """

        return x_listeml, y_listeml, x_listemr, y_listemr, x_listeyl, y_listeyl, x_listeyr, y_listeyr

    def gcornersmkbb(self):
        x_listeml, y_listeml, x_listemr, y_listemr, x_listeyl, y_listeyl, x_listeyr, y_listeyr = self.extract_listgon()
        targets = []
        labels = [0, 1, 2, 3]
        boxes_labels_dataml = []
        boxes_labels_datamr = []
        boxes_labels_datayl = []
        boxes_labels_datayr = []
        x_minml, x_maxml, y_minml, y_maxml = None, None, None, None
        x_minmr, x_maxmr, y_minmr, y_maxmr = None, None, None, None
        x_minyl, x_maxyl, y_minyl, y_maxyl = None, None, None, None
        x_minyr, x_maxyr, y_minyr, y_maxyr = None, None, None, None

        boxes_labels_data = {
            0: None,
            1: None,
            2: None,
            3: None
        }

        try:    #ML
            if len(x_listeml) != 0 and len(y_listeml) != 0:
                x_minml, x_maxml = min(x_listeml), max(x_listeml)
                y_minml, y_maxml = min(y_listeml), max(y_listeml)
                #my_dict.update({'city': 'New York', 'email': 'alice@example.com'})
            if len(x_listemr) != 0 and len(y_listemr) != 0:
                x_minmr, x_maxmr = min(x_listemr), max(x_listemr)
                y_minmr, y_maxmr = min(y_listemr), max(y_listemr)
            if len(x_listeyl) != 0 and len(y_listeyl) != 0:
                x_minyl, x_maxyl = min(x_listeyl), max(x_listeyl)
                y_minyl, y_maxyl = min(y_listeyl), max(y_listeyl)
            if len(x_listeyr) != 0 and len(y_listeyr) != 0:
                x_minyr, x_maxyr = min(x_listeyr), max(x_listeyr)
                y_minyr, y_maxyr = min(y_listeyr), max(y_listeyr)

            # Gibt die Liste mit Boxen und Labels zurück

            boxes_labels_data[0] = [x_minml, x_maxml, y_minml, y_maxml]
            boxes_labels_data[1] = [x_minmr, x_maxmr, y_minmr, y_maxmr]
            boxes_labels_data[2] = [x_minyl, x_maxyl, y_minyl, y_maxyl]
            boxes_labels_data[3] = [x_minyr, x_maxyr, y_minyr, y_maxyr]

        except:
            print("nicht richtiges Format von x_liste und y_listeml bsp")


        # print("boxes_labels_datafertig", boxes_labels_data)

#{0: [158.19072164948457, 725.6134020618556, 435.0567010309279, 718.7680412371134],
# 1: [486.7680412371135, 1104.3350515463917, 362.479381443299, 718.7680412371134],
# 2: [575.180412371134, 1110.9329896907216, 283.3041237113403, 630.3556701030928],
# 3: [298.0670103092784, 812.7061855670103, 229.20103092783515, 712.1701030927835]}
#
#muss so aussehen:


    #Wir brauchen aber:                         [x_minml, x_maxml, y_minml, y_maxml]
        # boxes_labels_data = [{'labels': 0, 'boxes': torch.tensor([x_minml, x_maxml, y_minml, y_maxml], dtype = torch.float32).to('cpu')},
        #              {'labels': 1, 'boxes': torch.tensor([x_minmr, x_maxmr, y_minmr, y_maxmr], dtype = torch.float32).to('cpu')},
        #              {'labels': 2, 'boxes': torch.tensor([x_minyl, x_maxyl, y_minyl, y_maxyl], dtype = torch.float32).to('cpu')},
        #              {'labels': 3, 'boxes': torch.tensor([x_minyr, x_maxyr, y_minyr, y_maxyr], dtype = torch.float32).to('cpu')}]
        # print("boxes_labels_data", boxes_labels_data)
    # #    for i in range(len(self.imgs)):
    #     target = {
    #         "boxes": torch.tensor([[x_minml, x_maxml, y_minml, y_maxml], [x_minmr, x_maxmr, y_minmr, y_maxmr], [ x_minyl, x_maxyl, y_minyl, y_maxyl], [x_minyr, x_maxyr, y_minyr, y_maxyr]], dtype=torch.float32, device='cpu'),
    #         "labels": torch.tensor([1,2,3,4], dtype=torch.float32, device='cpu'),
    #         "image_id": torch.tensor([i], dtype=torch.float32, device='cpu'),
    #     }
    #Wir brauchen aber:     #target['boxes'] ist Tensor (N, 4)
        boxes_labels_data = [{"boxes": torch.tensor([x_minml, x_maxml, y_minml, y_maxml], dtype=torch.float32),
                             "labels": torch.tensor([1], dtype=torch.int64)},
                             {"boxes": torch.tensor([x_minmr, x_maxmr, y_minmr, y_maxmr], dtype=torch.float32),
                             "labels": torch.tensor([2], dtype=torch.int64)},
                             {"boxes": torch.tensor([x_minyl, x_maxyl, y_minyl, y_maxyl], dtype=torch.float32),
                             "labels": torch.tensor([3], dtype=torch.int64)},
                             {"boxes": torch.tensor([x_minyr, x_maxyr, y_minyr, y_maxyr], dtype=torch.float32),
                             "labels": torch.tensor([4], dtype=torch.int64)}]

        # for elem in boxes_labels_data:
        #     if "boxes" in elem:
        #         # Konvertiere die Boxen in einen Tensor
        #         boxes_tensor = torch.tensor(elem["boxes"], dtype=torch.float32)  # Form: (N, 4)
        #         print("boxes_tensor", boxes_tensor)
        #         labels_tensor = torch.tensor(elem["labels"], dtype=torch.int64)  # Labels als Tensor
        #         print("labels_tensor", labels_tensor)
        #         # Füge die Boxen und Labels in ein Dictionary für das Target ein
        #         target = {
        #             "boxes": boxes_tensor,
        #             "labels": labels_tensor,
        #         }
        #
        #         targets.append(target)
        #     return targets
        # print("target", targets)
        # print("targettype:", type(targets))


        return boxes_labels_data
        # return x_minml, x_maxml, y_minml, y_maxml, x_minmr, x_maxmr, y_minmr, y_maxmr,  x_minyl, x_maxyl, y_minyl, y_maxyl, x_minyr, x_maxyr, y_minyr, y_maxyr

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        target = boxes[idx]
        min_val = img.min()
        max_val = img.max()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)      #damit es in Pillow-Format vorliegt, sonst schimpft 'transform'
        img = (img - min_val) / (max_val - min_val)
        print("img type0", type(img))
        # img = np.array(img, dtype=np.float32)
        # print("imgtype0", type(img))
        #
        # if np.issubdtype(img.dtype, np.floating):
        #     print("The array is a floating-point array.")
        # else:
        #     print("The array is not a floating-point array.")
        if self.transform:
            img = self.transform(img)
        print("img type1", type(img))

        image_tensor = torch.from_numpy(img)
        print("img type2", type(image_tensor))

        img = image_tensor.permute(2, 0, 1)
        print("img type3", type(img))

        if img.is_floating_point():
            print("The img is a floating-point tensor.")
        else:
            print("The img is not a floating-point tensor.")

        self.extract_polygons(self.full_path)
        targets = self.gcornersmkbb()
        print("hugo", targets)
        # for i, elem in enumerate(targets):
        #     print(f"targetboxesequiv[{i}]:", elem.get("boxes"))
        #     target = elem.get("boxes")
        #     print(f"targetboxetype[{i}]:", type(elem.get("boxes")))
        #     print("boxes device:", elem.get("boxes").device)
        #
        #     print("Boxes content:", target)  # Inhalt der Boxen
        #     print("Boxes type:", type(target))  # Typ der Boxen

        # for elem in boxes_labels_data:
        #     print(f"targetboxesequiv[]:", elem.get("boxes"))
        #     target = elem.get("boxes")
        #     print(f"targetboxetype[]:", type(elem.get("boxes")))
        #     print("boxes device:", elem.get("boxes").device)
        #
        #     print("Boxes content:", target)  # Inhalt der Boxen
        #     print("Boxes type:", type(target))  # Typ der Boxen

        # if isinstance(target['boxes'], dict):
        #     target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32, device=img.device)
        print("imgshape", img.shape)
        return img, targets

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