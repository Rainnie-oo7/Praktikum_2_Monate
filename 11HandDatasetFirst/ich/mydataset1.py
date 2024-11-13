import scipy
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

"""
1. Mydataset: Lädt die Polygone und die dazugehörigen Bilder.
2. DataLoader: Dient zum Laden der Daten in Batches.
3. PolygonModel: Ein einfaches Modell, das die Polygondaten verarbeitet. Es verwendet eine lineare Schicht, um die Polygone zu transformieren.
"""

def extract_polygons(path):
    mat_data = scipy.io.loadmat(path)
    data = mat_data['polygons']
    # myleft = PennFudanPed['myleft'][0][0]
    # myright = PennFudanPed['myright'][0][0]
    # yourleft = PennFudanPed['yourleft'][0][0]
    # yourright = PennFudanPed['yourright'][0][0]
    myleft = data[0][0][0]
    myright = data[0][1][0]
    yourleft = data[0][2][0]
    yourright = data[0][3][0]
    maskso = [[ml,mr,yl,yr] for ml, mr, yl, yr in zip(myleft,myright,yourleft,yourright)]
    return maskso

def polygon_to_tensor(polygon):
    return torch.tensor(polygon, dtype=torch.float32)

class Mydataset(Dataset):
    def __init__(self, path, transform=None):
        self.imgs = []
        self.masks = []
        self.maskspoly = self.masks
        self.transforms = transform

        for dir in os.listdir(path):
            for file in os.listdir(os.path.join(path, dir)):
                if file.startswith('frame'):
                    #_
                    #muss hier nicht noch transformiert, etwa geblurrt, verpixeln gewerden und in ndArray/Tensor umgewandelt?
                    #~
                    self.imgs.append(cv2.imread(os.path.join(path, dir, file)))
                else:
                    self.masks = self.masks + extract_polygons(os.path.join(path, dir, file))

            break

    def __len__(self):
        return len(self.imgs)
        pass



    def __getitem__(self, idx):
        mask_list = []
        img = self.imgs[idx]
        img_height, img_width, _ = img.shape

        black_img = np.zeros_like(img)

        for idx, img in enumerate(self.imgs):
            if idx < 0 or idx >= len(self.maskspoly):
                print(f"Index {idx} is out of range for maskspoly with length {len(self.maskspoly)}")
            else:
                for mask in self.maskspoly[idx]:
                    if mask.size > 0:
                        # poly = []
                        # for p in poly:
                        #     if len(p) > 0:
                        print("Das ist mask:", mask)
                        # mask = np.concatenate((mask, np.array(mask, dtype=np.int32).reshape((-1, 1, 2))), axis=0)
                        mask_list = np.append(mask, np.array(mask, dtype=np.int32).reshape((-1, 1, 2)))
                        # # Konvertiere das Polygon in einen Tensor
                        # polygon = polygon_to_tensor(self.maskspoly[idx])
                        #
                        # cv2.fillPoly(black_img, self.maskspoly, color=(255, 0, 0))
                        print("mask_list:", mask_list)
                        print("lenmask_list:", len(mask_list))
                        print("typemask_list:", type(mask_list))

                        return mask_list
        return img, black_img
# poly = [np.array(p, dtype = np.int32).reshape((-1, 1, 2)) for p in masks if len(p) > 0]
#

