import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import scipy
import numpy as np

def extract_polygons(path):
    masks = []
    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            if file.startswith('polygons'):
                mat_data = scipy.io.loadmat(os.path.join(path, dir, file))
                data = mat_data['polygons']
                data = data[:0] #????
                print("das ist Data[:0]:", data)
                for i in range(data.shape[0]):
                    myleft, myright, yourleft, yourright = data[['myleft'], ['myright'], ['yourleft'], ['yourright']][i, 0]
                    # myright = PennFudanPed['myright'][i, 0]
                    # yourleft = PennFudanPed['yourleft'][i, 0]
                    # yourright = PennFudanPed['yourright'][i, 0]
                    print("Ml:", myleft, "mr:", myright, "yl:", yourleft, "yr:", yourright)
                    maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in zip(myleft, myright, yourleft, yourright)]
                    print(type(maske))
                    masks = masks + maske
                print(len(masks))  # // 19???
                return masks
        break

    # Iteriere zeilenweise durch die Daten
    # for row in PennFudanPed[0][0].T:
    #     for r in row:
    #         print(r)
    #     for i in row:
            # print("das ist i", i)
            # myleft = PennFudanPed['myleft'][0, i]
            # print("das ist myl", myleft)
            # myright = PennFudanPed['myright'][0, i]
            # yourleft = PennFudanPed['yourleft'][0, i]
            # yourright = PennFudanPed['yourright'][0, i]
            # return myleft, myright, yourleft, yourright

class Mydataset(Dataset):
    def __init__(self, path):
        self.imgs = []
        self.masks = []

        for dir in os.listdir(path):
            for file in os.listdir(os.path.join(path, dir)):
                if file.startswith('frame'):
                    # _
                    # muss hier nicht noch transformiert, etwa geblurrt, verpixeln gewerden und in ndArray/Tensor umgewandelt?
                    # ~
                    self.imgs.append(cv2.imread(os.path.join(path, dir, file)))
                else:
                    self.masks = self.masks + extract_polygons(os.path.join(path, dir, file))
                    print(len(self.masks)) #// 19???
            break
        # self.run()
    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        self.mask_list = []
        for dir in os.listdir(path):
            for file in os.listdir(os.path.join(path, dir)):
                if file.startswith('poly'):
                    self.masks = extract_polygons()
        # self.maskspoly = self.masks
        img = self.imgs[idx]
        img_height, img_width, _ = img.shape
        black_img = np.zeros_like(img)
        for idx, img in enumerate(self.imgs):
            if idx < 0 or idx > len(self.masks):
                print(f"Index {idx} is out of range für len(self.maske): {len(self.masks)}")
            else:
                 for mask in self.masks[idx]:
                     if mask.size > 0:

                         print("Das ist mask:", mask)
                         self.mask_list = np.append(mask, np.array(mask, dtype=np.int32).reshape((-1, 1, 2)))
                         self.polygon = self.mask_list[idx]
                         polygon_tensor = torch.tensor(self.polygon, dtype=torch.float32)
                         # Genau zwei Werte zurückgeben The only specificity that we require is that the dataset __getitem__ should return a tuple:

                         return img, polygon_tensor

def run(self):
    # This method executes all the necessary steps in order
    self.extract_polygons()


if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
    maske = extract_polygons(path)
    dataset = Mydataset(path)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # process = train()
