import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from numpy.lib.function_base import extract
from skimage.draw import polygon
from torchvision import transforms
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import tv_tensors
import scipy
import numpy as np

def einlesen(path):
    imgs = []
    masks = []

    # image_size = (720, 1280)
    # masks = create_mask_image(maske, image_size)
    for idx, img in enumerate(imgs):
        if idx < 0 or idx > len(masks):
            print(f"Index {idx} is out of range für len(maske): {len(masks)}")
        else:
            for dir in os.listdir(path):
                full_dir_path = os.path.join(path, dir)
                if os.path.isdir(full_dir_path):
                    # Iteriere durch alle Dateien im Unterverzeichnis
                    for file in os.listdir(full_dir_path):
                        if file.startswith('frame'):
                            imgs.append(cv2.imread(os.path.join(full_dir_path, file)))

                        else:
                            masks = masks + extract_polygons(path)
                            # print(len(masks)) #// 19???


# def getimg():
#     img = imgs[idx]
def extract_polygons():
    color = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 255)]
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
                    print("data", data)
                    # print(PennFudanPed)
                    # print(PennFudanPed.shape[0]) # // 1, 1, 1, ..
                    # print("Das ist PennFudanPed:", PennFudanPed)
                    # If 'PennFudanPed' is a structured array, you may need to iterate over its rows
                    maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                             zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
                    # maske = torch.tensor(maske)

                    for idx, elem in enumerate(maske):
                        for mask in elem:
                            black_img = np.zeros_like(img)
                            if mask.size > 0:
                                cv2.fillPoly(black_img, mask, color[idx])
                        mymasks.append(black_img)
                        # mymasksIter = mymasks
                        print("mymasks:", mymasks)
                        # print("mymasks:", len(mymasks))

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
                    #          enumerate(zip(myleft, myright, yourleft, yourright))}

                    # if len(myleft) == len(myright) == len(yourleft) == len(yourright):

                    print("maske:", maske)
                    print(type(maske))
                    print("shapemaske:", maske.shape)
                    masks.extend(mymasks)  # makes no structured or ensted array, but an array

    # print(f"gesamtanzahl masks: {len(masks)}")  # //2485
    return mymasksIter, mymasks


path = osp.normpath(osp.join(osp.dirname(__file__), "data"))

if __name__ == '__main__':
    einlesen(path, transform=None)
