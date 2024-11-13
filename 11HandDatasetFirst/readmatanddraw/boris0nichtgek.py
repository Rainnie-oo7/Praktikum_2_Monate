import scipy.io
import scipy
import os.path as osp
import numpy as np
from fontTools.merge.util import first
import matplotlib.pyplot as plt
import os
import torch
from skimage import io, transform
from skimage.draw import polygon
from snntorch.spikegen import dtype
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

path = os.path.join('/boris.grillborzer/11HandDatasetFirst/PennFudanPed/')
imagepath = os.path.join('/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T')
image = os.path.join('/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg')
img = io.imread(image)
dateiname = 'ergebnisse.txt'
def getmeta():
    width = img.shape[1]
    height = img.shape[0]
    print("das ist width: ", width)
    print("das ist height: ", height)
    return width, height

def getfolderlist():           # Hier alles in die getpolygonpoints for Schleife
    frames_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("frame"):
                frames_list.append(file)
    return frames_list
    # print(frames_list)
def getpolfilelist():
    polygons_list = []
    for root, dirs, files in os.walk(path):
        for matfile in files:
            if matfile.startswith("polygons"):
                polygons_list.append(matfile)
                # print(frames_list)
    return polygons_list

def getpolygonpoints(frames_list, polygons_list):
    # Listen für x- und y-Werte
    xmyleft = []
    ymyleft = []
    xmyright = []
    ymyright = []
    xyourleft = []
    yyourleft = []
    xyourright = []
    yyourright = []
    # path = osp.join(osp.dirname(__file__), "polygons2.mat")
    #
    for file in polygons_list:
        polygon = np.array(scipy.io.loadmat(file)['polygons'])
        dtype = [('myleft', 'f8'), ('myright', 'f8'), ('yourleft', 'f8'), ('yourright', 'f8')]
        polygon = np.array(polygon[2:], dtype=dtype)
        abschnitt = polygon[0]
        abschnitt_as_tuple = tuple(abschnitt)
        print("das ist abschnitt: ", abschnitt)
    # print("Getting tuples")
    # tup = []
    # myleft = abschnitt['myleft']
    # myright = abschnitt['myright']
    # yourleft = abschnitt['yourleft']
    # yourright = abschnitt['yourright']
    #     with open(dateiname, 'w') as datei:
            # first example ist Row
        counter = 0
        for row in polygon:
            for element in row:
                # print(type(element))

                counter += 1
                if counter == 1:
                    # print(f"Erstes Element: {element}")
                    # datei.write(f"Erstes Element: {element[2:]}")
                    element = np.array(element)
                    # print(f"Yooo Element: {element}")
                    # element = element[2:]
                    xmyleft, ymyleft = element.reshape(-1, 2).T    # Split Array Vertik
                    # for pair in element:
                    #     xmyleft.append(pair[0])
                    #     ymyleft.append(pair[1])
                    print(f"x-Wertemyleft: {xmyleft}")
                    print(f"y-Wertemyleft: {ymyleft}")
                    """
                elif counter == 2:
                    print(f"Zweites Element: {element}")
                    # datei.write(f"Zweites Element: {element}")
                    xmyright, ymyright = element.reshape(-1, 2).T
#                     for pair in element:
#                         xmyright.append(pair[0])
#                         ymyright.append(pair[1])
                    print(f"x-Wertemyright: {xmyright}")
                    print(f"y-Wertemyright: {ymyright}")
                elif counter == 3:
                    print(f"Drittes Element: {element}")
                    # datei.write(f"Drittes Element: {element}")
                    xyourleft, yyourleft = element.reshape(-1, 2).T
#                     for pair in element:
#                         xyourleft.append(pair[0])
#                         yyourleft.append(pair[1])
                    print(f"x-Werteyourleft: {xyourleft}")
                    print(f"y-Werteyourleft: {yyourleft}")
                elif counter == 4:
                    print(f"Viertes Element: {element}")
                    # datei.write(f"Viertes Element: {element}")
                    xyourright, yyourright = element.reshape(-1, 2).T
#                     for pair in element:
#                         xyourright.append(pair[0])
#                         yyourright.append(pair[1])
                    print(f"x-Werteyourright: {xyourright}")
                    print(f"y-Werteyourright: {yyourright}")
                if counter >= 4:
                    break
        return xmyleft, ymyleft, xmyright, ymyright, xyourleft, yyourleft, xyourright, yyourright

# print("New example:")
# myleft, myright, yourleft, yourright = row

# print(f"My left: {myleft.shape}")
# datei.write(f"\nMy left: {myleft.shape}")
# print(f"My right: {myright.shape}")
# datei.write(f"\nMy right: {myright.shape}")
# print(f"Your left: {yourleft.shape}")
# datei.write(f"\nYour left: {yourleft.shape}")
# print(f"Your right: {yourright.shape}")
# datei.write(f"\nYour right: {yourright.shape}")

# if yourright.size > 0:
            #     for co in np.nditer(yourright):
            #         print("Das ist Koordianten in yourright: ", co, end=' ')
            #         datei.write(f"\nDas ist Koordianten in yourright: {co} ")
            # else:
            #     print("Das Array ist leer.")
            #     datei.write("\nDas Array ist leer.")
            # print("Das ist Tupel: ")
            # datei.write(f"\nDas ist Tupel:")

            # for i, tu in enumerate(myleft):
            #     tupmyleft = tuple(map(tuple, myleft))
            #     print(f"\n{tupmyleft[i]}")
            #     datei.write(f"\n{tupmyleft[i]}")
            #     print(f"Ergebnisse wurden in {dateiname} gespeichert.")
            #     landmarksmyleft = np.asarray(tupmyleft, dtype=float).reshape(-1, 2)
            # print("Das sind die landmarks :", landmarksmyleft)
            # datei.write(f"\n{landmarksmyleft}")
#
#         for i, tu in enumerate(myright):
#             tupmyright = tuple(map(tuple, myright))
# #             print(f"\n{tupmyright[i]}")
#         #     datei.write(f"\n{tupmyright[i]}")
#         # print(f"Ergebnisse wurden in {dateiname} gespeichert.")
#             landmarksmyright = np.asarray(tupmyright, dtype=float).reshape(-1, 2)
#         # print("Das sind die landmarks :", landmarksmyright)
#           datei.write(f"\n{landmarksmyright}")

#         for j, tu in enumerate(yourleft):
#             tupyourleft = tuple(map(tuple, yourleft))
#             print(f"\n{tupyourleft[j]}")
#             datei.write(f"\n{tupyourleft[j]}")
#             landmarksyourleft = np.asarray(tupyourleft, dtype=float).reshape(-1, 2)
#         print("Das sind die landmarks :", landmarksyourleft)
#         datei.write(f"\n{landmarksyourleft}")
#         print(f"Ergebnisse wurden in {dateiname} gespeichert.")
#
#         for i, tu in enumerate(yourright):
#             tupyourright = tuple(map(tuple, yourright))
#             print(f"\n{tupyourright[i]}")
#             datei.write(f"\n{tupyourright[i]}")
#             landmarksyourright = np.asarray(tupyourright, dtype=float).reshape(-1, 2)
#         print("Das sind die landmarks :", landmarksyourright)
#         datei.write(f"\n{landmarksyourright}")
#         print(f"Ergebnisse wurden in {dateiname} gespeichert.")
# #         return landmarksmyleft, landmarksmyright, landmarksyourleft, landmarksyourright
#         return landmarksyourright, landmarksyourleft

def drawpolygonpoints(img, xmyleft, ymyleft, xmyright, ymyright, xyourleft, yyourleft, xyourright, yyourright):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(xmyleft, ymyleft, s=10, marker='.', c='r')
    ax.scatter(xmyright, ymyright, s=10, marker='.', c='r')
    ax.scatter(xyourleft, yyourleft, s=10, marker='.', c='r')
    ax.scatter(xyourright, yyourright, s=10, marker='.', c='r')
    # ax.invert_yaxis()
"""
if __name__ == '__main__':
    # width, height = getmeta()
    frames_list = getfolderlist()
    polygons_list = getpolfilelist()
    xmyleft, ymyleft, xmyright, ymyright, xyourleft, yyourleft, xyourright, yyourright = getpolygonpoints(frames_list, polygons_list)   # Könnte prblmtisch in einer anderen Reihenfolge sein
    drawpolygonpoints(img, xmyleft, ymyleft, xmyright, ymyright, xyourleft, yyourleft, xyourright, yyourright)

    plt.show()

    # landmarks = np.asarray(tup, dtype=float).reshape(-1, 2)
    # n = 65
    # img_name = landmarks_frame.iloc[n, 0]
    # landmarks = landmarks_frame.iloc[n, 1:]
    # landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    #
    # print('Image name: {}'.format(img_name))
    # print('Landmarks shape: {}'.format(landmarks.shape))
    # print('First 4 Landmarks: {}'.format(landmarks[:4]))
    # plt.imshow(image)

    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    # plt.pause()  # pause a bit so that plots are updated


    #     for i, myleft in enumerate(first_example):
    #         if i >= 1:                  # Stoppe nach den ersten Eintrag
    #             break
    #         print("MyLeft start:")
    #         if len(myleft) > 0:          #Kurz Check, sonst: IndexError: index 0 is out of bounds for axis 0 with size 0
    #             print(myleft[0])
    #         else:
    #             print("myleft ist leer.")
    #         # x, y = myleft[0]
    #         # print(f"Tupel MyLeft: ({x}, {y})")
    #     for i, myright in enumerate(first_example):
    #         if i >= 1:
    #             break
    #         print("MyRight start:")
    #         if len(myright) > 0:
    #             print(myright[0])
    #         else:
    #             print("myright ist leer.")
    #     for i, yourleft in enumerate(first_example):
    #         if i >= 1:
    #             break
    #         print("YourLeft start:")
    #         if len(yourleft) > 0:
    #             print(yourleft[0])
    #         else:
    #             print("yourleft ist leer.")
    #     for i, yourright in enumerate(first_example):
    #         if i >= 1:
    #             break
    #         print("YourRight start:")
    #         if len(yourright) > 0:
    #             print(yourright[0])
    #         else:
    #             print("yourright ist leer.")
    #
    # print("Dies ist yourright, erstes Sample (1/100)")
    # print(yourright)


"""
def printoutboxes(src, file_to_delete):
    # Durchlaufe alle Ordner und Unterordner

    for root, dirs, files in os.walk(src):
        # Prüfe, ob die Datei 'polygons2.mat' im aktuellen Ordner vorhanden ist
        if file_to_read in files:
            file_path = os.path.join(root, file_to_delete)
            # Lösche die Datei
            print(file_to_read)
            print(f"{file_to_read} geprinted aus: {file_path}")

# Pfad anpassen
source_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/09Hand-Tracking-Pytorch/RCNN_(egos_hand_dataset)/_LABELLED_SAMPLES'
file_to_read = 'polygons2.mat'

printoutboxes(source_folder, file_to_read)
"""