import cv2
import os
import os.path as osp
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def extract_polygons(file):
    imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
    img = cv2.imread(imgsizepath)

    color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}
    mymasks = []

    mat_data = scipy.io.loadmat(file)
    data = mat_data['polygons']
    maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
             zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
    labels = ['ml', 'mr', 'yl', 'yr']
    # print(f"{maske} hat maske.")

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
                    black_img = np.zeros_like(img)
                    print("coloridx:", color[label])
                    cv2.fillPoly(black_img, [mask], color[label])
                    print("idx2:", idx)

            mymasks.append(black_img)

        # print("mymasks:", mymasks)    # OK
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
#War zu lang also weg-comment
    # print("maske:", maske)
    # print(type(maske))    #<class 'list'>
    # print("shapemaske:", maske.shape)
    # masks.extend(mymasks)  # makes no structured or ensted array, but an array

    # print(f"gesamtanzahl masks: {len(masks)}")  # //2485
    return mymasks

def matplot(mymasks):
    for pic in mymasks:
        print("typepic", type(pic))
        plt.imshow(pic)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    aktueller_pfad = os.getcwd()
    path = osp.normpath(aktueller_pfad)
    for file in os.listdir(path):
        if file.startswith('polygons'):
            # self.full_path = os.path.join(self.full_dir_path, file)
            # print("full_path:", self.full_path)
            boxes = extract_polygons(file)
            matplot(boxes)

