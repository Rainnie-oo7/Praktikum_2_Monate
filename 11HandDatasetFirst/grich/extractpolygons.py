import os.path as osp
import os
import cv2
import numpy as np
import scipy

def extract_polygons(path2):
    imgsizepath = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg"))
    img = cv2.imread(imgsizepath)
    color = {'ml': (255, 255, 0), 'mr': (255, 0, 255), 'yl': (0, 255, 255), 'yr': (0, 0, 255)}
    mymasks = []
    black_img = np.zeros_like(img)
    if isinstance(path2, (str, bytes, os.PathLike)):
        print("richtiger Type")
        if os.path.exists(path2) and os.path.isdir(path2):
            for file in os.listdir(path2):
                if file.startswith('polygons'):
                    print("Datei mit 'polygons':", file)
                    full_path = os.path.join(path2, file)
                    print("Pfad Datei mit 'polygons':", full_path)
                    print(file)
                    print(f"Verarbeite Datei: {path2}","/polygons.mat")
                    # Lade die .mat-Datei
                    # mat_data = scipy.io.loadmat(file)
                    mat_data = scipy.io.loadmat(full_path)
                    print("mat_data", mat_data)
                    data = mat_data['polygons']

                    maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in
                             zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
                    print("maske:", maske)
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
                                # continue
                                cv2.fillPoly(black_img, [mask], color[label])
                                # for x, y in mask:
                                #     print(f"{label}: x-coordinate: {x}, y-coordinate: {y}")
                                #     min_x = int(min(x))
                                #     max_x = int(max(x))
                                #     min_y = int(min(y))
                                #     max_y = int(max(y))
                                #     tupel1 = (min_x, min_y)
                                #     tupel2 = (max_x, max_y)
                                #     matrix = np.array([tupel1, tupel2])
                                #     print("das ist matrix:", matrix)
                                #     return matrix
                            mymasks.append(black_img)

                        # print("mymasks:", mymasks)    # OK
                        print("mymasks:", len(mymasks))

                    # print("maske:", maske)
                    print(type(maske))

            return mymasks

        print("mymasks:", len(mymasks))
    else:
        print("Not richtiger Type")
if __name__ == '__main__':
    path2 = osp.normpath((osp.join(osp.dirname(__file__), "PennFudanPed/CARDS_COURTYARD_B_T/")))
    extract_polygons(path2)