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
    color = {'myleft': (255, 255, 0), 'myright': (255, 0, 255), 'yourleft': (0, 255, 255), 'yourright': (0, 0, 255)}
    mymasks = []
    mat_data = scipy.io.loadmat(file)
    data = mat_data['polygons']

    if isinstance(data['myleft'], np.ndarray):
        length = data['myleft'].shape[1]  # Wenn es ein 2D-Array ist
        print("length = PennFudanPed['myleft'].shape[1]", length)

    else:
        length = len(data['myleft'])  # Wenn es eine Liste ist


    # Es geht: idx0 Label1, Zellen-Array, Label2, Zellen-Array, Label3, Zellen-Array, Label4, Zellen-Array | idx1 ..
    for idx in range(99):
        if idx < 0 or idx >= length:  # Beispiel: 2D-Array
            raise IndexError(f"Index {idx} ist außerhalb des gültigen Bereichs.")
        mask_data = {
            'myleft': data['myleft'][0][idx],
            'myright': data['myright'][0][idx],
            'yourleft': data['yourleft'][0][idx],
            'yourright': data['yourright'][0][idx]
        }
        print("typemask_data", type(mask_data))

    # Schwarzes Bild für Maske erstellen
    black_img = np.zeros_like(img)

    # Für jedes Label und die zugehörige Maske
    for label, polygon_coords in mask_data.items():
        if polygon_coords.size > 0:  # Überprüfen, ob die Maske existiert
            polygon_coords = np.array(polygon_coords, dtype=np.int32)  # Koordinaten in Ganzzahl umwandeln
            polygon_coords = polygon_coords.reshape((-1, 1, 2))  # Umformen für OpenCV
            print("lenpolygon_coords", len(polygon_coords))
            print("typepolygon_coords", type(polygon_coords))
            print("lenpolygon_coords", len(polygon_coords))

            # Maske auf schwarzem Bild einzeichnen
            cv2.fillPoly(black_img, [polygon_coords], color[label])
            mymasks.append(black_img.copy())
            print("lenmymasks", len(mymasks))
        else:
            print(f"{label} ist leer.")

    print(f"mymasks für idx {idx}:", len(mymasks))

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

