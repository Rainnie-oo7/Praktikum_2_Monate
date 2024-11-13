import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle   # Hieraus DUMP

class Flip:
    def __init__(self, img_flipped):
        self.img = Image.open(
            '/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg')
        self.mat_data = scipy.io.loadmat(
            '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')
        self.data = self.mat_data['polygons']
        print(self.mat_data.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'polygons'])

        # Zugriff auf die verschiedenen Spaltennamen
        myleft = self.data['myleft'][0, 0]
        myright = self.data['myright'][0, 0]
        yourleft = self.data['yourleft'][0, 0]
        yourright = self.data['yourright'][0, 0]

        myleft_coords = self.extract_coordinates(myleft)
        myright_coords = self.extract_coordinates(myright)
        yourleft_coords = self.extract_coordinates(yourleft)
        yourright_coords = self.extract_coordinates(yourright)

        # Bild vertikal und horizontal spiegeln
        # img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)   # Würde nochmal Horiz spiegeln, wir korrigieren nur den durch: vertik spiegeln durch  plt.gca().invert_yaxis()
        img_flipped = self.img.transpose(Image.FLIP_TOP_BOTTOM)

        # Bildgrößen ermitteln
        img_width, img_height = img_flipped.size
        print("width und height: ", img_width, img_height)

        flipped_myleft_coords = self.flip_coordinates(myleft_coords, img_height)
        flipped_myright_coords = self.flip_coordinates(myright_coords, img_height)
        flipped_yourleft_coords = self.flip_coordinates(yourleft_coords, img_height)
        flipped_yourright_coords = self.flip_coordinates(yourright_coords, img_height)

        # Schritt 4: Plotten des Bildes und der Koordinaten
        plt.figure(figsize=(10, 8))
        plt.imshow(img_flipped)
        plt.axis('on')  # Achsen ausblenden, um nur das Bild zu zeigen

        # Koordinaten plotten
        self.plot_coordinates(flipped_myleft_coords, 'My Left', 'red', 'o')
        self.plot_coordinates(flipped_myright_coords, 'My Right', 'blue', 's')
        self.plot_coordinates(flipped_yourleft_coords, 'Your Left', 'green', '^')
        self.plot_coordinates(flipped_yourright_coords, 'Your Right', 'purple', 'x')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    # Funktion zum Extrahieren von Koordinaten
    def extract_coordinates(self, cell):
        if isinstance(cell, np.ndarray):
            return [np.array(item) for item in cell]
        return []

    # Schritt 3: Koordinaten anpassen
    def flip_coordinates(self, coords, height):
        flipped_coords = []
        for coord_set in coords:
            if coord_set.ndim == 2 and coord_set.shape[1] == 2:
                # Bei 2D Koordinaten
                flipped_set = np.array([
                    [x, height - y] for x, y in coord_set
                ])
            elif coord_set.ndim == 1 and len(coord_set) == 2:
                # Bei 1D Koordinaten (einzelner Punkt)
                flipped_set = np.array([
                    [coord_set[0], height - coord_set[1]]
                ])
            else:
                # Bei unerwarteten Formen einfach leere Koordinaten anfügen
                flipped_set = np.array([])
            flipped_coords.append(flipped_set)
        return flipped_coords

    # Funktion zum Plotten der Koordinaten
    def plot_coordinates(self, coords, label, color, marker):
        for coord in coords:
            if coord.size > 0:
                coord = np.array(coord)
                if coord.ndim == 2 and coord.shape[1] == 2:
                    plt.scatter(coord[:, 0], coord[:, 1], color=color, marker=marker, label=label)
                elif coord.ndim == 1 and len(coord) == 2:
                    plt.scatter(coord[0], coord[1], color=color, marker=marker, label=label)


