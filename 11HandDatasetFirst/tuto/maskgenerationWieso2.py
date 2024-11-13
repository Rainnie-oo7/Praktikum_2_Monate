import os
import fnmatch
from os.path import dirname
import torch.utils
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

num_polygons = 100
def getpolygons():
    for subdir, dirs, files in os.walk(parent_folder):
        for file in files:
            if file == 'polygons.mat':
                mat_file_path = os.path.join(subdir, file)
                print(f"Polygon-Datei gefunden in {subdir}")
                mat_data = scipy.io.loadmat(mat_file_path)
                data = mat_data['polygons']

                return data
# Funktion zum Extrahieren von Koordinaten
def extract_coordinates(cell):
    if isinstance(cell, np.ndarray):
        return [np.array(item) for item in cell]
    return []


def draw_polygons(data):
    for row in data:
        # Angenommen, jede Zeile ist ein Dictionary oder eine ähnliche Struktur
        myleft = row['myleft']
        myright = row['myright']
        yourleft = row['yourleft']
        yourright = row['yourright']
        return myleft, myright, yourleft, yourright

# Funktion zum Plotten der Koordinaten
def plot_coordinates(coords, color, alpha):
    # for coord in coords:
    #     if coord.size > 0:
    #         coord = np.array(coord)
    #         if coord.ndim == 2 and coord.shape[1] == 2:
    #             plt.fill(coord[:, 0], coord[:, 1], color=color, alpha=alpha)
    #         elif coord.ndim == 1 and len(coord) == 2:
    #             plt.fill(coord[0], coord[1], color=color, alpha=alpha)

        # Zeichnen der Polygone
    plt.fill(myleft[:, 0], myleft[:, 1], color='blue', alpha=0.5)
    plt.fill(myright[:, 0], myright[:, 1], color='red', alpha=0.5)
    plt.fill(yourleft[:, 0], yourleft[:, 1], color='green', alpha=0.5)
    plt.fill(yourright[:, 0], yourright[:, 1], color='black', alpha=0.5)
        # return myleft, myright, yourleft, yourright

    plt.xlim(0, 10)  # Beispielgrenzen für die x-Achse
    plt.ylim(0, 10)  # Beispielgrenzen für die y-Achse
    plt.axis('off')  # Achsen ausblenden
    plt.show()  # Anzeige der Zeichnung



parent_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/'
if __name__ == '__main__':
    data = getpolygons()

    # Zugriff auf die verschiedenen Spaltennamen
    myleft = data['myleft'][0, 0]
    myright = data['myright'][0, 0]
    yourleft = data['yourleft'][0, 0]
    yourright = data['yourright'][0, 0]
    # print("yourleft_coords1: ", yourleft)



    # print("myleft_coords: ", myleft_coords)
    # myleft, myright, yourleft, yourright = draw_polygons(PennFudanPed)
    myleft_coords = extract_coordinates(myleft)
    myright_coords = extract_coordinates(myright)
    yourleft_coords = extract_coordinates(yourleft)
    yourright_coords = extract_coordinates(yourright)
    plot_coordinates(myleft_coords, 'red', alpha=0.5)
    plot_coordinates(myright_coords, 'blue', alpha=0.5)
    plot_coordinates(yourleft_coords, 'green', alpha=0.5)
    plot_coordinates(yourright_coords, 'black', alpha=0.5)