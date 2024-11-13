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
def extract_col(data):
    for row in data:
        myleft = row['myleft']
        myright = row['myright']
        yourleft = row['yourleft']
        yourright = row['yourright']
        return myleft, myright, yourright, yourleft

def draw_polygons(column, color, alpha):
        if isinstance(column,  np.ndarray):
            for coord in column:
                if coord.size > 0:
                    coord = np.array(coord)
                    if coord.ndim == 2 and coord.shape[1] == 2:
                        plt.fill(coord[:, 0], coord[:, 1], color=color, alpha=alpha)
                    elif coord.ndim == 1 and len(coord) == 2:
                        plt.fill(coord[0], coord[1], color=color, alpha=alpha)
            return [np.array(item) for item in column]
        # return []


parent_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/'
if __name__ == '__main__':
    data = getpolygons()
    # myleft, myright, yourleft, yourright = draw_polygons(PennFudanPed)
    column = extract_col(data)

    draw = draw_polygons(column, 'black', 0.5)
    # plt.xlim(0, 10)  # Beispielgrenzen für die x-Achse
    # plt.ylim(0, 10)  # Beispielgrenzen für die y-Achse
    plt.axis('off')  # Achsen ausblenden
    plt.show()  # Anzeige der Zeichnung