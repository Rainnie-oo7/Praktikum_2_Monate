import os
import fnmatch
from os.path import dirname
import torch.utils
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Coordinatesdraw:
    def __init__(self, parent_folder):
        for subdir, dirs, files in os.walk(parent_folder):
            for file in files:
                if file == 'polygons.mat':
                    mat_file_path = os.path.join(subdir, file)
                    print(f"Polygon-Datei gefunden in {subdir}")
                    self.mat_data = scipy.io.loadmat(mat_file_path)
                    self.data = self.mat_data['polygons']
                    # print(self.PennFudanPed['myleft'].shape)

                    if self.mat_data['polygons'] is None:
                        print(f"Kein 'polygons' key gefundne in {mat_file_path}")
                        continue
                    self.num_polygons = self.data.shape[1]
                    print(f"({self.num_polygons} Polygone gefunden in {mat_file_path}")

                    self.image_files = sorted([f for f in files if f.startswith('frame')])
                    self.image_count = len(self.image_files)
                    if self.image_count == 100:
                        print(f"100 Bilder in {subdir} sind werden verarbeitet/vorhanden.")
                        for i, image_file in enumerate(self.image_files):
                            self.image_path = os.path.join(subdir, image_file)
                            if i == 0:  # Nur für das erste Bild
                                self.img = Image.open(self.image_path)
                                img_width, img_height = self.img.size
                                print("width and height for the first image: ", img_width, img_height)
                            for i in range(100):
                                if i < self.num_polygons:
                                    self.maske = self.data[0][i]
                                    print(f"Nutzung der Polygondaten: {self.maske} für das Bild {image_file}")
                                    self.myleft = self.maske['myleft']
                                    self.myright = self.maske['myright']
                                    self.yourleft = self.maske['yourleft']
                                    self.yourright = self.maske['yourright']
                                else:
                                    print(f"Keine Polygondaten für das Bild {image_file} gefunden")
                    else:
                        print(f"Es sind widererwartend keine 100 Bilder, sondern {self.image_count} in {subdir}")

        # Extract coordinates und draw alles innerhalb __init__
        self.run()

    def __init__(self, path):
        self.imgs = []
        self.masks = []
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def extract_coordinates(self, cell):
        # Extracts coordinates from a cell array if it's a numpy array
        if isinstance(cell, np.ndarray):
            return [np.array(item) for item in cell]
        return []

    # def flip_coordinates(self, coords, height):
    #     flipped_coords = []
    #     for coord_set in coords:
    #         if coord_set.ndim == 2 and coord_set.shape[1] == 2:
    #             flipped_set = np.array([
    #                 [x, height - y] for x, y in coord_set
    #             ])
    #         elif coord_set.ndim == 1 and len(coord_set) == 2:
    #             flipped_set = np.array([
    #                 [coord_set[0], height - coord_set[1]]
    #             ])
    #         else:
    #             flipped_set = np.array([])  # Handle unexpected shapes
    #         flipped_coords.append(flipped_set)
    #     return flipped_coords

    def plot_coordinates(self, coords, label, color, marker):
        for coord in coords:
            if coord.size > 0:
                if coord.ndim == 2 and coord.shape[1] == 2:
                    plt.scatter(coord[:, 0], coord[:, 1], color=color, marker=marker, label=label)
                elif coord.ndim == 1 and len(coord) == 2:
                    plt.scatter(coord[0], coord[1], color=color, marker=marker, label=label)

    def plot_image_and_coordinates(self):

        # Bild vertikal *NEINund horizontal* spiegeln
        # img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)   # Würde nochmal Horiz spiegeln, wir korrigieren nur den durch: vertik spiegeln durch  plt.gca().invert_yaxis()
        # img_flipped = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        # img_width, img_height = self.img.size
        # print("width and height: ", img_width, img_height)

        # Extract coordinates
        myleft_coords = self.extract_coordinates(self.myleft)
        myright_coords = self.extract_coordinates(self.myright)
        yourleft_coords = self.extract_coordinates(self.yourleft)
        yourright_coords = self.extract_coordinates(self.yourright)
        # Inverse/flip coordinates so that they're appearing righteous to the flippedIMAGE and inverted yAchse
        # flipped_myleft_coords = self.flip_coordinates(myleft_coords, img_height)
        # flipped_myright_coords = self.flip_coordinates(myright_coords, img_height)
        # flipped_yourleft_coords = self.flip_coordinates(yourleft_coords, img_height)
        # flipped_yourright_coords = self.flip_coordinates(yourright_coords, img_height)

        # Plot the image and coordinates
        plt.figure(figsize=(10, 8))
        plt.imshow(self.img)
        plt.axis('on')
        self.plot_coordinates(myleft_coords, 'My Left', 'red', 'o')
        self.plot_coordinates(myright_coords, 'My Right', 'blue', 's')
        self.plot_coordinates(yourleft_coords, 'Your Left', 'green', '^')
        self.plot_coordinates(yourright_coords, 'Your Right', 'purple', 'x')

        # plt.gca().invert_yaxis()  # Invert y-axis damit image flipping passt
        plt.legend()
        plt.show()

    def run(self):
        # This method executes all the necessary steps in order
        self.plot_image_and_coordinates()


# Example usage:
coordinatesdraw = Coordinatesdraw('/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/')


