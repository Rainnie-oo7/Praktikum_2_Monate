import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Flip:
    def __init__(self, img_path, mat_path):
        self.img = Image.open(img_path)
        self.mat_data = scipy.io.loadmat(mat_path)
        self.data = self.mat_data['polygons']
        print(self.mat_data.keys())


        # for i in range(len(self.PennFudanPed)):
        self.myleft = self.data['myleft'][0, 0]
        self.myright = self.data['myright'][0, 0]
        self.yourleft = self.data['yourleft'][0, 0]
        self.yourright = self.data['yourright'][0, 0]
        # Extract coordinates and perform flipping inside __init__
        self.run()

    def extract_coordinates(self, cell):
        # Extracts coordinates from a cell array if it's a numpy array
        if isinstance(cell, np.ndarray):
            return [np.array(item) for item in cell]
        return []

    def flip_coordinates(self, coords, height):
        flipped_coords = []
        for coord_set in coords:
            if coord_set.ndim == 2 and coord_set.shape[1] == 2:
                flipped_set = np.array([
                    [x, height - y] for x, y in coord_set
                ])
            elif coord_set.ndim == 1 and len(coord_set) == 2:
                flipped_set = np.array([
                    [coord_set[0], height - coord_set[1]]
                ])
            else:
                flipped_set = np.array([])  # Handle unexpected shapes
            flipped_coords.append(flipped_set)
        return flipped_coords

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
        img_flipped = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        img_width, img_height = img_flipped.size
        print("width and height: ", img_width, img_height)

        # Extract coordinates
        myleft_coords = self.extract_coordinates(self.myleft)
        myright_coords = self.extract_coordinates(self.myright)
        yourleft_coords = self.extract_coordinates(self.yourleft)
        yourright_coords = self.extract_coordinates(self.yourright)
        # Inverse/flip coordinates so that they're appearing righteous to the flippedIMAGE and inverted yAchse
        flipped_myleft_coords = self.flip_coordinates(myleft_coords, img_height)
        flipped_myright_coords = self.flip_coordinates(myright_coords, img_height)
        flipped_yourleft_coords = self.flip_coordinates(yourleft_coords, img_height)
        flipped_yourright_coords = self.flip_coordinates(yourright_coords, img_height)

        # Plot the image and coordinates
        plt.figure(figsize=(10, 8))
        plt.imshow(img_flipped)
        plt.axis('on')
        self.plot_coordinates(flipped_myleft_coords, 'My Left', 'red', 'o')
        self.plot_coordinates(flipped_myright_coords, 'My Right', 'blue', 's')
        self.plot_coordinates(flipped_yourleft_coords, 'Your Left', 'green', '^')
        self.plot_coordinates(flipped_yourright_coords, 'Your Right', 'purple', 'x')

        plt.gca().invert_yaxis()  # Invert y-axis damit image flipping passt
        plt.legend()
        plt.show()

    def run(self):
        # This method executes all the necessary steps in order
        self.plot_image_and_coordinates()


# Example usage:
flip_instance = Flip('/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg',
                     '/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')


