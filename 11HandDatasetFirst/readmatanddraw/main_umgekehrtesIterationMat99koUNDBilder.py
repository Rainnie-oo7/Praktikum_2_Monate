import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os.path as osp
import os
import cv2

def mkimgslist(path):
    imgs = []
    if isinstance(path, str):
        for dir in os.listdir(path):
            full_dir_path = os.path.join(path, dir)
            if os.path.isdir(full_dir_path):
                # print(f"Checking directory: {self.full_dir_path}")
                for file in os.listdir(full_dir_path):
                    if file.startswith('frame'):
                        imgs.append(file)

                # # Dann die Bilder in der sortierten Reihenfolge laden
                # for file in files:
                #     img_path = os.path.join(full_dir_path, file)
                #     imgs.append(img_path)   #   Ohne CV2-read
                    elif file.startswith('polygons'):
                        full_path = os.path.join(full_dir_path, file)
                        myleft, myright, yourleft, yourright = readdataout(file)

                imgs.sort()


        else:
            print("Der Pfad muss ein String sein.")

    # print("imgs", imgs)
    print("imgslen", len(imgs))
    return myleft, myright, yourleft, yourright, imgs

def readdataout(file):
    mat_data = scipy.io.loadmat(file)
    data = mat_data['polygons']
    arrays_to_modify = ['myleft', 'myright', 'yourleft', 'yourright']
    for i in range(99):  #Bis 100 Iterationen
        for array_name in arrays_to_modify:
            mask_data = {
                'myleft': data['myleft'][0][i],
                'myright': data['myright'][0][i],
                'yourleft': data['yourleft'][0][i],
                'yourright': data['yourright'][0][i]
            }
    return myleft, myright, yourleft, yourright

def gimgout(imgs):
    for i in imgs:
        imgcv = cv2.imread(i)
        return i, imgcv

def print_data_info(data, name):
    print(f"Info für {name}:")
    if isinstance(data, np.ndarray):
        print(f"Typ: {type(data)}")
        print(f"Shape: {data.shape}")
        if len(data) > 0:
            print(f"Erstes Element: {data[0]}")
            if isinstance(data[0], np.ndarray):
                print(f"Erste Form des ersten Elements: {data[0].shape}")
    else:
        print(f"Unbekannter Typ: {type(data)}")

def extract_coordinates(zelle):
    if isinstance(zelle, np.ndarray):
        return [np.array(item) for item in zelle]
    return []
# plotten des Bildes und der Koordinaten

# Funktion zum Plotten der Koordinaten
def plot_coordinates(coords, label, color, s, marker):
    for coord in coords:
        coord = np.array(coord)
        if coord.ndim == 2 and coord.shape[1] == 2:
            plt.scatter(coord[:, 0], coord[:, 1], color=color, marker=marker, label=label)
        elif coord.ndim == 1 and len(coord) == 2:
            plt.scatter(coord[0], coord[1], color=color, marker=marker, label=label)

if __name__ == '__main__':
    img = Image.open('home/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/data/CARDS_COURTYARD_B_T/frame_0011.jpg')
    path = osp.normpath(osp.join(osp.dirname(__file__), "data"))
    # mat_data = scipy.io.loadmat('/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')

    imgs, myleft, myright, yourleft, yourright = mkimgslist(path)
    # myleft, myright, yourleft, yourright = readdataout(mat_data)
    i, imgcv = gimgout(imgs)
    print_data_info(myleft, 'Myleft')
    print_data_info(myright, 'Myright')
    print_data_info(yourleft, 'Yourleft')
    print_data_info(yourright, 'Yourright')
    plt.figure(figsize=(10, 8))
    plt.imshow(imgcv)
    plt.axis('on')  # Achsen ausblenden, um nur das Bild zu zeigen
    myleft_coords = extract_coordinates(myleft)
    myright_coords = extract_coordinates(myright)
    yourleft_coords = extract_coordinates(yourleft)
    yourright_coords = extract_coordinates(yourright)

    # Koordinaten plotten
    plot_coordinates(myleft_coords, 'My Left', 'red', s=10, marker='.')
    plot_coordinates(myright_coords, 'My Right', 'blue', s=10, marker='.')
    plot_coordinates(yourleft_coords, 'Your Left', 'green', s=10, marker='.')
    plot_coordinates(yourright_coords, 'Your Right', 'purple', s=10, marker='.')


    plt.xlabel('X-Koordinaten')
    plt.ylabel('Y-Koordinaten')
    plt.title('Koordinaten Plot')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.grid(True)
    plt.show()
