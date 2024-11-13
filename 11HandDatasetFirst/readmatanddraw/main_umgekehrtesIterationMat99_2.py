import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import os.path as osp

def init(path):
    imgs = []
    mymasks = []
    full_path = ''
    if isinstance(path, str):
        for dir in os.listdir(path):
            full_dir_path = os.path.join(path, dir)
            if os.path.isdir(full_dir_path):
                # print(f"Checking directory: {full_dir_path}")
                for file in os.listdir(full_dir_path):
                    # print(f"Found file: {file}")
                    if file.startswith('frame'):
                        files = sorted([file for file in os.listdir(full_dir_path) if file.startswith('frame')])
                        # Dann die Bilder in der sortierten Reihenfolge laden
                        for file in files:
                            img_path = os.path.join(full_dir_path, file)
                            imgs.append(cv2.imread(img_path))
                    else:
                        if file.startswith('polygons'):
                            full_path = os.path.join(full_dir_path, file)
                            # print("full_path:", full_path)
                            hands = extract_data_iterrowsgetcolumns(full_path)
    else:
        print("Der Pfad muss ein String sein.")
    # for idx, img in enumerate(self.imgs):
    #     if idx < 0 or idx > len(self.mymasks):
    #         print(f"Index {idx} is out of range für len(self.maske): {len(self.mymasks)}")
    print("lenimg:", len(imgs))
    return imgs, full_path, hands
img = Image.open('/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/data/CARDS_COURTYARD_B_T/frame_0011.jpg')
# img = Image.open('G:/Meine Ablage/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg')
# mat_data = scipy.io.loadmat('G:/Meine Ablage/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')
# Was macht das besondere einer init funktion aus
# -es müssen nur der pfad und die evtl die Datei gegeben werden
# -es kommen VORGESPEICHERTE sachen rein, die verlangsamen könnten.
# def loadthedata(path):
#     PFad -> mache myleft, myright, yoleft, yorigt
#          -> mache das image extrahien
#          -> extracte die zelle. zelle = (192, 2)
#          -> mache eine blackimage-list
#             (mache scatterplot-polyzug und unterlege auch bild in/unter plt objk)
#               nzw. Gebe mir alle xylons mit dem Label miteinander aus
#          -> Gebe jedem image_id eine idx. SO kann das Modell die image-id anhand des Targets finden
#          -
#     main
#         -> plt.show



def extract_data_iterrowsgetcolumns(full_path):#LOAD HERE THE ONLY polygons.mat DATEI WHOLE UNIT/PROCESS LONG.
    mat_data = scipy.io.loadmat(full_path)
    # mat_data = scipy.io.loadmat(
    #     '/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')
    data = mat_data['polygons']
    arrays_to_modify = ['myleft', 'myright', 'yourleft', 'yourright']
    for idx in range(99):  # Bis 100 Iterationen
        for array_name in arrays_to_modify:
            # Setze den Wert des Arrays an Position [0, i] auf z. B. eine gewünschte Zahl oder 1
            try:
                # Zugriff auf die verschiedenen Spaltennamen    die idx ist der zweite Wert. Pixel Squares/Trauben Sense
                myleft = data['myleft'][0, idx]
                myright = data['myright'][0, idx]
                yourleft = data['yourleft'][0, idx]
                yourright = data['yourright'][0, idx]
                print("i", idx)
                print("PennFudanPed[array_name][0, i]", data[array_name][0, idx])
                return myleft, myright, yourleft, yourright
            except IndexError:
                print(f"{array_name} hat keine ausreichende Größe für Index [0, {idx}].")

    # myleft = PennFudanPed['myleft'][0, 99]
    # myright = PennFudanPed['myright'][0, 99]
    # yourleft = PennFudanPed['yourleft'][0, 99]
    # yourright = PennFudanPed['yourright'][0, 99]

def extract_image(imgs):
    img = Image.open('/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/data/CARDS_COURTYARD_B_T/frame_0011.jpg')
    for i in imgs:
        imgcv = cv2.imread(i)
        return i, imgcv

# überprüfe die Struktur der einzelnen Koordinatendaten
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
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('on')  # Achsen ausblenden, um nur das Bild zu zeigen
# Funktion zum Plotten der Koordinaten
def blob_the_coordinates(coords, color, marker, label, s=10):
    blob_data = []
    for coord in coords:
        coord = np.array(coord)
        if coord.ndim == 2 and coord.shape[1] == 2:
            plt.scatter(coord[:, 0], coord[:, 1], color=color, marker=marker, label=label)
            blob_data.append(coord)
        elif coord.ndim == 1 and len(coord) == 2:
            plt.scatter(coord[0], coord[1], color=color, marker=marker, label=label)
            blob_data.append(np.array([coord]))
            print("blob_data type", type(blob_data))
            print("blob_data", blob_data)
            print("blob_data len", len(blob_data))

    return np.vstack(blob_data) if blob_data else np.array([])



if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
    full_path = init(path)
    myleft, myright, yourleft, yourright = extract_data_iterrowsgetcolumns(full_path)
    img = extract_image(imgs)
    print_data_info(myleft, 'Myleft')
    print_data_info(myright, 'Myright')
    print_data_info(yourleft, 'Yourleft')
    print_data_info(yourright, 'Yourright')

    myleft_coords = extract_coordinates(myleft)
    myright_coords = extract_coordinates(myright)
    yourleft_coords = extract_coordinates(yourleft)
    yourright_coords = extract_coordinates(yourright)

    blobs_tensor = [
        blob_the_coordinates(myleft_coords, 'red', marker='.', label='My Left'),
        blob_the_coordinates(myright_coords, 'blue', marker='.', label='My Right'),
        blob_the_coordinates(yourleft_coords, 'green', marker='.', label='Your Left'),
        blob_the_coordinates(yourright_coords, 'purple', marker='.', label='Your Right')
    ]

    # Umwandeln in einen 4-Einträge-Tensor, falls gewünscht (z.B. für PyTorch)
    blobs_tensor = np.array(blobs_tensor, dtype=object)  # dtype=object für Arrays unterschiedlicher Länge

    plt.xlabel('X-Koordinaten')
    plt.ylabel('Y-Koordinaten')
    plt.title('Koordinaten Plot')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.grid(True)
    plt.show()

    print("Tensor mit 4 Einträgen:", blobs_tensor)

