import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# img = Image.open('/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg')
# mat_data = scipy.io.loadmat('/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')
img = Image.open('/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg')
mat_data = scipy.io.loadmat('G:/Meine Ablage/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')
# Was macht das besondere einer init funktion aus
# -es müssen nur der pfad und die evtl die Datei gegeben werden
# -es kommen VORGESPEICHERTE sachen rein, die verlangsamen könnten.
# def loadthedata(path):
#     PFad -> mache myleft, myright, yoleft, yorigt
#          -> mache das image extrahien
#          -> extracte die zelle. zelle = (192, 2)
#          -> mache scatterplot-polyzug und unterlege auch bild in/unter plt objk
#               nzw. Gebe mir alle xylons mit dem Label miteinander aus
#     main
#         -> plt.show

data = mat_data['polygons']

def extract_data_iterrowsgetcolumns(data = mat_data['polygons']):#LOAD HERE THE ONLY polygons.mat DATEI WHOLE UNIT/PROCESS LONG.

    arrays_to_modify = ['myleft', 'myright', 'yourleft', 'yourright']
    for i in range(99):  # Bis 100 Iterationen
        for array_name in arrays_to_modify:
            # Setze den Wert des Arrays an Position [0, i] auf z. B. eine gewünschte Zahl oder 1
            try:
                # Zugriff auf die verschiedenen Spaltennamen    die idx ist der zweite Wert. Pixel Squares/Trauben Sense
                myleft = data['myleft'][0, i]
                myright = data['myright'][0, i]
                yourleft = data['yourleft'][0, i]
                yourright = data['yourright'][0, i]
                print("i", i)
                print("PennFudanPed[array_name][0, i]", data[array_name][0, i])
                return myleft, myright, yourleft, yourright
            except IndexError:
                print(f"{array_name} hat keine ausreichende Größe für Index [0, {i}].")

    # myleft = PennFudanPed['myleft'][0, 99]
    # myright = PennFudanPed['myright'][0, 99]
    # yourleft = PennFudanPed['yourleft'][0, 99]
    # yourright = PennFudanPed['yourright'][0, 99]

def extract_image():
    img = Image.open('/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg')

    pass
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
    return np.vstack(blob_data) if blob_data else np.array([])



if __name__ == '__main__':
    myleft, myright, yourleft, yourright = extract_data_iterrowsgetcolumns(data=mat_data['polygons'])
    img = extract_image()
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

