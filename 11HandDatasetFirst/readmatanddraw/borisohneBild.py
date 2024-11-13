import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/frame_0011.jpg')
mat_data = scipy.io.loadmat('/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/polygons.mat')

print(mat_data.keys())
data = mat_data['polygons']
# Zugriff auf die verschiedenen Spaltennamen
myleft = data['myleft'][0, 0]
myright = data['myright'][0, 0]
yourleft = data['yourleft'][0, 0]
yourright = data['yourright'][0, 0]

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

print_data_info(myleft, 'Myleft')
print_data_info(myright, 'Myright')
print_data_info(yourleft, 'Yourleft')
print_data_info(yourright, 'Yourright')

def plot_coordinates(coords, label, marker):
    if isinstance(coords, np.ndarray) and coords.ndim > 1:
        for coord in coords:
            coord = np.array(coord)
            if coord.ndim == 2 and coord.shape[1] == 2:
                plt.plot(coord[:, 0], coord[:, 1], marker, label=label)
            elif coord.ndim == 1 and len(coord) == 2:
                plt.plot(coord[0], coord[1], marker, label=label)
    else:
        print(f"Keine gültigen Koordinaten zum Plotten für {label}")

plt.figure(figsize=(10, 8))

plot_coordinates(myleft, 'My Left', 'o')
plot_coordinates(myright, 'My Right', 's')
plot_coordinates(yourleft, 'Your Left', '^')
plot_coordinates(yourright, 'Your Right', 'x')

plt.xlabel('X-Koordinaten')
plt.ylabel('Y-Koordinaten')
plt.title('Koordinaten Plot')
plt.legend()
plt.grid(True)
plt.show()