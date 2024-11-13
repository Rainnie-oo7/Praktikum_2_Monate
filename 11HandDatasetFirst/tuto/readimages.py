import os
from PIL import Image

src = '/boris.grillborzer/11HandDatasetFirst/PennFudanPed'  # unsichere Methode stützt sich auf richtige Reihenfolge, später sehr schlecht nachverfolgbar, naja
image_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed'
c_folder = [
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_B_T/',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_H_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_S_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_COURTYARD_T_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_LIVINGROOM_B_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_LIVINGROOM_H_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_LIVINGROOM_S_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_LIVINGROOM_T_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_OFFICE_B_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_OFFICE_H_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_OFFICE_S_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CARDS_OFFICE_T_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_COURTYARD_B_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_COURTYARD_H_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_COURTYARD_S_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_COURTYARD_T_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_LIVINGROOM_B_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_LIVINGROOM_H_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_LIVINGROOM_S_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_LIVINGROOM_T_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_OFFICE_B_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_OFFICE_H_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_OFFICE_S_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/CHESS_OFFICE_T_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_COURTYARD_B_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_COURTYARD_H_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_COURTYARD_S_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_COURTYARD_T_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_LIVINGROOM_B_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_LIVINGROOM_H_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_LIVINGROOM_S_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_LIVINGROOM_T_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_OFFICE_B_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_OFFICE_H_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_OFFICE_S_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/JENGA_OFFICE_T_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_COURTYARD_B_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_COURTYARD_H_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_COURTYARD_S_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_COURTYARD_T_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_LIVINGROOM_B_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_LIVINGROOM_H_S',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_LIVINGROOM_S_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_LIVINGROOM_T_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_OFFICE_B_H',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_OFFICE_H_B',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_OFFICE_S_T',
'/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed/PUZZLE_OFFICE_T_S']

def getimagesfilelist():
    images_list = []
    polyg_list = []
    for root, dirs, files in os.walk(src):
        for imagefile in files:
            if imagefile.startswith("frame"):
                images_list.append(imagefile)
                # print(images_list)
        for polygfile in files:
            if polygfile.startswith("polygons"):
                polyg_list.append(polygfile)

        for d in dirs:  # ueberpruefe ob 100 Bilder und insgesamt 48 polyg-dateien
            folder_path = os.path.join(root, d)
            image_files = [f for f in os.listdir(folder_path) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            if len(image_files) != 100:
                print(f"Der Ordner '{folder_path}' enthält {len(image_files)} Bilder.")
                # print(polyg_list)
    if len(polyg_list) != 48:
        print(f"Zu viele oder zu wenige Polygon-Dateien: {len(polyg_list)}.")
    return images_list, polyg_list

getimagesfilelist()
images_list = getimagesfilelist()

  # Pfad zum Ordner
folderimages_list = os.listdir(image_folder)  # Liste aller Dateien im Ordner
# Optional: Filter nur auf Bilddateien, falls der Ordner auch andere Dateien enthält
folderimages_list = [f for f in folderimages_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

def transpose(images_list, image_folder):
    img_width = 0
    img_height = 0
    for image_name in images_list:
        # Erstelle den vollständigen Pfad zum Bild
        image_path = os.path.join(image_folder, image_name)
        try:
            # Öffne das Bild
            img = Image.open(image_path)

            # Bild vertikal spiegeln
            img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)

            # Bildgrößen ermitteln
            img_width, img_height = img_flipped.size
            print(f"{image_name}: Breite = {img_width}, Höhe = {img_height}")

        except Exception as e:
            print(f"Fehler beim Laden von {image_name}: {e}")

            # Rückgabe von img_width und img_height, wenn mindestens ein Bild verarbeitet wurde
        if img_width is not None and img_height is not None:
            return img_width, img_height
        else:
            print("Kein Bild wurde erfolgreich geladen.")
            return None, None



transpose(images_list, image_folder)


"""
def list_folders(parent_folder):
    folder_paths = []
    for root, dirs, files in os.walk(parent_folder):
        for d in dirs:
            folder_path = os.path.join(root, d)
            folder_paths.append(folder_path)
    folder_paths.sort()
    return folder_paths


# Beispiel: Pfad zum Überordner angeben
parent_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/PennFudanPed'

# Funktion aufrufen und Ordner ausgeben
folders = list_folders(parent_folder)
for folder in folders:
    print(folder)
"""