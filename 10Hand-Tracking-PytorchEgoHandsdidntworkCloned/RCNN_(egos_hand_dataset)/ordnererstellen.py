# import os
# import shutil
#
#
# def copy_folders_with_polygons(src, dest, file_to_copy):
#
#
#     # Durchlaufe alle Ordner und Unterordner
#     for root, dirs, files in os.walk(src):
#         # Relativer Pfad vom Quellordner
#         relative_path = os.path.relpath(root, src)
#         # Zielpfad für die Ordnerstruktur
#         dest_dir = os.path.join(dest, relative_path)
#
#         # Erstelle den Ordner, falls er noch nicht existiert
#         if not os.path.exists(dest_dir):
#             os.makedirs(dest_dir)
#
#         # Prüfe, ob die Datei 'polygons2.mat' im aktuellen Ordner vorhanden ist
#         if file_to_copy in files:
#             src_file_path = os.path.join(root, file_to_copy)
#             dest_file_path = os.path.join(dest_dir, file_to_copy)
#             # Kopiere die Datei in den neuen Ordner
#             shutil.copy2(src_file_path, dest_file_path)
#
#         # Nur die Ordnerstruktur kopieren, keine weiteren Dateien
#         for directory in dirs:
#             new_dir = os.path.join(dest_dir, directory)
#             if not os.path.exists(new_dir):
#                 os.makedirs(new_dir)
#
# # Pfade anpassen
# source_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/09Hand-Tracking-PytorchMATTYPEUNKNOWN/RCNN_(egos_hand_dataset)/PennFudanPed'
# destination_folder = '/home/boris.grillborzer/polygonsegohandsv3'
#
# file_to_copy = 'polygons2.mat'
#
# copy_folders_with_polygons(source_folder, destination_folder, file_to_copy)

import os
"""
def delete_polygons_files(src, file_to_delete):
    # Durchlaufe alle Ordner und Unterordner
    for root, dirs, files in os.walk(src):
        # Prüfe, ob die Datei 'polygons2.mat' im aktuellen Ordner vorhanden ist
        if file_to_delete in files:
            file_path = os.path.join(root, file_to_delete)
            # Lösche die Datei
            os.remove(file_path)
            print(f"{file_to_delete} gelöscht aus: {file_path}")

# Pfad anpassen
source_folder = '/home/boris.grillborzer/polygonsegohandsv3'
file_to_delete = 'polygons2.mat'

delete_polygons_files(source_folder, file_to_delete)
"""
#
"""
import os
import shutil


def copy_polygons_v7_to_folders(src, dest, file_to_copy):
    # Erstelle den Zielordner, falls er nicht existiert
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Durchlaufe alle Ordner und Unterordner im Quellverzeichnis
    for root, dirs, files in os.walk(src):
        # Relativer Pfad vom Quellordner
        relative_path = os.path.relpath(root, src)
        # Zielpfad für die Ordnerstruktur
        dest_dir = os.path.join(dest, relative_path)

        # Erstelle den Ordner im Zielverzeichnis, falls er nicht existiert
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Prüfe, ob die Datei 'polygons_v7.mat' im aktuellen Ordner vorhanden ist
        if file_to_copy in files:
            src_file_path = os.path.join(root, file_to_copy)
            dest_file_path = os.path.join(dest_dir, file_to_copy)
            # Kopiere die Datei in den neuen Ordner
            shutil.copy2(src_file_path, dest_file_path)
            print(f"{file_to_copy} kopiert nach: {dest_file_path}")


# Pfade anpassen
source_folder = '/home/boris.grillborzer/polygonsegohandsv3'
destination_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/09Hand-Tracking-PytorchMATTYPEUNKNOWN/RCNN_(egos_hand_dataset)/PennFudanPed'
file_to_copy = 'polygons_v4.mat'

copy_polygons_v7_to_folders(source_folder, destination_folder, file_to_copy)
"""