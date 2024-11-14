import cv2
import numpy as np


def convert_to_8bit(image_path, output_path):
    # Bild laden
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Lädt das Bild in BGR
    if img is None:
        print("Bild konnte nicht geladen werden!")
        return

    # Konvertiere das Bild in Graustufen (dies führt zu einem 8-Bit Bild)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: Um das Bild auf ein wirkliches 8-Bit Niveau zu bringen (Skalierung auf 0-255)
    img_8bit = np.uint8(img_gray)  # Dies stellt sicher, dass die Bildwerte im Bereich 0-255 liegen

    # Speichern des 8-Bit Bildes
    cv2.imwrite(output_path, img_8bit)
    print(f"Das Bild wurde als 8-Bit Graustufenbild gespeichert: {output_path}")


# Beispielaufruf
image_path = '../12CustomDatasetFudanPedNullst/A-ima-0001.png'  # Pfad zum Eingabebild
output_path = 'output_image_8bitmask.png'  # Pfad zum Ausgabebild
convert_to_8bit(image_path, output_path)
