import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch

# Maske laden
mask = read_image("output_image_8bitmask.png")[0]  # Nur ein Kanal
unique_ids = mask.unique()  # Finde die IDs in der Maske
print(len(unique_ids))
print(unique_ids)

# Definiere Farben für jedes Objekt
colors = [
    (255, 0, 0),   # Rot
    (0, 255, 0),   # Grün
    (0, 0, 255),   # Blau
    (255, 255, 0), # Gelb

    (0, 122, 50),  # kp
    (50, 0, 122),  # kp
    (122, 50, 0),   # kp
]

# Leeres RGB-Bild erzeugen, um die eingefärbte Maske zu speichern
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

# Durch die IDs iterieren und Farben zuweisen
for i, obj_id in enumerate(unique_ids):
    if obj_id == 0:  # 0 ist Hintergrund, überspringen
        continue
    color = colors[i % len(colors)]  # Zyklisch Farben zuweisen
    colored_mask[mask == obj_id.item()] = color  # Farbe für jede ID anwenden
    # Maske anzeigen
    plt.figure(figsize=(8, 8))
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.title("Colored Mask with Separate Colors for Each Object")
    plt.show()

# Maske anzeigen
plt.figure(figsize=(8, 8))
plt.imshow(colored_mask)
plt.axis('off')
plt.title("Colored Mask with Separate Colors for Each Object")
plt.show()
