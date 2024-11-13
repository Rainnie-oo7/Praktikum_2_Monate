import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.io import read_image

path = '''/home/boris.grillborzer/Downloads/LabPicsV1/Complex/Train/Beautiful Chemical Chameleon (KMnO4+NaOH+Sugar+H2O - Color Changing) + 200 SUBS !-screenshot (1)/ignore.png'''
# Load the binary mask image
# mask_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
mask_image = read_image("PennFudanPed/PedMasks/FudanPed00046_mask.png")
print(type(mask_image))
print(mask_image)

import os
print(os.path.exists(path))  # This should return True

# Define different color maps
colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
]

# Create colored masks
colored_masks = []
for color in colors:
    colored_mask = np.zeros((mask_image.shape[0], mask_image.shape[1], 3), dtype=np.uint8)
    colored_mask[mask_image == 3] = color  # Apply color where mask is 1
    colored_masks.append(colored_mask)

# Plot the masks
fig, axes = plt.subplots(1, len(colored_masks), figsize=(15, 5))
for ax, colored_mask in zip(axes, colored_masks):
    ax.imshow(colored_mask)
    ax.axis('off')
plt.show()

# Plot the masks
fig, axes = plt.subplots(1, len(mask_image), figsize=(15, 5))
for ax, mask_image in zip(axes, mask_image):
    ax.imshow(mask_image)
    ax.axis('off')
plt.show()
