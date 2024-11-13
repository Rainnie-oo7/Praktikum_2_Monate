import matplotlib.pyplot as plt
from torchvision.io import read_image

# Lade das Bild und die Maske
image = read_image("PennFudanPed/PNGImages/FudanPed00046.png")
mask = read_image("PennFudanPed/PedMasks/FudanPed00046_mask.png")

# Bild und Maske anzeigen
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))  # Permutation für das Bild
plt.axis('off')

plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0), cmap='gray')  # Permutation für die Maske
plt.axis('off')

plt.show()
