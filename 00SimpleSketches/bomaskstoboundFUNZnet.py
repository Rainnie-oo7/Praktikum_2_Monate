from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = os.path.join('/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/12CustomDatasetFudanPedNullst/PennFudanPed/PNGImages', "FudanPed00054.png")
msk = os.path.join('/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/12CustomDatasetFudanPedNullst/PennFudanPed/PedMasks', "FudanPed00054_mask.png")
rdm = np.random.rand(100, 100, 3)  # 3-Kanäle für RGB

img = read_image(img)
mask = read_image(msk)

# Zeige das Bild an
plt.imshow(img)
plt.axis('off')  # Versteckt die Achsen
plt.show()
