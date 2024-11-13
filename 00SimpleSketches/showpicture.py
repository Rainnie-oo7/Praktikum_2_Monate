import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io import decode_image
from pathlib import Path
from torchvision.utils import draw_bounding_boxes

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

eins = Path('/assets')
zwei = Path('/assets')
img1 = eins / 'gog.jpg'
img2 = zwei / 'dop.jpg'
dog1_int = decode_image(img1)
dog2_int = decode_image(img2)
dog_list = [dog1_int, dog2_int]

grid = make_grid(dog_list)
show(grid)

boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
colors = ["blue", "yellow"]
result = draw_bounding_boxes(dog1_int, boxes, colors=colors, width=5)
show(result)