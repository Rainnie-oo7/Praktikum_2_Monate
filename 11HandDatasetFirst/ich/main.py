from mydataset import *
from polygonmodell import PolygonModell
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from mydataset import Mydataset
import os.path as osp

def train():
    model = PolygonModell()

    for polygon, imgs in dataloader:
        # Polygone durch das Modell führen
        output = model(polygon.view(polygon.size(0), -1))  # BATCH_SIZE x 8
        print(output)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1, # how many samples per batch?
                                  num_workers=1, # how many subprocesses to use for PennFudanPed loading? (higher = more)
                                  shuffle=True) # shuffle the PennFudanPed?

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False) # don't usually need to shuffle testing PennFudanPed
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "./PennFudanPed"))
    maskso = extract_polygons
    Mydataset(path)
    dataset = Mydataset(path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    process = train()
