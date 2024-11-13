import torch.nn as nn
from torch.utils.data import random_split

from torch.utils.data import Dataset, DataLoader
from mydataset import Mydataset

class PolygonModell(nn.Module):
    def __init__(self):
        super(PolygonModell, self).__init__()
        # Beispielhafte lineare Schicht
        self.fc = nn.Linear(8, 256)  # 8 für die 4 (x, y)-Punkte des Polygons

    def forward(self, polygon):
        # Polygon durch das Modell führen
        out = self.fc(polygon)
        return out
