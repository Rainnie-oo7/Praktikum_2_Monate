import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from skimage.draw import polygon
from torchvision import transforms
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.ops.boxes import masks_to_boxes
import scipy
import numpy as np

class PolygonModell(nn.Module):
    def __init__(self):
        super(PolygonModell, self).__init__()
        # Convolutional Layers für Bildmerkmale
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Lineare Schicht für Polygondaten (8 Werte für die 4 (x, y)-Punkte) in: 8 out: 64
        self.fc_polygon = nn.Linear(2, 921600)

        # Kombinierte Features
        self.fc_combined = nn.Linear(64 + 64, 128)
        self.fc_output = nn.Linear(128, 1)  # 1 für die Vorhersage (z. B. Objektklasse oder andere Ausgabe)

    def forward(self, image, polygon):
        image = image.float()
        # image = image.permute(0, 3, 1, 2)  # Ändere die Dimensionen von (Batch, Höhe, Breite, Kanäle) zu (Batch, Kanäle, Höhe, Breite)

        # Convolutional layers für Bildmerkmale
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        polygon = F.relu(self.fc_polygon(polygon))
        print(f"x shape: {x.shape}")
        print(f"polygon shape: {polygon.shape}")
        combined = torch.cat((x, polygon), dim=1)

        # Durch das Netzwerk führen
        combined = F.relu(self.fc_combined(combined))
        output = self.fc_output(combined)

        return output



class Mydataset(Dataset):
    def __init__(self, path):
        self.imgs = []
        self.masks = []

        for dir in os.listdir(path):
            full_dir_path = os.path.join(path, dir)

            if os.path.isdir(full_dir_path):
                # Iteriere durch alle Dateien im Unterverzeichnis
                for file in os.listdir(full_dir_path):
                    if file.startswith('frame'):
                        file_path = os.path.join(full_dir_path, file)
        # for dir in os.listdir(path):
        #     for file in os.listdir(os.path.join(path, dir)):
        #         if file.startswith('frame'):
                    # _
                    # muss hier nicht noch transformiert, etwa geblurrt, verpixeln gewerden und in ndArray/Tensor umgewandelt?
                    # ~
                    #     self.imgs.append(cv2.imread(os.path.join(path, dir, file)))
                        self.imgs.append(cv2.imread(file_path))
                    else:
                        self.masks = self.masks + self.extract_polygons(path)
                        print(len(self.masks)) #// 19???
            break

    def extract_polygons(self, path):
        masks = []

        # Iteriere durch alle Verzeichnisse im angegebenen Pfad
        for dir in os.listdir(path):
            full_dir_path = os.path.join(path, dir)

            # Überprüfen, ob es sich um ein Verzeichnis handelt
            if os.path.isdir(full_dir_path):
                # Iteriere durch alle Dateien im Unterverzeichnis
                for file in os.listdir(full_dir_path):
                    if file.startswith('polygons') and file.endswith('.mat'):
                        file_path = os.path.join(full_dir_path, file)
                        # print(f"Verarbeite Datei: {file_path}")

                        # Lade die .mat-Datei
                        mat_data = scipy.io.loadmat(file_path)
                        data = mat_data['polygons']
                        print(data.shape[0])
                        # print("Das ist PennFudanPed:", PennFudanPed)
                        # If 'PennFudanPed' is a structured array, you may need to iterate over its rows
                        for i in range(data.shape[0]):
                            myleft = data['myleft'][i, 0]
                            myright = data['myright'][i, 0]
                            yourleft = data['yourleft'][i, 0]
                            yourright = data['yourright'][i, 0]
                            # Iteriere durch die Polygondaten
                            # for i in range(PennFudanPed.shape[0]):
                            #     myleft, myright, yourleft, yourright = PennFudanPed[['myleft'], ['myright'], ['yourleft'], ['yourright']][i, 0]
                            #     print(f"Ml: {myleft}, mr: {myright}, yl: {yourleft}, yr: {yourright}")

                            # Erstelle Maske
                            maske = [[ml, mr, yl, yr] for ml, mr, yl, yr in zip(myleft, myright, yourleft, yourright)]
                            masks.extend(maske)

        print(f"gesamtanzahl masks: {len(masks)}")  # //2485
        return masks


    def create_mask_image(self, polygons, image_size):
        black_img = np.zeros(image_size, dtype=np.uint8)
        for poly in polygons:
            # angenehmen, dass poly inform ist: [[x1, y1], [x2, y2], ...]
            pts = poly.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(black_img, pts, color=(255))  # White color for mask
        mask_img = black_img
        # cv2.imwrite('mask_image.png', mask_img)
        return mask_img




    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.imgs):
            raise IndexError(f"Index {idx} is out of range für dataset größe: {len(self.imgs)}")

        img = self.imgs[idx]

        # Verwende extract_polygons nur einmal außerhalb der Schleife oder in __init__, falls möglich
        if not hasattr(self, 'masks'):
            self.masks = self.extract_polygons(path)

        if idx >= len(self.masks):
            raise IndexError(f"Index {idx} is out of range für masks größe {len(self.masks)}")

        mask_list = []
        for mask in self.masks[idx]:
            if mask.size > 0:
                polygon = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                mask_list.append(polygon)

        if mask_list:
            polygon_tensor = torch.tensor(mask_list, dtype=torch.float32)
        else:
            polygon_tensor = torch.empty(0)
        return img, polygon_tensor

def train():
    model = PolygonModell()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(3):
        for img, polygon_tensor in dataloader:
            optimizer.zero_grad()
            print("Imgshape", img.shape)


            # forward
            output = model(img, polygon_tensor)

            # Dummy-Labels, hier kannst du deine Zielwerte verwenden #???
            labels = torch.ones(img.size(0), 1)

            loss = criterion(output, labels)
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

            # backwarddurchlauf und Optimierung
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
    image_size = (720, 1280)

    dataset = Mydataset(path)

    train()
