import scipy.io
import scipy
import os
import os.path as osp
import numpy
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.PennFudanPed import Dataset, DataLoader
# from torchvision import transforms, utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



start_folder = '/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/12HandDatasetFirst/PennFudanPed'
# landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

# n = 65
# img_name = landmarks_frame.iloc[n, 0]
# landmarks = landmarks_frame.iloc[n, 1:]
# landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
#
# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks: {}'.format(landmarks[:4]))
"""v
Dataset class

torch.utils.PennFudanPed.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

    __len__ so that len(dataset) returns the size of the dataset.

    __getitem__ to support the indexing such that dataset[i] can be used to get iith sample.

Let’s create a dataset class for our face landmarks dataset. We will read the csv in __init__ but leave the reading of images to __getitem__. This is memory efficient because all the images are not stored in the memory at once but read as required.

Sample of our dataset will be a dict {'image': image, 'landmarks': landmarks}. Our dataset will take an optional argument transform so that any required processing can be applied on the sample. We will see the usefulness of transform in the next section.
"""

def find_polygon_mat(start_folder):
    # Durchlaufe alle Ordner und Unterordner
    for root, dirs, files in os.walk(start_folder):
        if "polygons2.mat" in files:
            print("hello")
            # print(f"Gefunden: {os.path.join(root, 'polygon.mat')}")
            print(scipy.io.loadmat("polygons2.mat"))
            # print(scipy.io.loadmat("polygons2.mat").keys())
            # print(scipy.io.loadmat("polygons2.mat")['polygons'].shape)

def print_tuples_from_arrays():    # Hier später Class schreiben mit self.methoden
    nested = scipy.io.loadmat("polygons2.mat")['polygons'].shape

    for tuple in len(nested.shape[1]):


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause()  # pause a bit so that plots are updated

if __name__ == '__main__':
    find_polygon_mat(start_folder)
    # path = osp.join(osp.dirname(__file__), "polygons2.mat")
    # print(scipy.io.loadmat("polygons2.mat").keys())
    # print(scipy.io.loadmat("polygons2.mat")['polygons'].shape)

"""
    plt.ion()  # interactive mode
    plt.figure()
    show_landmarks(io.imread(os.path.join('faces/', img_name)), landmarks)
    plt.show()



class FaceLandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
    
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

"""