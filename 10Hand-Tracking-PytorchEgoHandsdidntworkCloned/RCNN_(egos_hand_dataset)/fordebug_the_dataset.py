from Mydatasetandereohnefarbe import Mydataset
import torch
import os.path as osp


if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "data"))
    dataset_train = EgoDataset(path, n_elements=450, transforms=get_transform(False))
    print(dataset[14])