# -*- coding: utf-8 -*-
"""
File containing the dataset class used in the train of the RCNN.
This dataset is the EgoHands dataset.
The class is an extension of the Dataset class provided by Pytorch.
To optimize memory consumption the dataset doesn't store the images but only the path to them. Images are read on fly when you access to an element of the dataset

Based on the script and tutorial on the Pytorch website (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

N.B. Inside path you must have the folders that you find inside _LABELLED_SAMPLES.

@author: Alberto Zancanaro (Jesus)
"""

# %%

import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import scipy
from PIL import Image

import random
import cv2
from support_function import *

import h5py

# %%

class EgoDataset(torch.utils.data.Dataset):

    def __init__(self, path, n_elements=-1, transforms=None):
        if (n_elements > 2400): n_elements = 450
        entire_folder = int(n_elements / 100)  # n_elements will become the nearest lower multiple of 100
        sample_last_Folder = n_elements % 100  # Not used

        # Work similar to MyDataset
        folder_list = []
        for element in os.walk(path): folder_list.append(element)
        del folder_list[0]

        folder_list = random.sample(folder_list, entire_folder + 1)

        with h5py.File('yourfile.mat', 'r') as file:
            data = list(file.keys())
            print(data)


