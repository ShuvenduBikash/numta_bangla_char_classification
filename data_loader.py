import glob
import random
import os
import pandas as pd
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
np.random.seed(123)


def file_names(root="E:\\Datasets\\NUMTA"):
    images = []
    classes = []
    
    datasets = ['a','b','c','d','e']
    for dataset in datasets:
        labels = pd.read_csv(root + "/training-{0}.csv".format(dataset))                
        labels = labels[['filename', 'digit']]
        
        files = root+  "\\training-{0}\\".format(dataset) + np.array(labels['filename'])
        cls = list(labels['digit'])
        
        images.extend(files)
        classes.extend(cls)
        
    return np.array(images), np.array(classes)


class ImageDataset(Dataset):
    def __init__(self, root="E:\\Datasets\\NUMTA", transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.images = []
        self.classes = []
        self.root = root
        images, classes = file_names(root)
        # print('total images', len(images))

        idx = np.random.permutation(len(images))
        train_len = int(len(images)*.9)

        if mode == 'train':
            self.images = images[idx[:train_len]]
            self.classes = classes[idx[:train_len]]
        else:
            self.images = images[idx[train_len:]]
            self.classes = classes[idx[train_len:]]

    def __getitem__(self, index):
        image = self.transform(Image.open(self.images[index % len(self.images)]).convert('RGB'))
        label = self.classes[index % len(self.images)]

        return image, label

    def __len__(self):
        return len(self.images)
