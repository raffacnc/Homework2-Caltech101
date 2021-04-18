from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def init(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).init(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')


        self.root = root+"/"+split+".txt" #./root/train.txt
        self.caltech_frame = pd.read_csv(self.root, delimiter = '\n', header=None)
        self.transform = transform
        self.target_transform = target_transform



      

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the getitem method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def getitem(self, index):
        '''
        getitem should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        img_name = os.path.join(self.root, self.caltech_frame.iloc[index, 0])
        label = img_name.split("/")[3]
        
        name = img_name.split("/")[4]

        image = io.imread("./root/101_ObjectCategories/"+"/"+label+"/"+name)
        image = Image.fromarray(image)
        sample = {'image' : image, 'label' : label} # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return sample

    def len(self):
        '''
        The len method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''

        length = self.caltech_frame.size # Provide a way to get the length (number of elements) of the dataset
        return length
