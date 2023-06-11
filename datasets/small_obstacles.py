import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, InterpolationMode

def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img

class SmallObstacles(Dataset):
    """
    The root of the dataset is expected to contain 3 folders:
        - train/
        - val/
        - test/
    Inside each of these folders, there are also folders where every folder represents a sequence.
    In a folder representing a sequence, the following folders should be found:
        - image/ : contains images in *.png format
        - labels/ : contains segmentation labels in *.png fromat
    """
    def __init__(self, hparams, transforms, mode='val'):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms

        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Mode should be one of train, val, test, but instead {mode} was given')

        root = os.path.join(hparams.dataset_root, mode)

        self.images = []
        self.labels = []

        for seq in os.listdir(root):
            images_path = os.path.join(root, seq, 'image')
            labels_path = os.path.join(root, seq, 'labels')
            labels = os.listdir(labels_path)
            for name in labels:
                self.images.extend([os.path.join(images_path, name)])
                self.labels.extend([os.path.join(labels_path, name)])
        
        self.data_size = len(self.images)

    def transform_labels_to_ood(self, label):
        """
        This transformation aims to transform the labels in the dataset into 3 types:
            - background / void: regions that are ignored in evaluation
            - OOD: regions that should be predicted as anomalous
            - non-OOD: basically the road region.

        The mapping can be achieved as follows:
        - Road (128, 0, 0) -> 0
        - Void (0 ,  0, 0) -> 255
        - Everything else  -> 1
        """
        H, W, C = label.shape

        R = label[:, :, 0]
        G = label[:, :, 1]
        B = label[:, :, 2]

        void_mask = (R == 0) & (G == 0) & (B == 0)
        road_mask = (R == 128) & (G == 0) & (B == 0)
        
        label = np.ones((H, W))
        label[void_mask] = 255
        label[road_mask] = 0

        return label

    def __getitem__(self, index):
        
        image = read_image(self.images[index])
        label = read_image(self.labels[index])
        
        label = self.transform_labels_to_ood(label)

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)
    
    def __len__(self):
        return self.data_size