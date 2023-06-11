import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class Mapillary(Dataset):

    def __init__(self, hparams, transform=None, mode='val', labels_mode="v2"):
        super().__init__()

        self.hparams = hparams
        self.transform = transform
        self.mode = mode
        self.labels_mode = labels_mode
    
        if mode == 'train':
            mode_folder = "training"
        elif mode == "val":
            mode_folder = "validation"
        elif mode == "test":
            mode_folder = "testing"
        else:
            raise ValueError(f"Undefined dataset mode: {mode}")

        if labels_mode == "v2":
            version = "v2.0"
        elif labels_mode == "v1":
            version = "v1.2"
        else:
            raise ValueError(f"Undefined labels version: {labels_mode}")

        images_path = os.path.join(hparams.dataset_root, mode_folder, "images")
        labels_path = os.path.join(hparams.dataset_root, mode_folder, "labels")

        self.images = []
        self.labels = []

        for img in os.listdir(images_path):
            
            self.images.extend([os.path.join(images_path, img)])
            self.labels.extend([os.path.join(labels_path, img[:-3] + "png")])
        
        self.data_size = len(self.images)


    def __getitem__(self, index):
        
        image = np.array(Image.open(self.images[index]).convert('RGB'))
        target = np.array(Image.open(self.labels[index]))

        if self.transform:
            aug = self.transform(image=image, mask=target)
            image, target = aug["image"], aug["mask"]

        return image, target


    def __len__(self):
        return self.data_size