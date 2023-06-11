import os
import cv2
import torch
import numpy as np
import webp

from torch.utils.data import Dataset


class RoadAnomaly21(Dataset):
    """
    The given dataset_root entry in hparams is expected to be the root folder that contains the Anomaly Track data of
    the SegmentMeIfYouCan benchmark. The contents of that folder are 'images/', 'label_masks/', and 'LICENSE.txt'.
    Not all images are expected to have ground truth labels because the test set is held-out. However for consistency
    the dataset __getitem__ will return two tensors, the image and a dummy label (if the original one does not exist).
    
    The dataset has 10 extra validation samples with ground truth labels. Therefore in this dataset we define 3 modes of
    the data:
    - test: only loads the test samples without existing labels
    - val: only loads the validation samples that have ground truth labels. They are distinguished by the filenames which contain
        the string 'validation'.
    - all: loads everything including validation and testing samples.

    """


    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparam = hparams
        self.transforms = transforms
        
        images_root = os.path.join(hparams.dataset_root, 'images')
        labels_root = os.path.join(hparams.dataset_root, 'labels_masks')
        self.images = os.listdir(images_root)

        if hparams.dataset_mode == 'test':
            
            self.images = [os.path.join(images_root, img_path) for img_path in self.images if not ('validation' in img_path)]
            self.labels = [''] * len(self.images)
        elif hparams.dataset_mode == 'val':
            
            self.labels = [os.path.join(labels_root, img_path[:-4] + '_labels_semantic.png') for img_path in self.images if 'validation' in img_path]
            self.images = [os.path.join(images_root, img_path) for img_path in self.images if 'validation' in img_path]
        elif hparams.dataset_mode == 'all':
            
            self.labels = []
            for img_path in self.images:
                if 'validation' in img_path:
                    self.labels.extend([os.path.join(labels_root, img_path[:-4] + '_labels_semantic.png')])
                else:
                    self.labels.extend([''])
                
            self.images = [os.path.join(images_root, img_path) for img_path in self.images]
        else:
            raise Exception(f'Undefined Dataset Mode for RoadAnomaly21: {hparams.dataset_mode}')

        self.num_samples = len(self.images)

    def __getitem__(self, index):
        
        image = self.read_image(self.images[index])

        if 'validation' in self.images[index]:
            label = self.read_image(self.labels[index])
        else:
            label = np.zeros_like(image)
        
        label = label[:, :, 0]
        
        if self.transforms is not None:
            aug = self.transforms(image=image, mask=label)
            image = aug['image']
            label = aug['mask']

        return image, label.type(torch.LongTensor)   


    def __len__(self):
        return self.num_samples

    
    @staticmethod
    def read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img


class RoadObstacle21(Dataset):
    """
    The given dataset_root entry in hparams is expected to be the root folder that contains the Obstacle Track data of
    the SegmentMeIfYouCan benchmark. The contents of that folder are 'images/', 'label_masks/', and 'image-sources.txt'.
    Not all images are expected to have ground truth labels because the test set is held-out. However for consistency
    the dataset __getitem__ will return two tensors, the image and a dummy label (if the original one does not exist).
    
    The dataset has 30 extra validation samples with ground truth labels. Therefore in this dataset we define 3 modes of
    the data:
    - test: only loads the test samples without existing labels
    - val: only loads the validation samples that have ground truth labels. They are distinguished by the filenames which contain
        the string 'validation'.
    - all: loads everything including validation and testing samples.

    """


    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparam = hparams
        self.transforms = transforms
        
        images_root = os.path.join(hparams.dataset_root, 'images')
        labels_root = os.path.join(hparams.dataset_root, 'labels_masks')
        self.images = os.listdir(images_root)

        if hparams.dataset_mode == 'test':
            
            self.images = [os.path.join(images_root, img_path) for img_path in self.images if not ('validation' in img_path)]
            self.labels = [''] * len(self.images)
        elif hparams.dataset_mode == 'val':
            
            self.labels = [os.path.join(labels_root, img_path[:-5] + '_labels_semantic.png') for img_path in self.images if 'validation' in img_path]
            self.images = [os.path.join(images_root, img_path) for img_path in self.images if 'validation' in img_path]
        elif hparams.dataset_mode == 'all':
            
            self.labels = []
            for img_path in self.images:
                if 'validation' in img_path:
                    self.labels.extend([os.path.join(labels_root, img_path[:-5] + '_labels_semantic.png')])
                else:
                    self.labels.extend([''])

            self.images = [os.path.join(images_root, img_path) for img_path in self.images]
        else:
            raise Exception(f'Undefined Dataset Mode for RoadObstacle21: {hparams.dataset_mode}')

        self.num_samples = len(self.images)

    def __getitem__(self, index):
        
        image = self.read_webp(self.images[index])
        
        if 'validation' in self.images[index]:
            label = self.read_image(self.labels[index])
        else:
            label = np.zeros_like(image)
        
        label = label[:, :, 0]
        
        if self.transforms is not None:
            aug = self.transforms(image=image, mask=label)
            image = aug['image']
            label = aug['mask']

        return image, label.type(torch.LongTensor)   


    def __len__(self):
        return self.num_samples

    
    @staticmethod
    def read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img

    @staticmethod
    def read_webp(path):

        img = webp.load_image(path, 'RGB')
        img = np.array(img)

        return img
