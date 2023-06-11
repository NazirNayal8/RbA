import os
import cv2
import random
import json
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms.functional import resize


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


class StreetHazards(Dataset):
    class_names = [
    "unlabeled", 
    "building", 
    "fence", 
    "pedestrian", 
    "pole", 
    "road line", 
    "road",
    "sidewalk", 
    "vegetation", 
    "car", 
    "wall", 
    "trafic sign", 
    "other",
    "anomaly"
]

    def __init__(
            self,
            hparams,
            mode='test',
            transforms=None,
            image_size=None,
            image_resize_func=None,
    ):
        super().__init__()

        if not (mode in ['train', 'val', 'test']):
            raise Exception(f'Unsupported dataset mode: {mode}')

        self.hparams = hparams
        self.transforms = transforms
        self.image_size = image_size
        self.image_resize_func = image_resize_func
        self.mode = mode
        if mode == "val":
            self.mode = "train"

        if image_size is None:
            raise Exception(f'Image Size Cannot be None')

        # if no transforms given use default transforms
        if self.transforms is None:
            self.transforms = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        if mode == 'train':
            self.images_list, self.labels_list = self._read_data_file(hparams.train_file)

        elif mode == 'val':
            self.images_list, self.labels_list = self._read_data_file(hparams.val_file)

        else:
            self.images_list, self.labels_list = self._read_data_file(hparams.test_file)

        self.current_sample = 0  # used to track whether we entered a new batch
        self.num_samples = len(self.images_list)

    def __getitem__(self, index):

        image = self._read_image(self.images_list[index])
        label = self._read_label(self.labels_list[index])

        image, label = self.process_train(image, label)
        
        return image, label.squeeze()

    def process_train(self, image, label):

        aug = self.transforms(image=image, mask=label)
        image, label = aug['image'], aug['mask']

        # aug = self.image_resize_func(image=image, mask=label)
        # image, label = aug['image'], aug['mask']

        # #label = self.mask_downsample_func(label.unsqueeze(0))

        return image, label.type(torch.LongTensor)

    def process_val(self, image, label):

        if self.transforms is not None:
            aug = self.transforms(image=image, mask=label)
            image, label = aug['image'], aug['mask']

        if self.hparams.val_image_strategy in ['custom', 'short_size_ratio']:
            aug = self.image_resize_func(image=image, mask=label)
            image, label = aug['image'], aug['mask']

            return image, label.type(torch.LongTensor)

        if self.hparams.val_image_strategy == 'multi_scale':

            aug = self.image_resize_func(image=image, mask=label)
            image, label = aug['image'], aug['mask']

            images = []
            for sz in self.hparams.val_image_sizes:
                new_size = [
                    sz,
                    round_to_nearest_multiple(sz * self.hparams.hw_ratio, self.hparams.seg_downsample_rate)
                ]
                images.extend([resize(image, size=new_size)])

            image = images

        elif self.hparams.val_image_strategy == 'multi_augment':
            raise NotImplementedError()
            # images = []
            # for _ in range(self.hparams.val_num_augments):
            #
            #     images.append(self.random_augment(image))
            #
            # label = self.mask_downsample_func(label)

        elif self.hparams.val_image_strategy == 'multi_scale_and_multi_augment':
            raise NotImplementedError()

        return image, label.type(torch.LongTensor)

    def random_augment(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img

    def _read_label(self, path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.int8)
        
       
        if self.mode == 'test':
            pass
            # img[img == 4] = 255
            # ood_mask = (img == 14)
            
            # img[(img != ood_mask) & (img != 255) & (img != -1)] = 0
            # img[ood_mask] = 1

        else:
            img = img - 1
            img[img == 3] = 13
            img[img >= 3] = img[img >= 3] - 1


        # 3 channels are equal, so only use one of them (H, W, 3) -> (H, W)
        img = img[:, :, 0]
        return img

    def _read_data_file(self, path):

        images = []
        labels = []
        with open(os.path.join(self.hparams.dataset_root, self.mode, path)) as f:
            data = json.load(f)
        for d in data:
        
            images.extend([os.path.join(self.hparams.dataset_root, self.mode, d["fpath_img"])])
            if self.mode != 'test':
                labels.extend([os.path.join(self.hparams.dataset_root, self.mode, d["fpath_segm"])])
            else:
                labels.extend([os.path.join(self.hparams.dataset_root, self.mode, d["fpath_segm"].replace(".png", "_processed.png"))])

        return images, labels
