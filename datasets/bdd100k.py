import os
import cv2
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms.functional import resize


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


class BDD100KSeg(Dataset):
    class_names = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "void"
    ]

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [119, 11, 32],
        [0, 0, 230],
        [0, 80, 100],
        [0, 0, 0],
    ]

    def __init__(
            self,
            hparams,
            mode='train',
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

        if image_size is None:
            raise Exception(f'Image Size Cannot be None')

        # if no transforms given use default transforms
        if self.transforms is None:
            self.transforms = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        self.mask_downsample_func = Resize((self.image_size[0] // hparams.seg_downsample_rate,
                                            self.image_size[1] // hparams.seg_downsample_rate),
                                           interpolation=InterpolationMode.NEAREST)

        if mode == 'train':
            self.images_list, self.labels_list = self._read_data_file(hparams.train_file)

        elif mode == 'val':
            self.images_list, self.labels_list = self._read_data_file(hparams.val_file)

        else:
            self.images_list, self.labels_list = self._read_data_file(hparams.images_test_file)

        self.current_sample = 0  # used to track whether we entered a new batch
        self.num_samples = len(self.images_list)

    def __getitem__(self, index):

        image = self._read_image(self.images_list[index])
        label = self._read_label(self.labels_list[index])

        if self.mode == 'train':
            image, label = self.process_train(image, label)
        if self.mode == 'val':
            image, label = self.process_val(image, label)

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
       
        img[img == -1] = 19
        # 3 channels are equal, so only use one of them (H, W, 3) -> (H, W)
        img = img[:, :, 0]

        return img

    def _read_data_file(self, path):

        images = []
        labels = []
        fl = open(os.path.join(self.hparams.dataset_root, path), 'r')
        for f in fl:
            p = f[:-1]
            if ',' in p:
                p = p.split(',')
                images.extend([os.path.join(self.hparams.dataset_root, p[0])])
                labels.extend([os.path.join(self.hparams.dataset_root, p[1])])
            else:
                images.extend(images.extend([os.path.join(self.hparams.dataset_root, p[0])]))

        return images, labels
