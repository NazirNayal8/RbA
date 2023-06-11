import torch
import os
import cv2
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


class FishyscapesLAF(Dataset):
    """
    The Dataset folder is assumed to follow the following structure. In the given root folder, there must be two
    sub-folders:
    - fishyscapes_lostandfound: contains the mask labels.
    - laf_images: contains the images taken from the Lost & Found Dataset
    """

    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms

        self.images = []
        self.labels = []

        labels_path = os.path.join(
            hparams.dataset_root, 'fishyscapes_lostandfound')
        label_files = os.listdir(labels_path)
        label_files.sort()
        for lbl in label_files:

            self.labels.extend([os.path.join(labels_path, lbl)])
            img_name = lbl[5:-10] + 'leftImg8bit.png'
            self.images.extend(
                [os.path.join(hparams.dataset_root, 'laf_images', img_name)])

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        label = label[:, :, 0]

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples


class FishyscapesStatic(Dataset):
    """
    The dataset folder is assumed to follow the following structure. In the given root folder there must be two
    sub-folders:
    - fs_val_v1 (or fs_val_v2): contains the mask labels in .png format
    - fs_static_images_v1 (or fs_static_images_v2): contains the images also in .png format. These images need a processing step to be created from
    cityscapes. the fs_val_v3 file contains .npz files that contain numpy arrays. According to ID of each file, the
    corresponding image from cityscapes should be loaded and then the cityscape image and the image from the .npz file
    should be summed to form the modified image, which should be stored in fs_static_images folder. The images files are
    named using the label file name as follows: img_name = label_name[:-10] + 'rgb.png'
    """

    def __init__(self, hparams, transforms, version):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms
        self.version = version

        if version not in [1, 2]:
            raise ValueError(
                f"Supported versions for Fishyscapes Static currently are 1 and 2, given was {version}")

        labels_root = os.path.join(hparams.dataset_root, f'fs_val_v{version}')
        images_root = os.path.join(
            hparams.dataset_root, f'fs_static_images_v{version}')
        files = os.listdir(labels_root)

        self.images = []
        self.labels = []
        for f in files:
            if f[-3:] != 'png':
                continue

            self.labels.extend([os.path.join(labels_root, f)])
            image_path = os.path.join(images_root, f[:-10] + 'rgb.png')
            self.images.extend([image_path])

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        label = label[:, :, 0]

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples
