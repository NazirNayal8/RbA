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


class LostAndFound(Dataset):

    def __init__(self, hparams, transforms, mode='test'):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms
        self.mode = mode

        self.images = []
        self.labels = []
        for root, _, filenames in os.walk(os.path.join(hparams.dataset_root, 'leftImg8bit', mode)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    filename_base = '_'.join(filename.split('_')[:-1])
                    city = '_'.join(filename.split('_')[:-3])
                    self.images.append(os.path.join(root, filename_base + '_leftImg8bit.png'))
                    target_root = os.path.join(hparams.dataset_root, 'gtCoarse', mode)
                    self.labels.append(os.path.join(target_root, city, filename_base + '_gtCoarse_labelTrainIds.png'))

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        label = label[:, :, 0]
        label[label == 1] -= 1
        label[label == 2] -= 1
        
        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):

        return self.num_samples
