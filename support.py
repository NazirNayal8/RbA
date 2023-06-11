import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from albumentations.pytorch import ToTensorV2
from easydict import EasyDict as edict

from datasets.cityscapes import Cityscapes
from datasets.bdd100k import BDD100KSeg
from datasets.road_anomaly import RoadAnomaly
from datasets.fishyscapes import FishyscapesLAF, FishyscapesStatic
from datasets.segment_me_if_you_can import RoadAnomaly21, RoadObstacle21
from datasets.lost_and_found import LostAndFound
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from typing import Callable
from sklearn.metrics import roc_curve, auc, average_precision_score
from ood_metrics import fpr_at_95_tpr


def get_datasets(datasets_folder):

    # Configs for Datasets
    bdd100k_config = edict(
        seg_downsample_rate=1,
        train_file='train_paths.txt',
        val_file='val_paths.txt',
        val_image_strategy='no_change',
        ignore_train_class=True,
        dataset_root=os.path.join(datasets_folder, 'bdd100k/seg')
    )

    cityscapes_config = edict(
        dataset_root=os.path.join(datasets_folder, 'cityscapes'),
    )

    road_anomaly_config = edict(
        dataset_root=os.path.join(datasets_folder,
                                'RoadAnomaly/RoadAnomaly_jpg'),
        test_image_strategy='no_change'
    )

    fishyscapes_laf_config = edict(
        dataset_root=os.path.join(datasets_folder, 'Fishyscapes'),
    )

    fishyscapes_static_config = edict(
        dataset_root=os.path.join(datasets_folder, 'Fishyscapes'),
    )

    road_anomaly_21_config = edict(
        dataset_root=os.path.join(datasets_folder,
                                'SegmentMeIfYouCan/dataset_AnomalyTrack'),
        dataset_mode='all'
    )

    road_obstacle_21_config = edict(
        dataset_root=os.path.join(datasets_folder,
                                'SegmentMeIfYouCan/dataset_ObstacleTrack'),
        dataset_mode='all'
    )

    laf_config = edict(
        dataset_root=os.path.join(datasets_folder, 'LostAndFound'),
    )

    transform = A.Compose([
        ToTensorV2()
    ])
    
    # Road Anomaly 21
    transform_ra_21 = A.Compose([
        A.Resize(height=720, width=1280),
        ToTensorV2()
    ])

    DATASETS = edict(
        cityscapes=Cityscapes(cityscapes_config, transform=transform, split='val', target_type='semantic'),
        bdd100k=BDD100KSeg(hparams=bdd100k_config, mode='val', transforms=transform, image_size=(720, 1280)),
        road_anomaly=RoadAnomaly(hparams=road_anomaly_config, transforms=transform),
        fishyscapes_laf=FishyscapesLAF(hparams=fishyscapes_laf_config, transforms=transform),
        fs_static=FishyscapesStatic(hparams=fishyscapes_static_config, transforms=transform, version=1),
        fs_static_v2=FishyscapesStatic(hparams=fishyscapes_static_config, transforms=transform, version=2),
        road_anomaly_21=RoadAnomaly21(hparams=road_anomaly_21_config, transforms=transform_ra_21),
        road_obstacles=RoadObstacle21(road_obstacle_21_config, transforms=transform),
        lost_and_found=LostAndFound(laf_config, transform) 
    )

    return DATASETS


def get_logits_plus(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].cuda()}], **kwargs)
    
    if "return_aux" in kwargs and kwargs["return_aux"]:
        return out[0][0]["sem_seg"].unsqueeze(0), out[1]

    return out[0]['sem_seg'].unsqueeze(0)

def get_logits(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])
    
    return out[0]['sem_seg'].unsqueeze(0)
    
def get_neg_logit_sum(model, x, **kwargs):
    """
    This function computes the negative logits sum of a given logits map as an anomaly score.

    Expected input:
    - model: detectron2 style pytorch model
    - x: image of shape (1, 3, H, W)

    Expected Output:
    - neg_logit_sum (torch.Tensor) of shape (H, W)
    """

    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])

    logits = out[0]['sem_seg']
    
    return -logits.sum(dim=0)


def get_RbA(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])

    logits = out[0]['sem_seg']
    
    return -logits.tanh().sum(dim=0)

def logistic(x, k=1, x0=0, L=1):
    
    return L/(1 + torch.exp(-k*(x-x0)))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_anns(anns, strength=0.35):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*strength)))

def get_seg_colormap(preds, colors):
    """
    Assuming preds.shape = (H,W)
    """
    H, W = preds.shape
    color_map = torch.zeros((H, W, 3)).long()
    
    for i in range(len(colors)):
        mask = (preds == i)
        if mask.sum() == 0:
            continue
        color_map[mask, :] = torch.tensor(colors[i])
    
    return color_map

def proc_img(img):
    
    if isinstance(img, torch.Tensor):
        ready_img = img.clone()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.permute(1, 2, 0)
        ready_img = ready_img.cpu()

    elif isinstance(img, np.ndarray):
        ready_img = img.copy()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.transpose(1, 2, 0)
    else:
        raise ValueError(
            f"Unsupported type for image: ({type(img)}), only supports numpy arrays and Pytorch Tensors")

    return ready_img

def resize_mask(m, shape):
    
    m = F.interpolate(
        m,
        size=(shape[0], shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return m


class OODEvaluator:

    def __init__(
        self,
        model: nn.Module,
        inference_func: Callable,
        anomaly_score_func: Callable,
    ):

        self.model = model
        self.inference_func = inference_func
        self.anomaly_score_func = anomaly_score_func

    def get_logits(self, x, **kwargs):
        return self.inference_func(self.model, x, **kwargs)

    def get_anomaly_score(self, x, **kwargs):
        return self.anomaly_score_func(self.model, x, **kwargs)

    def calculate_auroc(self, conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        # print('Started FPR search.')
        for i, j, k in zip(tpr, fpr, threshold):
            if i > 0.95:
                fpr_best = j
                break
        # print(k)
        return roc_auc, fpr_best, k

    def calculate_ood_metrics(self, out, label):

        # fpr, tpr, _ = roc_curve(label, out)

        prc_auc = average_precision_score(label, out)
        roc_auc, fpr, _ = self.calculate_auroc(out, label)
        # roc_auc = auc(fpr, tpr)
        # fpr = fpr_at_95_tpr(out, label)

        return roc_auc, prc_auc, fpr

    def evaluate_ood(self, anomaly_score, ood_gts, verbose=True):

        ood_gts = ood_gts.squeeze()
        anomaly_score = anomaly_score.squeeze()

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_score[ood_mask]
        ind_out = anomaly_score[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        if verbose:
            print(f"Calculating Metrics for {len(val_out)} Points ...")

        auroc, aupr, fpr = self.calculate_ood_metrics(val_out, val_label)

        if verbose:
            print(f'Max Logits: AUROC score: {auroc}')
            print(f'Max Logits: AUPRC score: {aupr}')
            print(f'Max Logits: FPR@TPR95: {fpr}')

        result = {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr
        }

        return result

    def evaluate_ood_bootstrapped(
        self,
        dataset,
        ratio,
        trials,
        device=torch.device('cpu'),
        batch_size=1,
        num_workers=10,
    ):
        results = edict()

        dataset_size = len(dataset)
        sample_size = int(dataset_size * ratio)

        for i in range(trials):

            indices = np.random.choice(
                np.arange(dataset_size), sample_size, replace=False)
            loader = DataLoader(Subset(dataset, indices),
                                batch_size=batch_size, num_workers=num_workers)

            anomaly_score, ood_gts = self.compute_anomaly_scores(
                loader=loader,
                device=device,
                return_preds=False
            )

            metrics = self.evaluate_ood(
                anomaly_score=anomaly_score,
                ood_gts=ood_gts,
                verbose=False
            )

            for k, v in metrics.items():
                if k not in results:
                    results[k] = []
                results[k].extend([v])

        means = edict()
        stds = edict()
        for k, v in results.items():

            values = np.array(v)
            means[k] = values.mean() * 100.0
            stds[k] = values.std() * 100.0

        return means, stds

    def compute_anomaly_scores(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):

        anomaly_score = []
        ood_gts = []
        predictions = []
        jj = 0
        if use_gaussian_smoothing:
            gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)

        for x, y in tqdm(loader, desc="Dataset Iteration"):

            if jj >= upper_limit:
                break
            jj += 1

            x = x.to(device)
            y = y.to(device)

            ood_gts.extend([y.cpu().numpy()])

            score = self.get_anomaly_score(x)  # -> (H, W)

            if use_gaussian_smoothing:
                score = gaussian_smoothing(score.unsqueeze(0)).squeeze(0)

            if return_preds:
                logits = self.get_logits(x)
                _, preds = logits[:, :19, :, :].max(dim=1)
                predictions.extend([preds.cpu().numpy()])

            anomaly_score.extend([score.cpu().numpy()])

        ood_gts = np.array(ood_gts)
        anomaly_score = np.array(anomaly_score)

        if return_preds:
            predictions = np.array(predictions)
            return anomaly_score, ood_gts, predictions

        return anomaly_score, ood_gts