import copy
import logging
import cv2
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

# from util.misc import filter_unseen_class, cum_map
import fvcore.transforms.transform as FT

__all__ = ["OpenPanopticCOCODatasetMapper"]


def cum_map(sem_seg, ignore_value=255):
    H, W = sem_seg.shape[-2:]
    one_hot = sem_seg.clone()
    one_hot[one_hot != ignore_value] = 0
    one_hot[one_hot == ignore_value] = 1
    ret =  []
    if len(sem_seg.shape) > 2:
        for m in one_hot:
            sem_seg_target = cv2.integral(m.cpu().numpy().astype('uint8'))
            ret.append(sem_seg_target)
    else:
        ret = cv2.integral(one_hot.numpy().astype('uint8'))
    sem_seg_target = torch.tensor(ret, device=sem_seg.device).float()
    return sem_seg_target

def filter_unseen_class(instances, unseen_label_set):
    gt_classes = instances.gt_classes
    filtered_idx = []
    for i, c in enumerate(gt_classes):
        if c not in unseen_label_set:
            filtered_idx.append(i)

    return instances[filtered_idx]

def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class OpenPanopticCOCODatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by EOPSN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.unlabeled_region_on = cfg.MODEL.MASK_FORMER.UNLABELED_REGION

        # Semantic Segmentation
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.sem_seg_unlabeled_region_on = cfg.MODEL.MASK_FORMER.SEM_SEG_UNLABELED_REGION
        self.num_sem_seg_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        if unseen_path != '' and self.is_train:
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            self.unseen_label_set = self._get_unseen_label_set(meta, unseen_path)
        else:
            self.unseen_label_set = None

        if cfg.MODEL.LOAD_PROPOSALS:
            self.proposal_topk = (
                                    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                                    if is_train
                                    else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
                                )
        else:
            self.proposal_topk = None



    def _get_unseen_label_set(self, meta, path):
        meta = {e: i for i, e in enumerate(meta)}
        with open(path, 'r') as f:
            lines = [meta[e.replace('\n','')] for e in f.readlines()]

        return lines



    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        print("DATASET DICT", dataset_dict.keys())
        print("ANNOTATIONS", dataset_dict['annotations'])
        print("SEM SEG FILE NAME", dataset_dict['sem_seg_file_name'])
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        original_image  = image

        if self.crop_gen is None or np.random.rand() > 0.5:
            tfm_gens = self.tfm_gens
        else:
            tfm_gens = self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:]


        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(original_image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(tfm_gens)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            return dataset_dict

        if type(transforms[0]) is FT.NoOpTransform:
            flip = 0
        elif type(transforms[0]) is FT.HFlipTransform:
            flip = 1
        else:
            flip = 2
        dataset_dict["flip"] = flip

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.sem_seg_unlabeled_region_on:
                sem_seg_gt[sem_seg_gt==self.ignore_value] = self.num_sem_seg_classes
            dataset_dict["sem_seg"] = sem_seg_gt

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                   anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            if self.unseen_label_set is not None:
                dataset_dict["instances"] = filter_unseen_class(dataset_dict["instances"], self.unseen_label_set)

        if self.unlabeled_region_on:
            if self.sem_seg_unlabeled_region_on:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.num_sem_seg_classes)
            else:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.ignore_value)
            dataset_dict["integral_sem_seg"] = cum_sem_seg

        return dataset_dict
