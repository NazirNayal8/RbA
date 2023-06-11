#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import numpy as np
import json
import time
from collections import defaultdict
import argparse
import multiprocessing
import copy
from pycocotools import mask as maskUtils
from tabulate import tabulate
import torch

from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.evaluation.panoptic_evaluation import logger
from detectron2.data import MetadataCatalog

import PIL.Image as Image
from panopticapi.utils import get_traceback, rgb2id
import matplotlib.pyplot as plt

OFFSET = 256 * 256 * 256
VOID = 0

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

# gt_data = json.load(
#     open(os.path.join(_root, 'coco/annotations/panoptic_val2017.json'), 'r'))
# id2cat = {}
# for gt in gt_data['annotations']:
#     for g in gt['segments_info']:
#         id2cat[g['id']] = g['category_id']
meta = MetadataCatalog.get(
    'coco_2017_val_panoptic_separated').thing_dataset_id_to_contiguous_id


class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self
    #                   isthing  isunknown
    # [("All",            None,  None), 
    #  ("Known Things",   True,  False),
    #  ("Unknown Things", True,  True), 
    #  ("Stuff",          False, None)]
    def pq_average(self, categories, isthing, isunknown):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
                cat_isunknown = label_info['id'] == 255
                if isunknown is None:  # Things
                    if label_info['id'] < -1:
                        continue
                elif isunknown:  # Unknwon
                    if isunknown != cat_isunknown:
                        continue
                elif label_info['id'] <= -1 or label_info["id"] == 255:  # Known
                    continue
            elif label_info['id'] < 0 or label_info["id"] == 255:
                continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {
                'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            n += 1
            pq += pq_class
            sq += sq_class
            rq += rq_class
        if n == 0:
            return {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'n': 0}, per_class_results
        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(
                proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(
            gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(
            pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(
                    gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(
                    gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(
                gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * \
            OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - \
                intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                if 'original_category_id' in gt_segms[gt_label]:
                    pq_stat[gt_segms[gt_label]['original_category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]
                            ['original_category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1
            if 'original_category_id' in gt_info:
                pq_stat[gt_info['original_category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get(
                    (crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    print('Core: {}, all {} images processed'.format(
        proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(
        cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)

    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None,
               unknown_label_list=None):
    start_time = time.time()

    # load ground truth and predictions
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')

    categories = {el['id']: el for el in gt_json['categories']}
    if unknown_label_list is not None:
        known_categories = {}
        unknown_category_id = []
        for c in categories:
            if categories[c]['name'] not in unknown_label_list:
                known_categories[c] = categories[c]
            else:
                unknown_category_id.append(categories[c]['id'])
                if True:
                    unknown_cat = categories[c]
                    unknown_cat['supercategory'] = 'unknown_' + \
                        unknown_cat['supercategory']
                    unknown_cat['id'] = -unknown_cat['id'] - 1  #
                    unknown_cat['name'] = 'unknown_' + unknown_cat['name']
                    known_categories[-c-1] = unknown_cat
        known_categories[255] = {
            'supercategory': 'unknown',
            'isthing': 1,
            'id': 255,
            'name': 'unknown'
            }
        categories = known_categories
        annos = gt_json['annotations']
        for ann in annos:
            for instance in ann['segments_info']:
                if instance['category_id'] in unknown_category_id:
                    instance['original_category_id'] = -instance['category_id']-1
                    instance['category_id'] = 255

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception(
            "Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception(
            "Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    # check if each ground truth image has a corresponding predicted image.
    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception(
                'no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(
        matched_annotations_list, gt_folder, pred_folder, categories)
    metrics = [("All", None, None), ("Known Things", True, False),
               ("Unknown Things", True, True), ("Stuff", False, None)]
    results = {}

    for name, isthing, isunknown in metrics:
        results[name], per_class_results = pq_stat.pq_average(
            categories, isthing=isthing, isunknown=isunknown)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s}   {:>5s}".format(
        "", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))
    for name, _isthing, _isunknown in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )
    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))
    return results


def _print_panoptic_results(pq_res, open=False):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    if open:
        names = ["All", "Things", "Stuff"]
    else:
        names = ["All", "Known Things", "Unknown Things", "Stuff"]
    for name in names:
        row = [name] + [pq_res[name][k] *
                        100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
    logger.info("Panoptic Evaluation Results:\n" + table)
    print("Panoptic Evaluation Results:\n" + table)


class COCOOpeneval(COCOeval):
    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def set_task(self, task):
        self.unknown = 'unknown' in task

    def pr_evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        iouThrs = np.linspace(.0, 0.95, int(
            np.round((0.95 - .0) / .05)) + 1, endpoint=True)

        T = len(iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1-1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                     for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(
            dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }

    def pr_evaluate(self, name='test'):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()  # flush
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds
                     for catId in catIds}

        evaluateImg = self.pr_evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
        return self.pr_curve(name=name)

    def pr_curve(self, p=None, name='test'):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]

        iouThrs = np.linspace(.0, 0.95, int(
            np.round((0.95 - .0) / .05)) + 1, endpoint=True)
        recThrs = np.linspace(.0, 1.00, int(
            np.round((1.00 - .0) / .01)) + 1, endpoint=True)

        T = len(iouThrs)
        R = len(recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        # -1 for the precision of absent categories
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))
        fps = []
        tps = []
        log_fps = [[[[[] for _ in range(M)] for _ in range(A)]
                    for _ in range(K)] for _ in range(T)]
        log_tps = [[[[[] for _ in range(M)] for _ in range(A)]
                    for _ in range(K)] for _ in range(T)]
        log_fps_cum_sum = [
            [[[[-1] for _ in range(M)] for _ in range(A)] for _ in range(K)] for _ in range(T)]
        log_tps_cum_sum = [
            [[[[-1] for _ in range(M)] for _ in range(A)] for _ in range(K)] for _ in range(T)]
        log_precisions = [
            [[[[-1] for _ in range(M)] for _ in range(A)] for _ in range(K)] for _ in range(T)]
        log_recalls = [
            [[[[-1] for _ in range(M)] for _ in range(A)] for _ in range(K)] for _ in range(T)]
        log_scores = [[[[-1] for _ in range(M)]
                       for _ in range(A)] for _ in range(K)]
        log_npigs = [0 for _ in range(A)]

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(
            map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet]
                                         for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if m == 0:
                        log_npigs[a] += npig
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm,  np.logical_not(dtIg))
                    fps = np.logical_and(
                        np.logical_not(dtm), np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    log_scores[k][a][m] = dtScoresSorted
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        log_fps_cum_sum[t][k][a][m] = fp
                        log_tps_cum_sum[t][k][a][m] = tp
                        log_recalls[t][k][a][m] = rc
                        log_precisions[t][k][a][m] = pr
                        log_fps[t][k][a][m] = fps[t]
                        log_tps[t][k][a][m] = tps[t]

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()
                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]
                        inds = np.searchsorted(rc, recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        logs = {
            'fps_cum': log_fps_cum_sum,
            'tps_cum': log_tps_cum_sum,
            'tps': log_tps,
            'fps': log_fps,
            'recalls': log_recalls,
            'precisions': log_precisions,
            'scores': log_scores
        }
        # t = iouThrs[10] = 0.5
        # k = cls
        # A = ['all', 'small', 'medium', 'large']
        # M = [1, 10, 100]

        log_all_fps = [[0 for _ in range(A)] for _ in range(T)]
        log_all_tps = [[0 for _ in range(A)] for _ in range(T)]
        log_all_recalls = [[0 for _ in range(A)] for _ in range(T)]
        log_all_precisions = [[0 for _ in range(A)] for _ in range(T)]
        for a in range(A):
            for t in range(T):
                tps = []
                fps = []
                scores = []
                for k in range(K):
                    tp = log_tps[t][k][a][-1]
                    fp = log_fps[t][k][a][-1]
                    s = log_scores[k][a][-1]
                    if len(tp) == len(s):
                        scores.append(s)
                        tps.append(tp)
                        fps.append(fp)

                scores = np.concatenate(scores)
                tps = np.concatenate(tps)
                fps = np.concatenate(fps)

                inds = np.argsort(-scores, kind='mergesort')
                scores = scores[inds]
                tps = tps[inds]
                fps = fps[inds]
                tp = np.cumsum(tps, axis=0).astype(dtype=np.float)
                fp = np.cumsum(fps, axis=0).astype(dtype=np.float)
                rc = tp / log_npigs[a]
                pr = tp / (fp+tp+np.spacing(1))
                log_all_fps[t][a] = fp
                log_all_tps[t][a] = tp
                log_all_recalls[t][a] = rc
                log_all_precisions[t][a] = pr
        logs['all_fps_cum'] = log_all_fps
        logs['all_tps_cum'] = log_all_tps
        logs['all_recalls'] = log_all_recalls
        logs['all_precisions'] = log_all_precisions

        self.draw_pr_curve(logs, name)
        return None

    def draw_pr_curve(self, logs, name='test'):
        A = ['all', 'small', 'medium', 'large']

        precisions = logs['all_precisions']
        recalls = logs['all_recalls']
        tps = logs['all_tps_cum']
        fps = logs['all_fps_cum']
        plt.figure(figsize=(9, 8))
        plt.subplot(211)
        for i, a in enumerate(A):
            pr = precisions[10][i]
            rec = recalls[10][i]
            plt.plot(rec, pr, lw=2, label='{}'.format(a))

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc='best')
        plt.subplot(212)
        for i, a in enumerate(A):
            fp = fps[10][i]
            tp = tps[10][i]
            fp = fp / max(fp)
            tp = tp / max(tp)
            plt.plot(fp, tp, lw=2, label='{}'.format(a))

        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.legend(loc='best')

        plt.savefig('{}.png'.format(name))


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None, task='bbox'):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOOpeneval(coco_gt, coco_dt, iou_type)
    coco_eval.set_task(task)

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params,
                           "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(
            next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

#    coco_eval.pr_evaluate(name='plots/'+task)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def get_gt(filename, panoptic_r, unseen_label_set=None):
    filename = filename.replace(
        '.jpg', '.png').replace('/val', '/panoptic_val')
    segment_map, inst_info = panoptic_r
    gt = torch.tensor(rgb2id(np.asarray(Image.open(filename),
                      dtype=np.uint32)).astype(int), device=segment_map.device)
    new_inst_info = []
    for inst in inst_info:
        if inst['category_id'] != -1:
            new_inst_info.append(inst)
            continue
        mask = segment_map == inst['id']
        area = mask.sum().float()
        gt_overlap = gt[mask]
        ids, counts = gt_overlap.unique(return_counts=True)
        idx = torch.argmax(counts)
        if counts[idx] >= 0.5 * area:
            if ids[idx] == 0:
                segment_map[mask] = 0
                continue
            id = id2cat[ids[idx].item()]
            id = id if id not in meta else meta[id]
            if id not in unseen_label_set:
                segment_map[mask] = 0
                continue
            # find id class.
        new_inst_info.append(inst)
    return segment_map, new_inst_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    args = parser.parse_args()
    pq_compute(args.gt_json_file, args.pred_json_file,
               args.gt_folder, args.pred_folder)
