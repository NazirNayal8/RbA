import os
import io
import contextlib
import json
import itertools
import tempfile

from collections import OrderedDict, defaultdict
from PIL import Image

import copy
import pickle
import logging
import numpy as np
import torch
import sys

# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, SemSegEvaluator
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.utils.logger import create_small_table
from detectron2.evaluation.panoptic_evaluation import COCOPanopticEvaluator
from detectron2.engine import default_argument_parser, default_setup


# from .config import add_config
from .evaluation import pq_compute, _evaluate_predictions_on_coco, _print_panoptic_results

logger = logging.getLogger('d2.evaluation.evaluator')


class SemSegOpenEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, distributed, num_classes, ignore_label=255, output_dir=None):
        super().__init__(dataset_name, distributed, num_classes, ignore_label, output_dir)
        if self._contiguous_id_to_dataset_id is not None:
            self._contiguous_id_to_dataset_id[54] = -1
        self._class_names.append('unknown')


class COCOOpenEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger('detectron2.evaluation.coco_evaluation')

        self._metadata = copy.deepcopy(MetadataCatalog.get(dataset_name))

        if cfg.DATASETS.UNSEEN_LABEL_SET != '':
            thing_classes = self._metadata.thing_classes
            del self._metadata.thing_classes
            thing_classes.append('unknown')
            self._metadata.thing_classes = thing_classes #['unknown']
            unknown_colors = self._metadata.thing_colors[-1]
            thing_colors = self._metadata.thing_colors
            thing_colors.append(unknown_colors)
            del self._metadata.thing_colors
            self._metadata.thing_colors = thing_colors # unknown_colors
            conv_id = self._metadata.thing_dataset_id_to_contiguous_id
            del self._metadata.thing_dataset_id_to_contiguous_id
            conv_id[-1] = -1
            self._metadata.thing_dataset_id_to_contiguous_id = conv_id #{-1:-1}


        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`."
                " Therefore trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
           self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

        if cfg.DATASETS.UNSEEN_LABEL_SET != '':
            with open(cfg.DATASETS.UNSEEN_LABEL_SET, 'r') as f:
                self.unknown_label_list = [e.replace('\n', '') for e in f.readlines()]
        else:
            self.unknown_label_list = None


    def _tasks_from_config(self, cfg):
        """
        Return:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.DATASETS.UNSEEN_LABEL_SET != '':
            tasks = tasks + ("known_bbox", "unknown_bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
            if cfg.DATASETS.UNSEEN_LABEL_SET != '':
                tasks = tasks + ("known_segm", "unknown_segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
            if cfg.DATASETS.UNSEEN_LABEL_SET != '':
                tasks = tasks + ("known_keypoints", "unknown_keypoints")
        return tasks


    def set_prediction(self, path): # for debug
        with open(path, 'rb') as f:
            self._predictions = pickle.load(f)


    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")

        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return


        known_coco_api, known_coco_results = self.convert_to_subset(coco_results, task="known")
        unknown_coco_api, unknown_coco_results = self.convert_to_subset(coco_results, task="unknown")

        self._logger.info("Evaluating predictions ...")
        class_names = self._metadata.get("thing_classes")[:-1]
        for task in sorted(tasks):
            class_names = self._metadata.get("thing_classes")[:-1]
            if 'known' in task:
                task_coco_api = known_coco_api
                task_coco_results = known_coco_results
                coco_task = task.split('_')[1]
                class_names = [c for c in class_names if c not in self.unknown_label_list]
                if 'unknown' in task:
                    task_coco_api = unknown_coco_api
                    task_coco_results = unknown_coco_results
                    coco_task = task.split('_')[1]
                    class_names = ['unknown']
            else:
                task_coco_api = self._coco_api
                task_coco_results = coco_results
                coco_task = task
            coco_eval = (
                _evaluate_predictions_on_coco(
                    task_coco_api, task_coco_results, coco_task, kpt_oks_sigmas=self._kpt_oks_sigmas,
                    task=task
                )
                if len(task_coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, coco_task, class_names=class_names
            )
            res2 = self._derive_coco_results_ar(
                coco_eval, coco_task, class_names=class_names
            )
            self._results[task] = res
            self._results[task].update(res2)


    def _derive_coco_results_ar(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AR", "ARs", "ARm", "ARl"],
            "segm": ["AR", "ARs", "ARm", "ARl"],
            "keypoints": ["AR", "ARm", "ARl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        t = -3 if iou_type == "keypoints" else -4
        results = {
            metric: float(coco_eval.stats[idx+t] * 100 if coco_eval.stats[idx+t] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }


        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AR
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        recalls = coco_eval.eval["recall"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == recalls.shape[1]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")
            results_per_category.append(("{}".format(name), float(ar * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AR: \n".format(iou_type) + table)
        results.update({"AR-" + name: ar for name, ar in results_per_category})
        return results

    def convert_to_subset(self, coco_results, task="known"):
        new_coco_gt = COCOOpen(copy.deepcopy(self._coco_api.dataset), self.unknown_label_list, task)
        if task == "known":
            new_coco_results = list(filter(lambda x : x['category_id'] >= 0, coco_results))
        else:
            new_coco_results = list(filter(lambda x : x['category_id'] < 0, coco_results))

        return new_coco_gt, new_coco_results


class COCOOpen(COCO):
    def __init__(self, dataset, unknown_label_list=None, task='known'):
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        self.dataset = self._annotation_filtering(dataset, unknown_label_list, task)
        self.createIndex()

    def _annotation_filtering(self, dataset, unknown_label_list, task):
        unknown_categories = [c['id'] for c in dataset['categories'] if c['name'] in unknown_label_list]
        annotations = []
        if task =='known':
            for anno in dataset['annotations']:
                if anno['category_id'] not in unknown_categories:
                    annotations.append(anno)
            categories = []
            for c in dataset['categories']:
                if c['id'] not in unknown_categories:
                    categories.append(c)

        if task =='unknown':
            for anno in dataset['annotations']:
                if anno['category_id'] in unknown_categories:
                    anno['original_category_id'] = anno['category_id']
                    anno['category_id'] = -1
                    annotations.append(anno)
            categories = []

            categories.append({
                            'supercategory': 'unknown',
                            'id': -1,
                            'name': 'unknown'
                            })
        dataset['categories'] = categories
        dataset['annotations'] = annotations
        return dataset


class COCOPanopticOpenEvaluator(COCOPanopticEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name, output_dir, cfg=None):
        """
        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save results for evaluation
        """
        super().__init__(dataset_name, output_dir)

        # read directly from MetaDataCatalog 
        self._metadata = copy.deepcopy(self._metadata)

        if cfg.DATASETS.UNSEEN_LABEL_SET != '':
            thing_classes = self._metadata.thing_classes
            del self._metadata.thing_classes
            
            # unknown is added as the last class
            thing_classes.append('unknown')
            # overwrite the metadata with thing_classes + unknown
            self._metadata.thing_classes = thing_classes #['unknown']

            # add last color from thing_colors as the unknown color
            unknown_colors = self._metadata.thing_colors[-1]
            thing_colors = self._metadata.thing_colors
            thing_colors.append(unknown_colors)
            del self._metadata.thing_colors
            self._metadata.thing_colors = thing_colors # unknown_colors

            # update mapping from dataset_id to contiguous_id so that unknown ID is -1 to -1
            conv_id = self._metadata.thing_dataset_id_to_contiguous_id
            del self._metadata.thing_dataset_id_to_contiguous_id
            conv_id[-1] = -1
            self._metadata.thing_dataset_id_to_contiguous_id = conv_id #{-1:-1}

            with open(cfg.DATASETS.UNSEEN_LABEL_SET, 'r') as f:
                self.unknown_label_list = [e.replace('\n', '') for e in f.readlines()]  # names of unknown classes
        else:
            self.unknown_label_list = None
        # Add unknown class id
        self._thing_contiguous_id_to_dataset_id[255] = 255
        self._stuff_contiguous_id_to_dataset_id[54] = -1
        self._predictions_json = os.path.join(output_dir, "predictions.json")


    def process(self, inputs, outputs):
        """
        Process a single batch of inputs and outputs of the Panoptic Model, and then save it to the disk.
        The self._predictions is a list of dict, each dict contains information for a single sample, 
        the information includes path to output on disk and the segments_info inside each sample.

        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN)
            outputs: the outputs of a COCO model (e.g., GeneralizedRCNN)

        """
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def set_prediction(self, path): # for debug
        with PathManager.open(self._predictions_json, "r") as f:
            self._predictions = json.loads(f.readlines()[0])['annotations']


    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)
        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions
            with PathManager.open(self._predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,  # path to ground truth annotations  
                    PathManager.get_local_path(self._predictions_json),  # path to predictions file
                    gt_folder=gt_folder, # path to gt folder
                    pred_folder=pred_dir, # path to pred folder
                    unknown_label_list=self.unknown_label_list  # list of unknown labels
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        if 'Known Things' in pq_res:
            res["PQ_known"] = 100 * pq_res["Known Things"]["pq"]
            res["SQ_known"] = 100 * pq_res["Known Things"]["sq"]
            res["RQ_known"] = 100 * pq_res["Known Things"]["rq"]
        if "Unknown Things" in pq_res:
            res["PQ_unknown"] = 100 * pq_res["Unknown Things"]["pq"]
            res["SQ_unknown"] = 100 * pq_res["Unknown Things"]["sq"]
            res["RQ_unknown"] = 100 * pq_res["Unknown Things"]["rq"]
        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results





# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     add_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(cfg, args)
#     return cfg


# if __name__ == "__main__":

#     args = default_argument_parser().parse_args()
#     cfg = setup(args)
#     dataset_name = 'coco_2017_val_panoptic_separated'
#     coco_eval = COCOPanopticOpenEvaluator(dataset_name, 'model/inference', cfg)
#     coco_eval.set_prediction('data.pkl')
#     result = coco_eval.evaluate()
