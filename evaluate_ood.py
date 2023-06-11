import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pickle
import matplotlib.image as mpimg
import albumentations as A
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.cityscapes import Cityscapes
from datasets.bdd100k import BDD100KSeg
from datasets.road_anomaly import RoadAnomaly
from datasets.fishyscapes import FishyscapesLAF, FishyscapesStatic
from datasets.lost_and_found import LostAndFound
from train_net import Trainer, setup
from detectron2.checkpoint import DetectionCheckpointer
from pprint import pprint
from support import get_datasets, OODEvaluator

from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='OOD Evaluation')

parser.add_argument('--batch_size', type=int, default=1,
                    help="Batch Size used in evaluation")
parser.add_argument('--num_workers', type=int, default=15,
                    help="Number of threads used in data loader")
parser.add_argument('--device', type=str, default='cuda',
                    help="cpu or cuda, the device used for evaluation")
parser.add_argument('--out_path', type=str, default='results',
                    help='output file for saving the results as a pickel file')
parser.add_argument('--verbose', type=bool, default=True,
                    help="If True, the records will be printed every time they are saved")
parser.add_argument('--datasets_folder', type=str, default='./',
                    help='the path to the folder that contains all datasets for evaluation')
parser.add_argument('--models_folder', type=str, default='ckpts/',
                    help='the path that contains the models to be evaluated')
parser.add_argument("--store_anomaly_scores", action='store_true',
                    help="""If passed, store anomaly score maps that are extracted in full evaluation. 
                    The map will be stored in a folder for each model, and under it a folder for each dataset. 
                    All will be stored under anomaly_scores/ folder""")
parser.add_argument('--model_mode', type=str, default='all',
                    help="""One of [all, selective]. Defines which models to evaluate, the default behavior is all, which is to 
                            evaluate all models in model_logs dir. You can also choose particular models
                            for evaluation, in which case you need to pass the names of the models to --selected_models""")
parser.add_argument("--selected_models", nargs="*", type=str, default=[],
                    help="Names of models to be evaluated, these should be name of directories in model_logs")
parser.add_argument('--dataset_mode', type=str, default='all',
                    help="""One of [all, selective]. Defines which datasets to evaluate on, the default behavior is all, which is to 
                            evaluate all available datasets. You can also choose particular datasets
                            for evaluation, in which case you need to pass the names of the datasets to --selected_datasets.
                            Available Datasets are: [
                                road_anomaly,
                                fishyscapes_laf,
                            ]
                            """)
parser.add_argument("--selected_datasets", nargs="*", type=str, default=[],
                    help="""Names of datasets to be evaluated.
                        Available Datasets are: [
                                road_anomaly,
                                fishyscapes_laf,
                            ]
                    """)
parser.add_argument("--score_func", type=str, default="rba", choices=["rba", "pebal", "dense_hybrid"],
                    help="outlier scoring function to be used in evaluations")
    

args = parser.parse_args()

DATASETS = get_datasets(args.datasets_folder)

dataset_group = [(name, dataset) for (name, dataset) in DATASETS.items() ]

# filter dataset group according to chosen option
if args.dataset_mode == 'selective':
    dataset_group = [g for g in dataset_group if g[0]
                     in args.selected_datasets]
    if len(dataset_group) == 0:
        raise ValueError(
            "Selective Mode is chosen but number of selected datasets is 0")
else:
    dataset_group = [g for g in dataset_group if g[0] in ['road_anomaly', 'fishyscapes_laf']]

print("Datasets to be evaluated:")
[print(g[0]) for g in dataset_group]
print("-----------------------")


# Dictionary for saving the results
# records will be a nested dictionary with the following hierarchy:
# - model_name:
#   - dataset_name:
#       - metric_name:
#           -mean:
#           -std:
#           -value: this shows the value without bootstrapping

# Device for computation
if args.device == 'cuda' and (not torch.cuda.is_available()):
    print("Warning: Cuda is requested but cuda is not available. CPU will be used.")
    args.device = 'cpu'
DEVICE = torch.device(args.device)


def get_model(config_path, model_path):
    """
    Creates a Mask2Former model give a config path and ckpt path
    """
    args = edict({'config_file': config_path, 'eval-only': True, 'opts': [
        "OUTPUT_DIR", "output/",
    ]})
    config = setup(args)

    model = Trainer.build_model(config)
    DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR).resume_or_load(
        model_path, resume=False
    )
    model.to(DEVICE)
    _ = model.eval()

    return model


def get_logits(model, x, **kwargs):
    """
    Extracts the logits of a single image from Mask2Former Model. Works only for a single image currently.

    Expected input:
    - x: torch.Tensor of shape (1, 3, H, W)

    Expected output:
    - Logits (torch.Tensor) of shape (1, 19, H, W)
    """
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])

    return out[0]['sem_seg'].unsqueeze(0)


def get_RbA(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])

    logits = out[0]['sem_seg']
    
    return -logits.tanh().sum(dim=0)

def get_energy(model, x, **kwargs):

    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])

    logits = out[0]['sem_seg']

    return -torch.logsumexp(logits, dim=0)

def get_densehybrid_score(model, x, **kwargs):

    with torch.no_grad():
        out, ood_pred = model([{"image": x[0].cuda()}], return_ood_pred=True)

    logits = out[0]['sem_seg']
    
    out = F.softmax(ood_pred, dim=1)
    p1 = torch.logsumexp(logits, dim=0)
    p2 = out[:, 1] # p(~din|x)
    probs = (- p1) + (p2 + 1e-9).log()
    conf_probs = probs
    return conf_probs

def save_dict(d, name):
    """
    Save the records into args.out_path. 
    Print the records to console if verbose=True
    """
    if args.verbose:
        pprint(d)
    store_path = os.path.join(args.out_path, name)
    Path(store_path).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(store_path, f'results.pkl'), 'wb') as f:
        pickle.dump(d, f)


def current_result_exists(model_name):
    """
    Check if the current results exist in the args.out_path
    """
    store_path = os.path.join(args.out_path, model_name)
    return os.path.exists(os.path.join(store_path, f'results.pkl'))

def run_evaluations(model, dataset):
    """
    Run evaluations for a particular model over all designated datasets.
    """

    score_func = None
    if args.score_func == "rba":
        score_func = get_RbA
    elif args.score_func == "pebal":
        score_func = get_energy
    elif args.score_func == "dense_hybrid":
        score_func = get_densehybrid_score

    evaluator = OODEvaluator(model, get_logits, score_func)

    loader = DataLoader(
        dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    anomaly_score, ood_gts = evaluator.compute_anomaly_scores(
        loader=loader,
        device=DEVICE,
        return_preds=False,
        upper_limit=1300
    )

    metrics = evaluator.evaluate_ood(
        anomaly_score=anomaly_score,
        ood_gts=ood_gts,
        verbose=False
    )

    return metrics


def main():

    # The name of every directory inside args.models_folder is expected to be the model name.
    # Inside a model's folder there should be 2 files (doesn't matter if there are extra stuff).
    # these 2 files are: config.yaml and [model_final.pth or model_final.pkl]

    models_list = os.listdir(args.models_folder)
    models_list = [m for m in models_list if os.path.isdir(
        os.path.join(args.models_folder, m))]
    print(args.models_folder)
    if args.model_mode == 'selective':
        models_list = [m for m in models_list if m in args.selected_models]

    if len(models_list) == 0:
        raise ValueError(
            "Number of models chosen is 0, either model_logs folder is empty or no models were selected")

    print("Evaluating the following Models:")
    [print(m) for m in models_list]
    print("-----------------------")

    for model_name in models_list:
        experiment_path = os.path.join(args.models_folder, model_name)

        results = edict()

        config_path = os.path.join(experiment_path, 'config.yaml')
        model_path= os.path.join(experiment_path, 'model_final.pth')

        if current_result_exists(model_name):
            print(f"Skipping {model_name} because results already exist, if you want to re-run, delete the results.pkl file")
            continue

        if not os.path.exists(model_path):
            model_path = os.path.join(
                'model_logs', model_name, 'model_final.pkl')

            if not os.path.exists(model_path):
                print("Model path does not exist, skipping")
                continue

        model = get_model(config_path=config_path, model_path=model_path)

        for dataset_name, dataset in dataset_group:

            if dataset_name not in results:
                results[dataset_name] = edict()

            results[dataset_name] = run_evaluations(model, dataset)
        
        save_dict(results, model_name)

if __name__ == '__main__':
    main()
