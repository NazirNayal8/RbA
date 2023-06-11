import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog


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


def read_data_file(root, path):

    with open(os.path.join(root, path)) as f:
        data = json.load(f)

    return data


def load_street_hazards(root, path):
    
    info = read_data_file(root, path)
    data = []
    for d in info:
        data.extend([{
            "file_name": os.path.join(root, "train", d["fpath_img"]),
            "sem_seg_file_name": os.path.join(root, "train", d["fpath_segm"]),
            "height": d["height"],
            "width": d["width"],
        }])

    return data
    

def get_street_hazards_meta():
    
    return {
        "thing_classes": class_names,
        "stuff_classes": class_names,
    }

_splits = [
    ("street_hazards_sem_seg_train", 'train/train.odgt'),
    ("street_hazards_sem_seg_val", 'train/validation_modified.odgt'),
]


def register_street_hazards(root):
    
    root = os.path.join(root, 'StreetHazards')
    for name, files_path in _splits:

        DatasetCatalog.register(
            name, lambda root=root, path=files_path: load_street_hazards(root, path)
        )
        MetadataCatalog.get(name).set(
            image_root=os.path.join(root, "train/images"),
            sem_seg_root=os.path.join(root, "train/annotations"),
            evaluator_type="sem_seg",
            ignore_label=12,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
            **get_street_hazards_meta(),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_street_hazards(_root)