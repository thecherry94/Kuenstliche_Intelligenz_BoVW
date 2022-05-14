import os
from utils import recopy_mvtec

dest_directory = 'mvtec_anomaly_detection_data_yolo'
img_directory = 'mvtec_anomaly_detection_data'

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

create_dir = 1
test_val_size = 0.25
recopy_mvtec(dest_directory, img_directory,create_dir, MVTEC_CATEGORIES, test_val_size, 512)



