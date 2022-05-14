import os
from utils import recopy_mvtec_yolo
from utils import recopy_mvtec_resnet

dest_directory_resnet = 'mvtec_anomaly_detection_data_yolo'
dest_directory_yolo = 'mvtec_anomaly_detection_data_yolo'
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
#recopy_mvtec_resnet(dest_directory_yolo, img_directory,create_dir, MVTEC_CATEGORIES, 0.2, 512)
recopy_mvtec_yolo(dest_directory_yolo, img_directory,create_dir, MVTEC_CATEGORIES, 0.2, 512)



