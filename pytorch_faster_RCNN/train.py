import pandas as pd
import numpy as np
import cv2
import os
import re
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

import pickle
import json
import sys

sys.path.append('..')
import utils


from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


from MVTec_Dataset import MVTEC_Dataset, Averager


import argparse



def main():
    
    image_size = -1 # FROM IMAGE
    DATASET_PATH = '../mvtec_anomaly_detection_data'
    num_epochs = 300 
    learning_rate = 0.0025
    to_load = 'tile' # PARAMETER
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=DATASET_PATH, help='path to dataset')
    parser.add_argument('-category', type=str, required=True, help='category to train on (for example bottle, screw, thread, etc...')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='number of epochs')
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
    
    args = parser.parse_args()
    
    DATASET_PATH = args.dataset_dir
    to_load = args.category
    num_epochs = args.epochs
    learning_rate = args.lr

    print(f'Preparing to train on {to_load} for {num_epochs} epochs with a learning rate of {learning_rate}')

    cpu_device = torch.device("cpu")
    device = torch.device('cuda') if torch.cuda.is_available() else cpu_device
    
    print(f'Using device: {device}')

    object_dict = utils.get_object_dict(DATASET_PATH)
    class_dict = utils.get_class_dict(DATASET_PATH, to_load)

    class_dict_inv = utils.inv_dict(object_dict)

    dataset = utils.load_annotation_file(f'{class_dict_inv[to_load]}')
    random.shuffle(dataset)
    #train_ds, test_ds, val_ds = utils.train_test_split_annotations(dataset, 0.6, 0)
    train_ds, test_ds, val_ds = utils.train_test_split_annotations_even(dataset, 0.6, 0)

    image_size = cv2.imread(dataset[0][2]).shape[0]

    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # num classes + background
    num_classes = len(class_dict) + 1 

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    train_dataset = MVTEC_Dataset(train_ds, image_size, utils.get_train_transform(image_size))
    test_dataset = MVTEC_Dataset(test_ds, image_size, utils.get_valid_transform())
    val_dataset = MVTEC_Dataset(val_ds, image_size, utils.get_valid_transform())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=6,
        shuffle=True,
        collate_fn=utils.collate_fn
    )


    test_data_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    """
    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=utils.collate_fn
    )
    """






    print('Starting training...')

    # TRAINING

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = None

    loss_hist = Averager()
    itr = 1

    for epoch in range(num_epochs):
        loss_hist.reset()
        
        for images, targets, image_ids in train_data_loader:
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")



    print('Finished training. Saving model...')
    # save model 
    if not os.path.isdir('models'):
        os.mkdir('models')
    print_lr = str(learning_rate).replace('.', '_')
    torch.save(model.state_dict(), f'models/{to_load}_lr{print_lr}_epochs{num_epochs}_model.pt')
    print('Model saved.')


    print('Creating confusion matrix...')

    # CONFUSION MATRIX
    model.eval()

    y_pred = []
    y_true = []

    for images, targets, image_ids in test_data_loader:
        
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        exclude = []
        
        #print(outputs)
        pred_labels = []
        for i, item in enumerate(outputs):
            if item['labels'].cpu().numpy().size == 0:
                exclude.append(i)
                continue
            pred_labels.append(item['labels'][0].cpu().numpy())
        y_pred.extend(pred_labels)
        
        true_labels = []
        for i, item in enumerate(targets):
            if i in exclude:
                continue
            true_labels.append(item['labels'][0].cpu().numpy())
        y_true.extend(true_labels)
        

    cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in class_dict.values()], columns = [i for i in class_dict.values()])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, vmax=1, vmin=0)
    
    if not os.path.isdir('cm'):
        os.mkdir('cm')
    plt.savefig(f'cm/confusion_matrix_{to_load}.png')
    
    print('Confusion matrix saved.')
    print(f'Done with model training of {to_load}')
    
if __name__ == '__main__':
    main()