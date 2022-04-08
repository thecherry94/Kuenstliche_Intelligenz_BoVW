import os
import cv2
import numpy as np

def load_mvtec_dataset(directory, object_type, resize_dim=0):
    """
    Loads the MVTEC dataset of the specified object type. (bottle, cable, capsule, etc.)
    """

    train_paths, test_paths, ground_truth_paths = load_mvtec_dataset_paths(directory, object_type)
    train_images, train_labels, test_images, test_labels = load_and_label_data((train_paths, test_paths), resize_dim)
    return train_images, train_labels, test_images, test_labels, ground_truth_paths


def load_mvtec_dataset_paths(directory, object_type):
    """
    Loads the MVTEC dataset of the specified object type. (bottle, cable, capsule, etc.)
    """

    # Check if directory exists
    if not os.path.isdir(directory):
        raise Exception(f"Can't load dataset: Directory \"{os.path.abspath(directory)}\" does not exist.")

    ground_truth_paths = {}
    test_paths = {'good': [], 'bad': []}
    train_paths = {'good': [], 'bad': []}

    # Extract all directories that only contain images with the specified object type
    filtered_paths = []
    for abs_path, sub_dirs, files in os.walk(directory):
        if len(sub_dirs) == 0:
            split_abs_path = abs_path.split('\\')
            if split_abs_path[1] == object_type:
                filtered_paths.append(abs_path)
    
    # Extract all image paths from the filtered directories
    for filtered_path in filtered_paths:
        split_abs_path = filtered_path.split('\\')
        type = split_abs_path[2]
        spec = split_abs_path[3]

        path_helper_func = lambda x: os.path.join(filtered_path, x)

        if type == 'ground_truth':
            ground_truth_paths[spec] = list(map(path_helper_func, os.listdir(filtered_path)))
        elif type == 'test':
            test_paths['bad' if spec != 'good' else 'good'].extend(list(map(path_helper_func, os.listdir(filtered_path))))
        elif type == 'train':
            train_paths['bad' if spec != 'good' else 'good'].extend(list(map(path_helper_func, os.listdir(filtered_path))))

    return train_paths, test_paths, ground_truth_paths


def load_and_label_data(dataset, resize_dim=0):
    train_paths, test_paths = dataset

    # Load all training images
    train_images = []
    train_labels = []
    for path in train_paths['good']:
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        train_images.append(image)
        train_labels.append(1)
    
    for path in train_paths['bad']:
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        train_images.append(image)
        train_labels.append(0)
    
    # Load all testing images
    test_images = []
    test_labels = []
    for path in test_paths['good']:
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        test_images.append(image)
        test_labels.append(1)
    
    for path in test_paths['bad']:
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        test_images.append(image)
        test_labels.append(0)
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)