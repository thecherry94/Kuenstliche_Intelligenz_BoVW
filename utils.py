import os
import cv2
import numpy as np

def load_mvtec_dataset(directory, object_type, resize_dim=0):
    """
    Loads the MVTEC dataset of the specified object type. (bottle, cable, capsule, etc.)

    :param str directory: The relative path to the dataset directory.
    :param str object_type: The object type to load. (bottle, cable, capsule, etc.)
    :param int resize_dim: The dimension to resize the images to. Default image sizes are 900x900 (0 for no resizing).
    :return: A tuple containing the training images, training labels, testing images, and testing labels.
    """

    train_paths, test_paths, ground_truth_paths = __filter_mvtec_dataset_paths(directory, object_type)
    train_images, train_labels, test_images, test_labels = __load_and_label_data((train_paths, test_paths), resize_dim)
    return train_images, train_labels, test_images, test_labels, ground_truth_paths


def __filter_mvtec_dataset_paths(directory, object_type):
    """
    Internal function. Shoudln't be called directly.
    
    Recursively walks through the entire dataset directory and returns the paths to all images of the specified object type.

    :param str directory: The relative path to the dataset directory.
    :param str object_type: The object type to load. (bottle, cable, capsule, etc.)
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


def __load_and_label_data(dataset, resize_dim=0):
    """
    Internal function. Shoudln't be called directly.

    Loads and labels the specified dataset.

    :param tuple dataset: A tuple containing the training and testing image paths.
    :param int resize_dim: The dimension to resize the images to. Default image sizes are 900x900 (0 for no resizing).
    """
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


def get_image_descriptors(images):
    """
    Returns the image descriptors for the specified images.

    :param images: The images to get descriptors for.
    :return: The image descriptors.
    """
    orb = cv2.ORB_create()
    desc_list = []
    for img in images:
        kp = orb.detect(img, None)
        keypoints, descriptor = orb.compute(img, kp)
        desc_list.append(descriptor)

    return np.array(desc_list)
    