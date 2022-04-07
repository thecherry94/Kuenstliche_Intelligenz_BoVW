import os

def load_mvtec_dataset(directory, object_type):

    ground_truth = []
    good = []
    bad = []

    # Extract all directories that only contain images with the specified object type
    filtered_paths = []
    for abs_path, sub_dirs, files in os.walk(directory):
        if len(sub_dirs) == 0:
            split_abs_path = abs_path.split('\\')
            if split_abs_path[1] == object_type:
                filtered_paths.append(abs_path)
    
    for filtered_path in filtered_paths:
        split_abs_path = filtered_path.split('\\')
        type = split_abs_path[2]
        spec = split_abs_path[3]

        ground_truth.append((spec, list(map(lambda x: filtered_path + '\\' + x, os.listdir(filtered_path)))))

    # Extract all images from the filtered directories
    return ground_truth