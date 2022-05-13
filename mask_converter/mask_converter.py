import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
'''
####################################################################
Mask Converter:
This file extracts the contours from a binary mask image (B/W) and
converts them to a polygonal mask in the style of the VGG Image 
Annotator. This data structure is used in the Mask RCNN model's
Balloons example.

Main differences: I've expanded the file name to ensure that it
is unique since the MVTEC file names are not unique. I've added an
entry for the image path for easier reference later. I've also added
a bounding box for the mask, see below.
####################################################################

# Load annotations
# VGG Image Annotator (up to version 1.6) saves each image in the form:
# { 'filename': '28503151_5b5b7ec140_b.jpg',
#   'regions': {
#       '0': {
#           'region_attributes': {},
#           'shape_attributes': {
#               'all_points_x': [...],
#               'all_points_y': [...],
#               'name': 'polygon'}},
#       ... more regions ...
#   },
#   'size': 100202
# }# We mostly care about the x and y coordinates of each region
# # Note: In VIA 2.0, regions was changed from a dict to a list.

####################################################################
Example from Balloon dataset JSON file:

I've added a bounding box to the structure under the regions_attributes key

                          corner_up_right
                ----------#
                |         |
                |         |
                #----------
    corner_low_left
    
####################################################################
    "24631331976_defa3bb61f_k.jpg668058":
            {"fileref": "",
            "size": 668058,
            "filename": "24631331976_defa3bb61f_k.jpg",
            "base64_img_data": "",
             "file_attributes": {},
             "regions":
                 {
                     "0":
                         {
                            "shape_attributes":
                                {
                                    "name": "polygon",
                                         "all_points_x": [916, 913, 905, 889, 868, 836, 809, 792, 789, 784, 777, 769, 767, 777,
                                                          786, 791, 769, 739, 714, 678, 645, 615, 595, 583, 580, 584, 595, 614,
                                                          645, 676, 716, 769, 815, 849, 875, 900, 916, 916],
                                         "all_points_y": [515, 583, 616, 656, 696, 737, 753, 767, 777, 785, 785, 778, 768, 766,
                                                          760, 755, 755, 743, 728, 702, 670, 629, 588, 539, 500, 458, 425, 394,
                                                          360, 342, 329, 331, 347, 371, 398, 442, 504, 515]},
                    "region_attributes": {}
                         }
                 }
             },
    '''
'''
####################################################################
Key Variables:
I moved all ground_truth images into a folder called ground_truth in
the root directory, but I maintained the original folder structure
for the classes and failure types. Pathlib reads this recursively
with the rglob() method.
####################################################################
'''
annotation_dir = './ground_truth/'
debug = False
confirmation_testing = False

'''
####################################################################
Mask List:
Gathers all masks from the annotation directory, stores the path
and image data in a list.
####################################################################
'''

mask_list = []
print("gathering masks")
for path in Path(annotation_dir).rglob("*.png"):
    image = cv2.imread(str(path))
    # image = cv2.resize(image, (size, size))
    mask_list.append((path, image))
print("gathering masks complete")

'''
####################################################################
Contour extraction:
finds the contours of the image in a single channel image, extracts
the points and splits it into lists of x and y coordinates.
####################################################################
'''
labels_info = {}
print("begin extracting contours")
i = 0
for path, mask in mask_list:
    if debug == True:
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        if i == 9:
            debug = False
            i = i + 1
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # convert to grayscale, doesn't function without this step
    # opencv 3.2
    contours, hierarchy = cv2.findContours(
        (mask).astype(np.uint8),  # convert to uint8 for opencv, findContours needs uint8
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
    if len(segmentation) == 0:
        continue

    # Split segmentation into x and y coordinates
    x = segmentation[0][::2]
    y = segmentation[0][1::2]

    if debug == True:
        print(segmentation)
        print("x:", x)
        print("y:", y)

    '''
    ####################################################################
    Append Dictionary:
    Build the dictionary entry for each mask image. The dictionary has
    the form of a JSON object. We give this out later directly to a 
    text file/JSON file.
    ####################################################################
    '''
    name = str(path.parts[1]) + "_" + str(path.parts[3]) + "_" + str(path.parts[4])
    labels_info[name] = {
                    "fileref": "",
                    "size": os.path.getsize(str(path)),
                    "relative_path": str(path),
                    "filename": str(path.parts[-1]),
                    "base64_img_data": "",
                    "file_attributes": {},
                    "regions":
                        {
                            "0":
                                {
                                    "shape_attributes":
                                        {
                                            "name": "polygon",
                                            "all_points_x": x,
                                            "all_points_y": y
                                        },
                                    "region_attributes": {
                                        "Name": "bbox",
                                        "corner_low_left": [min(x), min(y)],
                                        "corner_up_right": [max(x), max(y)],
                                    }
                                }
                        }
                }
output = [labels_info]
'''
####################################################################
Sanity Check:
Prints the first 10 masks, the points extracted for the contours, 
and the bounding boxes.
confirmation_testing flag must be set to True before run
####################################################################
'''
j = 0
if confirmation_testing == True:
    fig, ax = plt.subplots()

    ax.imshow(mask, cmap='gray')

    for i in range(len(x)):
        ax.plot(x[i], y[i], 'ro')
    # ax.imshow(Image2_mask, cmap='jet', alpha=0.5)  # interpolation='none'

    corner = (min(segmentation[0][::2]), min(segmentation[0][1::2]))
    width = max(segmentation[0][::2]) - min(segmentation[0][::2])
    height = max(segmentation[0][1::2]) - min(segmentation[0][1::2])
    rect = patches.Rectangle(corner, width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()
    if j == 9:
        confirmation_testing = False
        j = j + 1
if debug:
    print(labels_info)

'''
####################################################################
Write JSON:
Writes the dictionary to a text file in JSON format.
Located in the root directory of the project.
####################################################################
'''

for entry in output:
    with open('mvtec_annotations.json', 'w') as f:
        json.dump(entry, f)

print("done")
