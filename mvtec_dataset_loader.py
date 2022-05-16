import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf


def load_mvtec_dataset(main_dir, object_type, resize_dim=0):
    list_ds = tf.data.Dataset.list_files(os.path.join(main_dir, object_type, '*'), shuffle=False)