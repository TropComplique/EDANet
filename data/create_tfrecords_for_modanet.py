import os
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
import random
import math
import pandas as pd
import io
from PIL import Image

# you can get this from here:
# https://github.com/cocodataset/cocoapi
import sys
sys.path.append('/home/dan/work/cocoapi/PythonAPI/')
from pycocotools.coco import COCO


"""
This script creates training and validation data.

Just run:
python create_tfrecords.py

And don't forget set the right paths below.
"""


IMAGES_DIR = '/mnt/datasets/dan/moda/images/train/'
ANNOTATIONS_FILE = '/mnt/datasets/dan/moda/annotations/modanet2018_instances_train.json'
coco = COCO(ANNOTATIONS_FILE)

# it tells which image ids will be in validation
SPLIT = pd.read_csv('trainval_split.csv')

# path where converted data will be stored
RESULT_PATH = '/mnt/datasets/dan/moda/edanet2/'

# because dataset is big we will split it into parts
NUM_TRAIN_SHARDS = 300
NUM_VAL_SHARDS = 1

# classes will be encoded by integers as usual
with open('modanet_labels.txt', 'r') as f:
    integer_encoding = {line.strip(): i for i, line in enumerate(f.readlines()) if line.strip()}
print('Possible labels (and label encoding):', integer_encoding)
NUM_LABELS = len(integer_encoding)  # without background

categories = coco.loadCats(coco.getCatIds())
id_to_integer = {d['id']: integer_encoding[d['name']] for d in categories}


def to_tf_example(image_path, annotations):
    """
    Arguments:
        image_path: a string.
        annotations: a list of dicts.
    Returns:
        an instance of tf.train.Example.
    """

    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    image = Image.open(io.BytesIO(encoded_jpg))
    if not image.format == 'JPEG':
        return None

    if image.mode == 'L':  # if grayscale
        rgb_image = np.stack(3*[np.array(image)], axis=2)
        encoded_jpg = to_jpeg_bytes(rgb_image)
        image = Image.open(io.BytesIO(encoded_jpg))
    assert image.mode == 'RGB'
    width, height = image.size
    assert width > 0 and height > 0

    masks = np.zeros((height, width, NUM_LABELS), dtype='bool')
    for a in annotations:
        binary_mask = coco.annToMask(a) > 0
        i = id_to_integer[a['category_id']]
        masks[:, :, i] = np.logical_or(masks[:, :, i], binary_mask)

    # now transform one hot encoded masks into the sparse format
    background = np.logical_not(np.any(masks, axis=2))
    masks = np.concatenate([np.expand_dims(background, 2), masks], axis=2)
    masks = np.argmax(masks.astype('int32'), axis=2).astype('uint8')

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(encoded_jpg),
        'masks': _bytes_feature(masks.tostring())
    }))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_jpeg_bytes(array):
    image = Image.fromarray(array)
    tmp = io.BytesIO()
    image.save(tmp, format='jpeg')
    return tmp.getvalue()


def convert(image_ids, result_path, num_shards):

    shutil.rmtree(result_path, ignore_errors=True)
    os.mkdir(result_path)

    # randomize image order
    random.shuffle(image_ids)
    num_examples = len(image_ids)
    print('Number of images:', num_examples)

    shard_size = math.ceil(num_examples/num_shards)
    print('Number of images per shard:', shard_size)

    shard_id = 0
    num_examples_written = 0
    num_skipped_images = 0
    for example in tqdm(image_ids):

        if num_examples_written == 0:
            shard_path = os.path.join(result_path, 'shard-%04d.tfrecords' % shard_id)
            if not os.path.exists(shard_path):
                writer = tf.python_io.TFRecordWriter(shard_path)

        image_metadata = coco.loadImgs(example)[0]
        image_path = os.path.join(IMAGES_DIR, str(image_metadata['id']) + '.jpg')
        annIds = coco.getAnnIds(imgIds=image_metadata['id'])
        annotations = coco.loadAnns(annIds)

        tf_example = to_tf_example(image_path, annotations)
        if tf_example is None:
            num_skipped_images += 1
            continue
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != 0:
        writer.close()

    print('Number of skipped images:', num_skipped_images)
    print('Number of shards:', shard_id + 1)
    print('Result is here:', result_path, '\n')


shutil.rmtree(RESULT_PATH, ignore_errors=True)
os.mkdir(RESULT_PATH)

image_ids = list(SPLIT.loc[SPLIT.is_train, 'image_id'])
result_path = os.path.join(RESULT_PATH, 'train')
convert(image_ids, result_path, NUM_TRAIN_SHARDS)

image_ids = list(SPLIT.loc[~SPLIT.is_train, 'image_id'])
result_path = os.path.join(RESULT_PATH, 'val')
convert(image_ids, result_path, NUM_VAL_SHARDS)
