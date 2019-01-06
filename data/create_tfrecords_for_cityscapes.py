import os
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import math
import io
from PIL import Image


"""
This script creates training and validation data.

Just run:
python create_tfrecords.py

And don't forget set the right paths below.
"""


IMAGES_DIR = '/home/dan/datasets/cityscapes/leftImg8bit/'
ANNOTATIONS_DIR = '/home/dan/datasets/cityscapes/annotations/gtFine/'
RESULT_PATH = '/home/dan/datasets/cityscapes/edanet/'
CLASS_WEIGHTS = '/home/dan/datasets/cityscapes/edanet/weights.npy'

# because dataset is big we will split it into parts
NUM_TRAIN_SHARDS = 20
NUM_VAL_SHARDS = 1

# images and masks will be resized
WIDTH, HEIGHT = 1024, 512

NUM_LABELS = 19
# not counting ignore label


def to_tf_example(image_path, annotation_path):
    """
    Arguments:
        image_path: a string.
        annotation_path: a string.
    Returns:
        example: an instance of tf.train.Example.
        label_count: a numpy int array with shape [NUM_LABELS].
        area: an integer.
    """

    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_image = f.read()

    image = Image.open(io.BytesIO(encoded_image))
    assert image.format == 'PNG'
    assert image.mode == 'RGB'
    width, height = image.size
    assert width == 2048 and height == 1024

    image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
    encoded_jpeg = to_jpeg_bytes(image)  # convert from png to jpeg

    with tf.gfile.GFile(annotation_path, 'rb') as f:
        encoded_mask = f.read()

    mask = Image.open(io.BytesIO(encoded_mask))
    mask_width, mask_height = mask.size
    assert mask.format == 'PNG'
    assert mask.mode == 'L'
    assert mask_width == width and mask_height == height

    mask = mask.resize((WIDTH, HEIGHT), Image.NEAREST)
    encoded_mask = to_png_bytes(mask)
    
    # this is needed for creating class loss weights
    mask = np.array(mask)
    not_ignore = mask != 255
    area = not_ignore.sum()
    label_count = np.bincount(mask[not_ignore].reshape(-1), minlength=NUM_LABELS)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(encoded_jpeg),
        'masks': _bytes_feature(encoded_mask)
    }))
    return example, label_count, area


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_jpeg_bytes(image):
    b = io.BytesIO()
    image.save(b, format='jpeg')
    return b.getvalue()


def to_png_bytes(image):
    b = io.BytesIO()
    image.save(b, format='png')
    return b.getvalue()


def convert(subset, num_shards):

    result_path = os.path.join(RESULT_PATH, subset)
    shutil.rmtree(result_path, ignore_errors=True)
    os.mkdir(result_path)

    image_folder = os.path.join(IMAGES_DIR, subset)
    image_paths = []
    for path, _, files in os.walk(image_folder):
        for name in files:
            image_paths.append(os.path.join(path, name))

    # randomize image order
    random.shuffle(image_paths)
    num_examples = len(image_paths)
    print('Number of images:', num_examples)

    shard_size = math.ceil(num_examples/num_shards)
    print('Number of images per shard:', shard_size)

    total_label_count = np.zeros(NUM_LABELS)
    total_area = 0.0

    shard_id = 0
    num_examples_written = 0
    for image_path in tqdm(image_paths):

        if num_examples_written == 0:
            shard_path = os.path.join(result_path, 'shard-%04d.tfrecords' % shard_id)
            if not os.path.exists(shard_path):
                writer = tf.python_io.TFRecordWriter(shard_path)

        path = os.path.relpath(image_path, IMAGES_DIR)
        assert path.endswith('_leftImg8bit.png')
        path = path[:-16] + '_gtFine_labelTrainIds.png'
        annotation_path = os.path.join(ANNOTATIONS_DIR, path)

        tf_example, label_count, area = to_tf_example(image_path, annotation_path)
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        total_label_count += label_count
        total_area += area

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != 0:
        shard_id += 1
        writer.close()

    print('Number of shards:', shard_id)
    print('Result is here:', result_path, '\n')

    if subset == 'train':
        probability_of_label = total_label_count/total_area
        k = 1.12
        weights = 1.0/np.log(probability_of_label + k)
        np.save(CLASS_WEIGHTS, weights.astype('float32'))
        print('Class weights:', weights, '\n')


shutil.rmtree(RESULT_PATH, ignore_errors=True)
os.mkdir(RESULT_PATH)

convert('train', NUM_TRAIN_SHARDS)
convert('val', NUM_VAL_SHARDS)
