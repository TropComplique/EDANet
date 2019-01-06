import tensorflow as tf
from .random_rotation import random_rotation
from .color_augmentations import random_color_manipulations, random_pixel_value_scale


SHUFFLE_BUFFER_SIZE = 5000
NUM_PARALLEL_CALLS = 12
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
MIN_CROP_SIZE = 0.9
ROTATE = False


class Pipeline:
    def __init__(self, filenames, is_training, params):
        """
        During the evaluation we don't resize images.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            params: a dict.
        """
        self.is_training = is_training
        self.num_labels = params['num_labels']

        if is_training:
            batch_size = params['batch_size']
            height = params['image_height']
            width = params['image_width']
            self.image_size = [height, width]
        else:
            batch_size = 1

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples
        assert self.num_examples > 0

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if is_training:
            dataset = dataset.shuffle(buffer_size=num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if is_training else 1)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_PARALLEL_CALLS)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [image_height, image_width, 3],
                an RGB image with pixel values in the range [0, 1].
            labels: an int tensor with shape [image_height, image_width].
                The values that it can contain are {0, 1, ..., num_labels - 1}.
                It also can contain ignore label: 255.
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'masks': tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to the [0, 1] range

        # get a segmentation labels
        labels = tf.image.decode_png(parsed_features['masks'], channels=1)

        if self.is_training:
            image, labels = self.augmentation(image, labels)

        labels = tf.squeeze(labels, 2)
        labels = tf.to_int32(labels)
        return image, labels

    def augmentation(self, image, labels):

        if ROTATE:
            labels = tf.squeeze(labels, 2)
            binary_masks = tf.one_hot(labels, self.num_labels, dtype=tf.float32)
            image, binary_masks = random_rotation(image, binary_masks, max_angle=30, probability=0.1)
            labels = tf.argmax(binary_masks, axis=2, output_type=tf.int32)

        image, labels = randomly_crop_and_resize(image, labels, self.image_size, probability=0.9)
        image = random_color_manipulations(image, probability=0.1, grayscale_probability=0.05)
        image = random_pixel_value_scale(image, probability=0.1, minval=0.9, maxval=1.1)
        image, labels = random_flip_left_right(image, labels)
        return image, labels


def randomly_crop_and_resize(image, labels, image_size, probability=0.5):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        labels: a float tensor with shape [height, width, 1].
        image_size: a list with two integers [new_height, new_width].
        probability: a float number.
    Returns:
        image: a float tensor with shape [new_height, new_width, 3].
        labels: a float tensor with shape [new_height, new_width, 1].
    """

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    def get_random_window():

        crop_size = tf.random_uniform([], MIN_CROP_SIZE, 1.0)
        crop_size_y = tf.to_int32(MIN_CROP_SIZE * tf.to_float(height))
        crop_size_x = tf.to_int32(MIN_CROP_SIZE * tf.to_float(width))

        y = tf.random_uniform([], 0, height - crop_size_y, dtype=tf.int32)
        x = tf.random_uniform([], 0, width - crop_size_x, dtype=tf.int32)
        crop_window = tf.stack([y, x, crop_size_y, crop_size_x])
        return crop_window

    whole_image_window = tf.stack([0, 0, height, width])
    do_it = tf.less(tf.random_uniform([]), probability)
    window = tf.cond(
        do_it, lambda: get_random_window(),
        lambda: whole_image_window
    )

    image = tf.image.crop_to_bounding_box(image, window[0], window[1], window[2], window[3])
    labels = tf.image.crop_to_bounding_box(labels, window[0], window[1], window[2], window[3])

    image = tf.image.resize_images(image, image_size, method=RESIZE_METHOD)
    labels = tf.image.resize_images(labels, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, labels


def random_flip_left_right(image, labels):

    def flip(image, labels):
        flipped_image = tf.image.flip_left_right(image)
        flipped_labels = tf.image.flip_left_right(labels)
        return flipped_image, flipped_labels

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, labels = tf.cond(
            do_it,
            lambda: flip(image, labels),
            lambda: (image, labels)
        )
        return image, labels
