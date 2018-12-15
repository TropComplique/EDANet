import tensorflow as tf
from .random_rotation import random_rotation
from .color_augmentations import random_color_manipulations, random_pixel_value_scale


SHUFFLE_BUFFER_SIZE = 5000
NUM_PARALLEL_CALLS = 12
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR


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
        # not counting background

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
            labels: an int tensor with shape [image_height, image_width],
                it contains values in range [0, num_labels].
                Where `0` is the label of background.
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

        # get a segmentation masks
        masks = tf.decode_raw(parsed_features['masks'], tf.uint8)
        # unpack bits (reverse np.packbits)
        b = tf.constant([128, 64, 32, 16, 8, 4, 2, 1], dtype=tf.uint8)
        masks = tf.reshape(tf.bitwise.bitwise_and(masks[:, None], b), [-1])
        masks = masks[:(image_height * image_width * self.num_labels)]
        masks = tf.cast(masks > 0, tf.uint8)
        masks = tf.to_float(tf.reshape(masks, [image_height, image_width, self.num_labels]))

        if self.is_training:
            image, masks = self.augmentation(image, masks)
            image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]

        # transform into the sparse format
        background = tf.to_float(tf.logical_not(tf.reduce_any(masks > 0.0, axis=2)))
        masks = tf.concat([tf.expand_dims(background, 2), masks], axis=2)
        labels = tf.argmax(masks, axis=2, output_type=tf.int32)
        # it has shape [image_height, image_width]

        features, labels = image, labels
        return features, labels

    def augmentation(self, image, masks):
        image, masks = random_rotation(image, masks, max_angle=30, probability=0.2)
        image, masks = randomly_crop_and_resize(image, masks, self.image_size, probability=0.9)
        image = random_color_manipulations(image, probability=0.5, grayscale_probability=0.05)
        image = random_pixel_value_scale(image, probability=0.2, minval=0.9, maxval=1.1)
        image, masks = random_flip_left_right(image, masks)
        masks.set_shape(self.image_size + [self.num_labels])
        return image, masks


def randomly_crop_and_resize(image, masks, image_size, probability=0.5):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height, width, num_labels].
        image_size: a list with two integers [new_height, new_width].
        probability: a float number.
    Returns:
        image: a float tensor with shape [new_height, new_width, 3].
        masks: a float tensor with shape [new_height, new_width, num_labels].
    """

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    min_dimension = tf.to_float(tf.minimum(height, width))

    def get_random_window():

        crop_size = tf.random_uniform(
            [], tf.to_int32(0.5 * min_dimension),
            min_dimension, dtype=tf.int32
        )
        # min(height, width) > crop_size

        y = tf.random_uniform([], 0, height - crop_size, dtype=tf.int32)
        x = tf.random_uniform([], 0, width - crop_size, dtype=tf.int32)
        crop_window = tf.stack([y, x, crop_size, crop_size])
        return crop_window

    whole_image_window = tf.stack([0, 0, height, width])
    do_it = tf.less(tf.random_uniform([]), probability)
    window = tf.cond(
        do_it, lambda: get_random_window(),
        lambda: whole_image_window
    )

    image = tf.image.crop_to_bounding_box(image, window[0], window[1], window[2], window[3])
    masks = tf.image.crop_to_bounding_box(masks, window[0], window[1], window[2], window[3])

    image = tf.image.resize_images(image, image_size, method=RESIZE_METHOD)
    masks = tf.image.resize_images(masks, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, masks


def random_flip_left_right(image, masks):

    def flip(image, masks):
        flipped_image = tf.image.flip_left_right(image)
        flipped_masks = tf.image.flip_left_right(masks)
        return flipped_image, flipped_masks

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, masks = tf.cond(
            do_it,
            lambda: flip(image, masks),
            lambda: (image, masks)
        )
        return image, masks
