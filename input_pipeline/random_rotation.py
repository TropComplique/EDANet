import tensorflow as tf
import math


def random_rotation(image, masks, max_angle=45, probability=0.9):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height, width, num_labels].
        max_angle: an integer.
        probability: a float number.
    Returns:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height, width, num_labels].
    """
    def rotate(image, masks):
        with tf.name_scope('random_rotation'):

            # find the center of the image
            image_height = tf.to_float(tf.shape(image)[0])
            image_width = tf.to_float(tf.shape(image)[1])
            image_center = tf.reshape(0.5*tf.stack([image_height, image_width]), [1, 2])

            # to radians
            max_angle_radians = max_angle*(math.pi/180.0)

            # get a random angle
            theta = tf.random_uniform(
                [], minval=-max_angle_radians,
                maxval=max_angle_radians, dtype=tf.float32
            )

            rotation = tf.stack([
                tf.cos(theta), -tf.sin(theta),
                tf.sin(theta), tf.cos(theta)
            ], axis=0)
            rotation_matrix = tf.reshape(rotation, [2, 2])

            # rotate the image
            translate = image_center - tf.matmul(image_center, rotation_matrix)
            translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
            transform = tf.stack([
                tf.cos(theta), -tf.sin(theta), translate_x,
                tf.sin(theta), tf.cos(theta), translate_y,
                0.0, 0.0
            ])
            image = tf.contrib.image.transform(image, transform, interpolation='BILINEAR')

            # rotate masks
            masks = tf.contrib.image.transform(masks, transform, interpolation='NEAREST')
            # masks are binary so we use the nearest neighbor interpolation

            return image, masks

    do_it = tf.less(tf.random_uniform([]), probability)
    image, masks = tf.cond(
        do_it,
        lambda: rotate(image, masks),
        lambda: (image, masks)
    )
    return image, masks
