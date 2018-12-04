import tensorflow as tf


def random_rotation(image, masks, max_angle=45, probability=0.9):
    """
    This function takes a random box and rotates everything around its center.
    Then it translates the image's center to be at the box's center.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [mask_height, mask_width, 2],
            they are smaller than the image in DOWNSAMPLE times.
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3].
        max_angle: an integer.
        probability: a float number.
    Returns:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [mask_height, mask_width, 2].
        boxes: a float tensor with shape [num_remaining_boxes, 4],
            where num_remaining_boxes <= num_persons.
        keypoints: an int tensor with shape [num_persons, 17, 3],
            note that some keypoints might be out of the image, but
            we will correct that after doing a random crop.
    """
    def rotate(image, masks, boxes, keypoints):
        with tf.name_scope('random_rotation'):

            # find the center of the image
            image_height = tf.to_float(tf.shape(image)[0])
            image_width = tf.to_float(tf.shape(image)[1])
            image_center = tf.reshape(0.5*tf.stack([image_height, image_width]), [1, 2])

            # this changes the center of the image
            center_translation = box_center - image_center

            # to radians
            max_angle_radians = max_angle*(math.pi/180.0)

            # get a random angle
            theta = tf.random_uniform(
                [], minval=-max_angle_radians,
                maxval=max_angle_radians, dtype=tf.float32
            )


            # `tf.contrib.image.transform` needs inverse transform
            inverse_scale = 1.0 / scale
            inverse_rotation = tf.stack([
                tf.cos(theta), -tf.sin(theta),
                tf.sin(theta), tf.cos(theta)
            ], axis=0)
            inverse_rotation_matrix = inverse_scale * tf.reshape(inverse_rotation, [2, 2])

            # rotate the image
            translate = box_center - tf.matmul(box_center - center_translation, inverse_rotation_matrix)
            translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
            transform = tf.stack([
                inverse_scale * tf.cos(theta), -inverse_scale * tf.sin(theta), translate_x,
                inverse_scale * tf.sin(theta), inverse_scale * tf.cos(theta), translate_y,
                0.0, 0.0
            ])
            image = tf.contrib.image.transform(image, transform, interpolation='BILINEAR')

            # masks are smaller than the image
            scaler = tf.to_float(tf.stack([1.0/DOWNSAMPLE, 1.0/DOWNSAMPLE]))
            box_center *= scaler
            center_translation *= scaler

            # rotate masks
            translate = box_center - tf.matmul(box_center - center_translation, inverse_rotation_matrix)
            translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
            transform = tf.stack([
                inverse_scale * tf.cos(theta), -inverse_scale * tf.sin(theta), translate_x,
                inverse_scale * tf.sin(theta), inverse_scale * tf.cos(theta), translate_y,
                0.0, 0.0
            ])
            masks = tf.contrib.image.transform(masks, transform, interpolation='NEAREST')
            # masks are binary so we use the nearest neighbor interpolation

            return image, masks

    do_it = tf.less(tf.random_uniform([]), probability)
    image, masks, boxes, keypoints = tf.cond(
        do_it,
        lambda: rotate(image, masks),
        lambda: (image, masks)
    )
    return image, masks
