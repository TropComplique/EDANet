import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.99
BATCH_NORM_EPSILON = 1e-3
DROPOUT_RATE = 0.05


def eda_net(images, is_training, k, num_classes):
    """
    Arguments:
        images: a float tensor with shape [batch_size, height, width, 3],
            a batch of RGB images with pixels values in the range [0, 1].
        is_training: a boolean.
        k: an integer, growth rate.
        num_classes: an integer, number of labels.
    Returns:
        a float tensor with shape [batch_size, height, width, num_classes].
    """

    shape = tf.shape(images)
    height, width = shape[1], shape[2]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('EDANet'):

        dilation1 = [1, 1, 1, 2, 2]
        dilation2 = [2, 2, 4, 4, 8, 8, 16, 16]

        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.variance_scaling_initializer()
        }

        with slim.arg_scope([slim.conv2d], **params):
            keep_prob = 1.0 - DROPOUT_RATE
            with slim.arg_scope([slim.dropout], keep_prob=keep_prob, is_training=is_training):

                x = downsampling_block(
                    x, out_channels=15, scope='downsampling1',
                    normalizer_fn=batch_norm
                )
                x = downsampling_block(
                    x, out_channels=60, scope='downsampling2',
                    normalizer_fn=batch_norm
                )
                for i, rate in enumerate(dilation1, 1):
                    x = eda_module(x, k, rate, scope='block1/unit%d' % i)

                x = downsampling_block(
                    x, out_channels=130, scope='downsampling3',
                    normalizer_fn=batch_norm
                )
                for i, rate in enumerate(dilation2, 1):
                    x = eda_module(x, k, rate, scope='block2/unit%d' % i)

        x = slim.conv2d(x, num_classes, (1, 1), stride=1, activation_fn=None, scope='classification')
        logits = tf.image.resize_bilinear(x, [height, width], align_corners=True)
        return logits


def eda_module(x, k, rate, scope):
    with tf.variable_scope(scope):

        initial_x = x
        x = slim.conv2d(x, k, (1, 1), scope='conv_1x1')

        x = slim.conv2d(
            x, k, (3, 1), activation_fn=None,
            normalizer_fn=None, biases_initializer=None,
            scope='first_conv_3x1'
        )
        x = slim.conv2d(x, k, (1, 3), scope='first_conv_1x3')

        x = slim.conv2d(
            x, k, (3, 1), rate=[rate, 1], activation_fn=None,
            normalizer_fn=None, biases_initializer=None,
            scope='second_conv_3x1'
        )
        x = slim.conv2d(x, k, (1, 3), rate=[1, rate], scope='second_conv_1x3')
        x = slim.dropout(x)

        return tf.concat([x, initial_x], axis=3)


def downsampling_block(x, out_channels, normalizer_fn, scope):
    with tf.variable_scope(scope):

        in_channels = x.shape[3].value
        num_filters = out_channels - in_channels if in_channels < out_channels else out_channels

        x_initial = x
        x = slim.conv2d(
            x, num_filters, (3, 3), stride=2,
            activation_fn=None, normalizer_fn=None,
            biases_initializer=None, scope='conv_3x3'
        )

        if in_channels < out_channels:
            y = slim.max_pool2d(x_initial, (2, 2), stride=2, padding='SAME', scope='max_pool')
            x = tf.concat([x, y], axis=3)

        x = tf.nn.relu(normalizer_fn(x))
        return x
