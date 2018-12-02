import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def eda_module(x, k, rate):

    initial_x = x
    x = slim.conv2d(x, k, (1, 1), scope='conv_1x1')

    x = slim.conv2d(
        x, k, (3, 1), activation_fn=None,
        normalizer_fn=None, biases_initializer=None, scope='first_conv_3x1'
    )
    x = slim.conv2d(x, k, (1, 3), scope='first_conv_1x3')

    x = slim.conv2d(
        x, k, (3, 1), rate=[rate, 1], activation_fn=None,
        normalizer_fn=None, biases_initializer=None, scope='second_conv_3x1'
    )
    x = slim.conv2d(x, k, (1, 3), rate=[1, rate], scope='second_conv_1x3')
    x = slim.dropout(x)

    return tf.concat([x, initial_x], axis=3)


def downsampling_block(x, out_channels):

    in_channels = x.shape[3].value
    num_filters = out_channels - in_channels if in_channels < out_channels else out_channels

    x = slim.conv2d(
        x, num_filters, (3, 3), stride=2, scope='conv_3x3',
        activation_fn=None, normalizer_fn=None, biases_initializer=None
    )

    if in_channels < out_channels:
        y = slim.max_pool2d(x, (2, 2), stride=2, padding='SAME', scope='max_pool')
        x = tf.concat([x, y], axis=3)

    x = batch_norm(x)
    x = tf.nn.relu(x)
    return x


def eda_net(images, num_classes):

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

    shape = tf.shape(x)
    h, w = shape[1], shape[2]

    with tf.variable_scope('EDANet'):

        self.dilation1 = [1,1,1,2,2]
        self.dilation2 = [2,2,4,4,8,8,16,16]

        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        keep_prob=0.5,
            noise_shape=None,
            is_training=True,
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            x = downsampling_block(x, out_channels=15)
            x = downsampling_block(x, out_channels=60)

            with tf.variable_scope('block1'):
                for rate in dilation1:
                    x = eda_module(x, k, rate)


            x = downsampling_block(x, out_channels=130)
            with tf.variable_scope('block2'):
                for rate in dilation2:
                    x = eda_module(x, k, rate)

            logits = slim.conv2d(x, num_classes, (1, 1), stride=2, scope='projection')
            x = tf.image.resize_bilinear(x, [h, w], align_corners=False)
            return x
