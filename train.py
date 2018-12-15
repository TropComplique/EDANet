import os
import numpy as np
import tensorflow as tf
from model import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


GPU_TO_USE = '0'
NUM_STEPS = 50000

# a numpy float array with shape [num_labels + 1]
CLASS_WEIGHTS = np.load('data/class_weights.npy')
# zeros label means background

params = {
    'model_dir': 'models/run00/',
    'train_dataset': '/mnt/datasets/dan/moda/edanet/train/',
    'val_dataset': '/mnt/datasets/dan/moda/edanet/val/',

    'weight_decay': 1e-4,
    'k': 40,  # growth rate
    'num_labels': 13,  # without counting background
    'class_weights': CLASS_WEIGHTS,

    'num_steps': NUM_STEPS,
    'initial_learning_rate': 1e-3,
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-6,

    'batch_size': 10,
    'image_height': 256,
    'image_width': 256,
}


def get_input_fn(is_training=True):

    dataset_path = params['train_dataset'] if is_training else params['val_dataset']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        pipeline = Pipeline(filenames, is_training, params)
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'], session_config=session_config,
    save_summary_steps=200, save_checkpoints_secs=1800,
    log_step_count_steps=1000
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(
    model_fn, params=params, config=run_config
)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 2, throttle_secs=3600 * 2,
    hooks=[RestoreMovingAverageHook(params['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
