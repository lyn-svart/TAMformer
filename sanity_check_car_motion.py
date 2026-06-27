import numpy as np
import tensorflow as tf

from tamformer import TAMformer


def print_tensorflow_devices():
    print("TensorFlow version:", tf.__version__)
    print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    physical_gpus = tf.config.list_physical_devices('GPU')
    physical_cpus = tf.config.list_physical_devices('CPU')
    print("Physical GPUs:", physical_gpus if physical_gpus else "None")
    print("Physical CPUs:", physical_cpus if physical_cpus else "None")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print("Could not set memory growth for {}: {}".format(gpu, exc))
    print("Logical GPUs:", tf.config.list_logical_devices('GPU') if physical_gpus else "None")
    print("Logical CPUs:", tf.config.list_logical_devices('CPU'))


def main():
    print_tensorflow_devices()
    opts = {
        'obs_input_type': ['box', 'local_context'],
        'feat_size': [4, 512],
        'pose_enabled': False,
        'seq_len': 10,
        'sequence_length': 10,
        'obs_length': 10,
        'step': 1,
        'num_classes': 21,
        'prediction_mode': 'final',
        'dropout': 0.1,
        'trainable_backbone': True,
        'backbone_weights': None,
        'target_dim': (224, 224),
    }
    model = TAMformer(opts).tamformer()
    bbox = np.random.rand(2, 10, 4).astype('float32')
    local_context = (np.random.rand(2, 10, 224, 224, 3) * 255.0).astype('float32')
    output = model([bbox, local_context], training=False)
    assert tuple(output.shape) == (2, 21), output.shape
    assert any('local_context_vgg16' == layer.name and layer.trainable for layer in model.layers)
    print('Output tensor device:', output.device)
    print('Car motion TAMformer sanity check passed:', output.shape)


if __name__ == '__main__':
    main()
