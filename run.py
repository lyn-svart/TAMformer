import os
import sys
import yaml
import numpy as np
import getopt
import pickle
import cv2
import tensorflow as tf
import random as rn
from argparse import ArgumentParser
import copy

try:
    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)
except RuntimeError:
    pass


def _configure_gpu():
    """Prefer GPU when present; avoid grabbing all VRAM at once (memory growth)."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print(
            "No GPU visible to TensorFlow. On Windows, pip `tensorflow` is GPU-capable only up to "
            "2.10.x and needs a matching NVIDIA driver plus CUDA 11.2 / cuDNN 8.1 on the PATH "
            "(or use WSL2/Linux with a current stack). Continuing on CPU."
        )
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU memory growth setting skipped:", e)
    print("TensorFlow sees GPU(s):", [g.name for g in gpus])


_configure_gpu()

from pie_data import PIE
from jaad_data import JAAD
from data_generator import DataGenerator, DataGetter, TrackJSONAdapter
from tamformer import TAMformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow import keras


def run(config_path, auxiliary_loss, test, resume):
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    print(configs['model_opts']['dataset'], '--------------------------------------')
    fps = max(1, int(configs['model_opts'].get('interval', 30) / max(1, configs['data_opts'].get('fstride', 1))))
    obs_seconds = configs['model_opts'].get('obs_seconds', 1)
    configs['model_opts']['obs_length'] = max(1, int(obs_seconds * fps))
    configs['model_opts']['seq_len'] = configs['model_opts']['obs_length']
    configs['model_opts']['fstride'] = configs['data_opts'].get('fstride', 1)
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length']

    dataset_name = configs['model_opts']['dataset']
    if dataset_name == 'custom_json':
        json_path = configs['data_opts']['path_to_json']
        chunk_dt = configs['data_opts'].get('chunk_dt', 10)
        if chunk_dt is not None:
            chunk_dt = int(chunk_dt)
        adapter = TrackJSONAdapter(json_path, chunk_dt=chunk_dt)
        data_raw_train = adapter.load()
        # train-only mode: use the same split as val/test unless user provides separate files.
        data_raw_test = copy.deepcopy(data_raw_train)
        data_raw_val = copy.deepcopy(data_raw_train)
    else:
        if dataset_name == 'jaad':
            imdb = JAAD(data_path=configs['data_opts']['path_to_dataset'])
        else:
            imdb = PIE(data_path=configs['data_opts']['path_to_dataset'])

        data_raw_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
        data_raw_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
        data_raw_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])

    data_getter_train = DataGetter('train', data_raw_train, configs['model_opts'])
    data_getter_test = DataGetter('test', data_raw_test, configs['model_opts'])
    data_getter_val = DataGetter('val', data_raw_val, configs['model_opts'])

    data_train = data_getter_train.get_data()
    test_data = data_getter_test.get_data()
    val_data = data_getter_val.get_data()

    tamformer = TAMformer(configs['model_opts'], auxiliary_loss).tamformer()
    os.makedirs(configs['model_opts']['model_path'], exist_ok=True)
    model_name = configs['model_opts']['model_path']\
                 +'/tamformer_'+configs['model_opts']['dataset']+'_'\
                 +'_'.join(configs['model_opts']['obs_input_type'])+'_'\
                 +str(configs['model_opts']['lr'])+'.h5'

    if test or resume:
        print("Lodaing "+model_name+" ...")
        tamformer.load_weights(model_name, by_name=False, skip_mismatch=False)
    if not test:
        class_w = class_weights(configs['model_opts']['apply_class_weights'],
                                     data_train['count'],
                                     configs['model_opts'])
        optimizer = get_optimizer(configs['model_opts']['optimizer'])(learning_rate=configs['model_opts']['lr'])
        tamformer.compile(loss=weighted_sparse_categorical_crossentropy(weights=class_w),
                          optimizer=optimizer,
                          metrics=['sparse_categorical_accuracy'])

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_name,
                                                                 save_weights_only=True,
                                                                 monitor='val_loss',
                                                                 mode='min',
                                                                 save_best_only=True)
        history = tamformer.fit(x=data_train['data'][0],
                                y=None,
                                batch_size=configs['model_opts']['batch_size'],
                                epochs=configs['model_opts']['epochs'],
                                validation_data=val_data['data'][0],
                                verbose=1,
                                callbacks=[checkpoint_callback])

        tamformer = TAMformer(configs['model_opts'], auxiliary_loss).tamformer()
        tamformer.load_weights(model_name)

    print("Testing ...")
    test_results = tamformer.predict(test_data['data'][0], verbose=1)
    y_true = np.asarray(test_data['data'][1]).astype(int)
    y_pred = np.argmax(test_results, axis=1)
    num_classes = configs['model_opts'].get('num_classes', 5)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        y_true_one_hot = keras.utils.to_categorical(y_true, num_classes=num_classes)
        auc_macro = roc_auc_score(y_true_one_hot, test_results, multi_class='ovr', average='macro')
    except ValueError:
        auc_macro = 0.0

    print('acc:', acc,
          '- auc_macro_ovr:', auc_macro,
          '- f1_macro:', f1_macro,
          '- f1_weighted:', f1_weighted,
          '- precision_macro:', precision_macro,
          '- recall_macro:', recall_macro)

def class_weights(apply_weights, sample_count, model_opts):
    if not apply_weights:
        return None

    class_count = sample_count.get('class_count', {})
    num_classes = model_opts.get('num_classes', 5)
    configured_weights = model_opts.get('class_weights', [1.0] * num_classes)
    if len(configured_weights) != num_classes:
        configured_weights = [1.0] * num_classes

    if not class_count:
        return configured_weights

    total = sum(class_count.values())
    weights = []
    for class_id in range(num_classes):
        count = class_count.get(class_id, 0)
        inv_freq = (total / (num_classes * count)) if count else 1.0
        weights.append(float(configured_weights[class_id]) * float(inv_freq))
    print("### Class weights:", weights, "###")
    return weights


def weighted_sparse_categorical_crossentropy(weights=None, out_weight=1.0):
    def loss_func(y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred)
        if weights is None:
            return tf.reduce_mean(ce) * out_weight
        class_weights_tensor = tf.constant(weights, dtype=y_pred.dtype)
        sample_weights = tf.gather(class_weights_tensor, y_true_int)
        return tf.reduce_mean(ce * sample_weights) * out_weight
    return loss_func


def get_optimizer(optimizer):
    assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
    "{} optimizer is not implemented".format(optimizer)
    if optimizer.lower() == 'adam':
        return Adam
    elif optimizer.lower() == 'sgd':
        return SGD
    elif optimizer.lower() == 'rmsprop':
        return RMSprop



if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for TAMformer")
    parser.add_argument('--config_file', type=str, help="Path to the directory to load the config file")
    parser.add_argument('--auxiliary_loss', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    run(args.config_file, args.auxiliary_loss, args.test, args.resume)
