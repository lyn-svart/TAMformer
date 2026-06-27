from pie_data import PIE
from jaad_data import JAAD
from car_motion_data import CarMotion
from data_generator import DataGenerator, DataGetter
from tamformer import TAMformer
import os
import sys
import yaml
import numpy as np
import getopt
import pickle
import cv2
import glob
import re
import time
import tensorflow as tf
import random as rn
from argparse import ArgumentParser
import copy
from tensorflow.compat.v1.keras import backend as K


def configure_tensorflow_devices():
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


configure_tensorflow_devices()
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow import keras


def checkpoint_prefix(model_name):
    return os.path.splitext(os.path.basename(model_name))[0]


def get_checkpoint_dir(model_opts):
    return model_opts.get('checkpoint_dir',
                          os.path.join(model_opts['model_path'], 'checkpoints'))


def latest_checkpoint(checkpoint_dir, prefix):
    pattern = os.path.join(checkpoint_dir, prefix + '_epoch_*.h5')
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    epoch_re = re.compile(re.escape(prefix) + r'_epoch_(\d+)\.h5$')
    parsed = []
    for path in checkpoints:
        match = epoch_re.search(os.path.basename(path))
        if match:
            parsed.append((int(match.group(1)), path))
    if not parsed:
        return max(checkpoints, key=os.path.getmtime), 0
    epoch, path = max(parsed, key=lambda item: item[0])
    return path, epoch


def image_output_dir(model_opts, name):
    base_dir = model_opts.get('debug_output_dir',
                              os.path.join(model_opts['model_path'], 'debug_outputs'))
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def to_uint8_image(image):
    image = np.asarray(image)
    if image.ndim != 3:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def save_context_sequence(sequence, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for frame_idx, frame in enumerate(sequence):
        image = to_uint8_image(frame)
        if image is None:
            continue
        cv2.imwrite(os.path.join(output_dir, '{}_frame_{:02d}.jpg'.format(prefix, frame_idx)), image)


def context_input_index(input_types):
    for idx, input_type in enumerate(input_types):
        if 'context' in input_type or 'local' in input_type:
            return idx
    return None


def save_random_generator_inputs(generator, input_types, output_dir, prefix, sample_count=3):
    if generator is None or len(generator) == 0:
        return
    context_idx = context_input_index(input_types)
    if context_idx is None:
        return
    batch_idx = np.random.randint(0, len(generator))
    X, y = generator[batch_idx]
    context_batch = X[context_idx]
    count = min(sample_count, len(context_batch))
    if count == 0:
        return
    sample_indices = np.random.choice(len(context_batch), size=count, replace=False)
    for out_idx, sample_idx in enumerate(sample_indices):
        label = y[sample_idx] if not isinstance(y, list) else y[0][sample_idx]
        sample_prefix = '{}_batch_{:05d}_sample_{}_label_{}'.format(prefix, batch_idx, out_idx, label)
        save_context_sequence(context_batch[sample_idx], output_dir, sample_prefix)


def save_inference_samples(generator, input_types, output_dir, predictions=None, labels=None, sample_count=8):
    if generator is None or len(generator) == 0:
        return
    context_idx = context_input_index(input_types)
    if context_idx is None:
        return
    count = min(sample_count, len(generator))
    sample_indices = np.random.choice(len(generator), size=count, replace=False)
    for sample_idx in sample_indices:
        X, y = generator[sample_idx]
        context_batch = X[context_idx]
        pred_label = None
        if predictions is not None:
            pred = predictions[sample_idx]
            pred_label = int(np.argmax(pred)) if np.ndim(pred) > 0 else int(pred)
        true_label = labels[sample_idx] if labels is not None else (y[0] if np.ndim(y) > 0 else y)
        prefix = 'sample_{:05d}_true_{}'.format(sample_idx, int(true_label))
        if pred_label is not None:
            prefix += '_pred_{}'.format(pred_label)
        save_context_sequence(context_batch[0], output_dir, prefix)


class BatchDebugCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, input_types, output_dir, log_interval=50, sample_count=3):
        super(BatchDebugCallback, self).__init__()
        self.generator = generator
        self.input_types = input_types
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.sample_count = sample_count
        self.current_epoch = 0
        self.epoch_start_time = None
        self.batch_log_dir = os.path.join(output_dir, 'batch_logs')
        self.input_image_dir = os.path.join(output_dir, 'randomized_inputs')
        os.makedirs(self.batch_log_dir, exist_ok=True)
        os.makedirs(self.input_image_dir, exist_ok=True)

    def on_train_begin(self, logs=None):
        save_random_generator_inputs(self.generator, self.input_types, self.input_image_dir,
                                     'train_begin', self.sample_count)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_start_time = time.time()
        save_random_generator_inputs(self.generator, self.input_types, self.input_image_dir,
                                     'epoch_{:03d}'.format(self.current_epoch), self.sample_count)

    def _progress_bar(self, batch_number):
        total_batches = max(len(self.generator), 1)
        width = 30
        filled = int(width * batch_number / total_batches)
        if filled >= width:
            return '=' * width
        return '=' * filled + '>' + '.' * (width - filled - 1)

    def _eta(self, batch_number):
        if not self.epoch_start_time or batch_number <= 0:
            return '?:??:??'
        elapsed = time.time() - self.epoch_start_time
        seconds_per_batch = elapsed / float(batch_number)
        remaining = max(len(self.generator) - batch_number, 0) * seconds_per_batch
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)

    def on_train_batch_end(self, batch, logs=None):
        batch_number = batch + 1
        if batch_number % self.log_interval != 0:
            return
        logs = logs or {}
        total_batches = len(self.generator)
        loss = float(logs.get('loss', 0.0))
        accuracy = logs.get('accuracy', logs.get('sparse_categorical_accuracy', 0.0))
        accuracy = float(accuracy)
        message = '{}/{} [{}] - ETA: {} - loss: {:.4f} - accuracy: {:.4f}'.format(
            batch_number,
            total_batches,
            self._progress_bar(batch_number),
            self._eta(batch_number),
            loss,
            accuracy
        )
        with open(os.path.join(self.batch_log_dir, 'training_batches.txt'), 'a') as f:
            f.write('epoch {} - {}\n'.format(self.current_epoch, message))


def get_multiclass_labels(data_bundle):
    labels = np.asarray(data_bundle['data'][1]).astype(np.int32)
    return labels.reshape(-1)


def print_class_distribution(labels, num_classes, title='Class distribution'):
    counts = np.bincount(labels, minlength=num_classes)
    total = np.sum(counts)
    print("### {} ###".format(title))
    print("Total samples:", int(total))
    for class_idx, count in enumerate(counts):
        ratio = float(count) / float(total) if total else 0.0
        print("class {:02d}: {:7d} ({:.4f})".format(class_idx, int(count), ratio))
    return counts


def effective_number_class_weights(counts, beta=0.9999, clip_min=0.25, clip_max=5.0):
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    present = counts > 0
    if not np.any(present):
        return np.ones_like(counts, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights[present] = weights[present] / np.mean(weights[present])
    weights[present] = np.clip(weights[present], clip_min, clip_max)
    return weights.astype(np.float32)


def get_multiclass_class_weights(labels, model_opts):
    num_classes = model_opts.get('num_classes', int(np.max(labels)) + 1)
    counts = print_class_distribution(labels, num_classes, title='Training class distribution')
    strategy = model_opts.get('class_weight_strategy', 'effective_number')
    if strategy != 'effective_number':
        raise ValueError('Unsupported class_weight_strategy for multiclass: {}'.format(strategy))
    weights = effective_number_class_weights(
        counts,
        beta=model_opts.get('class_weight_beta', 0.9999),
        clip_min=model_opts.get('class_weight_clip_min', 0.25),
        clip_max=model_opts.get('class_weight_clip_max', 5.0))
    print("### Effective-number class weights ###")
    for class_idx, weight in enumerate(weights):
        print("class {:02d}: {:.6f}".format(class_idx, float(weight)))
    return weights


def weighted_sparse_focal_loss(class_weights, gamma=2.0):
    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss_func(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        logits = tf.cast(y_pred, tf.float32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        probs = tf.nn.softmax(logits, axis=-1)
        true_probs = tf.reduce_sum(tf.one_hot(y_true, tf.shape(logits)[-1]) * probs, axis=-1)
        sample_weights = tf.gather(class_weights, y_true)
        focal_factor = tf.pow(1.0 - true_probs, gamma)
        return tf.reduce_mean(sample_weights * focal_factor * ce)

    return loss_func


def run(config_path, auxiliary_loss, test, resume, fresh=False):
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    print(configs['model_opts']['dataset'], '--------------------------------------')
    tte = configs['model_opts']['time_to_event']
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + 2*tte

    if configs['model_opts']['dataset'] == 'jaad':
        imdb = JAAD(data_path= configs['data_opts']['path_to_dataset'])
    elif configs['model_opts']['dataset'] == 'car_motion':
        imdb = CarMotion(data_path=configs['data_opts']['path_to_dataset'])
    else:
        imdb = PIE(data_path= configs['data_opts']['path_to_dataset'])

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
    model_name = configs['model_opts']['model_path']\
                 +'/tamformer_'+configs['model_opts']['dataset']+'_'\
                 +'_'.join(configs['model_opts']['obs_input_type'])+'_'\
                 +str(configs['model_opts']['lr'])+'.h5'
    os.makedirs(configs['model_opts']['model_path'], exist_ok=True)
    ckpt_dir = get_checkpoint_dir(configs['model_opts'])
    os.makedirs(ckpt_dir, exist_ok=True)
    prefix = checkpoint_prefix(model_name)
    checkpoint_path, checkpoint_epoch = latest_checkpoint(ckpt_dir, prefix)
    initial_epoch = 0

    if test or resume:
        weights_to_load = checkpoint_path if checkpoint_path is not None else model_name
        print("Loading "+weights_to_load+" ...")
        partial_loading = configs['model_opts'].get('partial_weight_loading', False) and checkpoint_path is None
        tamformer.load_weights(weights_to_load, by_name=partial_loading, skip_mismatch=partial_loading)
    elif not test and not fresh and checkpoint_path is not None:
        print("Resuming from checkpoint {} ...".format(checkpoint_path))
        tamformer.load_weights(checkpoint_path)
        initial_epoch = checkpoint_epoch
    elif fresh:
        print("Fresh training requested; existing checkpoints will be ignored.")
    if not test:
        optimizer = get_optimizer(configs['model_opts']['optimizer'])(learning_rate=configs['model_opts']['lr'])
        if configs['model_opts'].get('label_format') == 'multiclass':
            classifier_loss = configs['model_opts'].get('classifier_loss', 'sparse_categorical_crossentropy')
            if classifier_loss == 'weighted_focal_loss':
                train_labels = get_multiclass_labels(data_train)
                class_weights = get_multiclass_class_weights(train_labels, configs['model_opts'])
                loss = weighted_sparse_focal_loss(
                    class_weights,
                    gamma=configs['model_opts'].get('focal_gamma', 2.0))
                print("### Multiclass loss: weighted_focal_loss ###")
                print("### Focal gamma: {:.4f} ###".format(configs['model_opts'].get('focal_gamma', 2.0)))
            else:
                if configs['model_opts'].get('print_class_distribution', False):
                    train_labels = get_multiclass_labels(data_train)
                    print_class_distribution(train_labels, configs['model_opts'].get('num_classes', 21),
                                             title='Training class distribution')
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                print("### Multiclass loss: sparse_categorical_crossentropy ###")
            tamformer.compile(loss=loss,
                              optimizer=optimizer,
                              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
        else:
            class_w = class_weights(configs['model_opts']['apply_class_weights'],
                                         data_train['count'],
                                         configs['model_opts']['negative_weight'],
                                         configs['model_opts']['positive_weight'])
            w = [class_w[0], class_w[1]]
            tamformer.compile(loss=weighted_binary_crossentropy(weights=w),
                              optimizer=optimizer,
                              metrics=['accuracy'])

        checkpoint_template = os.path.join(ckpt_dir, prefix + '_epoch_{epoch:03d}.h5')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_template,
                                                                 save_weights_only=True,
                                                                 save_best_only=False)
        debug_callback = BatchDebugCallback(data_train['data'][0],
                                            configs['model_opts']['obs_input_type'],
                                            image_output_dir(configs['model_opts'], prefix),
                                            log_interval=configs['model_opts'].get('batch_log_interval', 50),
                                            sample_count=configs['model_opts'].get('debug_sample_count', 3))
        history = tamformer.fit(x=data_train['data'][0],
                                y=None,
                                batch_size=configs['model_opts']['batch_size'],
                                epochs=configs['model_opts']['epochs'],
                                initial_epoch=initial_epoch,
                                validation_data=val_data['data'][0],
                                verbose=1,
                                callbacks=[checkpoint_callback, debug_callback])
        tamformer.save_weights(model_name)

        tamformer = TAMformer(configs['model_opts'], auxiliary_loss).tamformer()
        checkpoint_path, _ = latest_checkpoint(ckpt_dir, prefix)
        tamformer.load_weights(checkpoint_path if checkpoint_path is not None else model_name)

    print("Testing ...")
    test_results = tamformer.predict(test_data['data'][0], verbose=1)
    if configs['model_opts'].get('label_format') == 'multiclass':
        y_true = test_data['data'][1]
        y_pred = np.argmax(test_results, axis=-1)
        if configs['model_opts'].get('print_class_distribution', False):
            print_class_distribution(np.asarray(y_true).astype(np.int32),
                                     configs['model_opts'].get('num_classes', 21),
                                     title='Test true class distribution')
            print_class_distribution(np.asarray(y_pred).astype(np.int32),
                                     configs['model_opts'].get('num_classes', 21),
                                     title='Test predicted class distribution')
        save_inference_samples(test_data['data'][0],
                               configs['model_opts']['obs_input_type'],
                               os.path.join(image_output_dir(configs['model_opts'], prefix), 'inference_samples'),
                               predictions=test_results,
                               labels=y_true,
                               sample_count=configs['model_opts'].get('inference_sample_count', 8))
        print('acc:', accuracy_score(y_true, y_pred),
              '- macro_f1:', f1_score(y_true, y_pred, average='macro'))
        return
    best_perf_acc = [0 for i in range(40)]
    best_perf_auc = [0 for i in range(40)]
    best_perf_f1 = [0 for i in range(40)]
    AT = np.flip(np.arange(0, 4.1, 0.1))
    for t in np.arange(0.01, 1.0, 0.01):
       test_results_array = np.array([np.where(test_results[i]>=t, 1, 0) for i in range(40)])
       average = 'binary'
       multi_class = 'raise'
       count = 0
       index = int(configs['model_opts']['interval']/configs['model_opts']['step'])
       masking_index = (test_data['data'][2]/configs['model_opts']['step']).astype(int)
       for i in range(len(test_results)):
           rev_index = int((configs['model_opts']['seq_len']-configs['model_opts']['obs_length'])/configs['model_opts']['step'])\
                            + int(configs['model_opts']['obs_length']/configs['model_opts']['step']) - i
           acc = accuracy(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index)
           f1 = score_f1(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, average=average)
           auc = score_auc(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, multi_class=multi_class)
           precision = score_precision(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, average=average)
           recall = score_recall(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, average=average)

           if best_perf_f1[count]+best_perf_auc[count]<=f1+auc:
               best_perf_f1[count] = f1
               best_perf_auc[count] = auc
               best_perf_acc[count] = acc
           count += 1
    count = 0
    for i in range(len(test_results)):
        print(AT[count],':' ,'acc:', best_perf_acc[count], '- auc:', best_perf_auc[count], '- f1:', best_perf_f1[count])
        count += 1

def accuracy(true, pred, index, masking_index):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return accuracy_score(y_true, y_pred)

def score_f1(true, pred, index, masking_index, average):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return f1_score(y_true, y_pred, average=average)

def score_auc(true, pred, index, masking_index, multi_class):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return roc_auc_score(y_true, y_pred, multi_class=multi_class)

def score_precision(true, pred, index, masking_index, average):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return precision_score(y_true, y_pred, average=average)

def score_recall(true, pred, index, masking_index, average):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return recall_score(y_true, y_pred, average=average)


def class_weights(apply_weights, sample_count, w_neg, w_pos):
    if not apply_weights:
        return None

    total = sample_count['neg_count'] + sample_count['pos_count']
    neg_weight = w_neg #sample_count['pos_count']/total
    pos_weight = w_pos #sample_count['neg_count']/total

    print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
    return {0: neg_weight, 1: pos_weight}


def weighted_binary_crossentropy(weights, out_weight=1.0):
    def loss_func(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)
        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(y_pred, y_true)
        loss = K.mean(tf.multiply(ce, weights_v))
        return loss*out_weight
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
    parser.add_argument('--fresh', action='store_true',
                        help='Start training from scratch and ignore existing checkpoints')

    args = parser.parse_args()
    run(args.config_file, args.auxiliary_loss, args.test, args.resume, args.fresh)
