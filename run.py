from pie_data import PIE
from jaad_data import JAAD
from car_motion_data import CarMotion
from data_generator import DataGenerator, DataGetter
from tamformer import TAMformer
from motion_labels import CLASS_ID_TO_NAME
from result_logging import (
    default_results_dir,
    format_motion_metrics_line,
    format_per_class_metrics,
    test_log_path,
    write_test_results,
)
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
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


def _softmax_scores(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _motion_class_name(class_id):
    return CLASS_ID_TO_NAME.get(int(class_id), str(int(class_id)))


def _print_sample_inferences(y_true, y_pred, y_scores, sample_count=5):
    if sample_count <= 0:
        return
    total = len(y_true)
    if total == 0:
        print("No test samples available for sample inference preview.")
        return

    sample_count = min(int(sample_count), total)
    print("\nSample inferences ({} of {}):".format(sample_count, total))
    for i in range(sample_count):
        pred_class = int(y_pred[i])
        true_class = int(y_true[i])
        confidence = float(y_scores[i][pred_class])
        print(
            "  [{}] true={} ({}) pred={} ({}) conf={:.4f}".format(
                i,
                true_class,
                _motion_class_name(true_class),
                pred_class,
                _motion_class_name(pred_class),
                confidence,
            )
        )


def _print_per_class_metrics(y_true, y_pred, y_scores, num_classes, class_names=None):
    text = format_per_class_metrics(y_true, y_pred, y_scores, num_classes, class_names)
    print(text)
    return text


def _safe_imread(path):
    try:
        return cv2.imread(str(path))
    except Exception:
        return None


def _draw_label(img_bgr, text):
    if img_bgr is None:
        return None
    img = img_bgr
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    return img


def _mosaic_grid(images_bgr, rows=3, cols=3, tile_size=360):
    total = rows * cols
    tiles = []
    for i in range(total):
        if i < len(images_bgr) and images_bgr[i] is not None:
            tile = cv2.resize(images_bgr[i], (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        else:
            tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        tiles.append(tile)
    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(tiles[r * cols:(r + 1) * cols]))
    return np.vstack(grid_rows)


def _safe_bbox_int(box, img_w, img_h):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[0:4]]
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _resolve_visual_crop_type(obs_input_type):
    for d in obs_input_type:
        if 'local_box' in d:
            return 'bbox'
        if 'local_context' in d:
            return 'context'
        if 'surround' in d:
            return 'surround'
        if 'scene_context' in d:
            return 'none'
    return 'bbox'


def _context_bbox(box, img_w, img_h, enlarge_ratio=1.5):
    x1, y1, x2, y2 = [float(v) for v in box[0:4]]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, (x2 - x1))
    h = max(1.0, (y2 - y1))
    side = max(w, h) * float(enlarge_ratio)
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))
    nx1 = max(0, min(nx1, img_w - 1))
    nx2 = max(0, min(nx2, img_w - 1))
    ny1 = max(0, min(ny1, img_h - 1))
    ny2 = max(0, min(ny2, img_h - 1))
    if nx2 <= nx1:
        nx2 = min(img_w - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(img_h - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _crop_for_visual_sample(img, box, crop_type='bbox', enlarge_ratio=1.5):
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _safe_bbox_int(box, w, h)
    if crop_type == 'none':
        out = img.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return out
    if crop_type == 'bbox':
        return img[y1:y2 + 1, x1:x2 + 1].copy()
    if crop_type == 'context':
        cx1, cy1, cx2, cy2 = _context_bbox([x1, y1, x2, y2], w, h, enlarge_ratio=enlarge_ratio)
        return img[cy1:cy2 + 1, cx1:cx2 + 1].copy()
    if crop_type == 'surround':
        cx1, cy1, cx2, cy2 = _context_bbox([x1, y1, x2, y2], w, h, enlarge_ratio=enlarge_ratio)
        out = img[cy1:cy2 + 1, cx1:cx2 + 1].copy()
        rx1 = max(0, x1 - cx1)
        rx2 = min(out.shape[1] - 1, x2 - cx1)
        ry1 = max(0, y1 - cy1)
        ry2 = min(out.shape[0] - 1, y2 - cy1)
        out[ry1:ry2 + 1, rx1:rx2 + 1, :] = 128
        return out
    return img.copy()


def _render_preview_tile(img, box, crop_type, enlarge_ratio, target_dim):
    cropped = _crop_for_visual_sample(img, box, crop_type=crop_type, enlarge_ratio=enlarge_ratio)
    if cropped is None:
        return None
    if isinstance(target_dim, (list, tuple)) and len(target_dim) >= 2:
        tw, th = int(target_dim[0]), int(target_dim[1])
    else:
        tw, th = int(target_dim), int(target_dim)
    tw = max(1, tw)
    th = max(1, th)
    return cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)


def _sample_diverse_indices(total, count):
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    idxs = np.linspace(0, total - 1, num=count, dtype=int).tolist()
    deduped = []
    seen = set()
    for i in idxs:
        if i not in seen:
            deduped.append(i)
            seen.add(i)
    if len(deduped) < count:
        for i in range(total):
            if i not in seen:
                deduped.append(i)
                seen.add(i)
            if len(deduped) >= count:
                break
    return deduped[:count]


def _extract_record_drive_frame(frame_path):
    norm = str(frame_path).replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    frame_name = os.path.basename(norm)
    record = "UNKNOWN_RECORD"
    drive = "UNKNOWN_DRIVE"
    for p in parts:
        if p.startswith("RECORD"):
            record = p
        if p.startswith("DRIVE"):
            drive = p
    return record, drive, frame_name


def _save_visual_inference_samples(
        data_raw,
        y_true,
        y_pred,
        y_scores,
        out_dir,
        sample_count=3,
        num_frames=9,
        crop_type='bbox',
        enlarge_ratio=1.5,
        target_dim=(224, 224),
        draw_header=False):
    if sample_count <= 0:
        return
    if not data_raw or 'image' not in data_raw or 'bbox' not in data_raw:
        print("Visual samples skipped: raw image/bbox sequences not available.")
        return

    os.makedirs(out_dir, exist_ok=True)
    n = min(len(y_true), len(data_raw['image']), len(data_raw['bbox']))
    if n <= 0:
        print("Visual samples skipped: no test samples available.")
        return

    sample_count = min(int(sample_count), n)
    num_frames = max(1, min(int(num_frames), 9))
    sample_indices = _sample_diverse_indices(n, sample_count)
    print("\nSaving visual inference samples to:", out_dir)
    print("Diverse sample indices:", sample_indices)
    for out_i, i in enumerate(sample_indices):
        seq_imgs = data_raw['image'][i]
        seq_boxes = data_raw['bbox'][i]
        if not seq_imgs:
            continue

        k = min(num_frames, len(seq_imgs))
        frames = list(zip(seq_imgs[-k:], seq_boxes[-k:]))
        rendered = []
        for frame_path, box in frames:
            img = _safe_imread(frame_path)
            if img is None:
                rendered.append(None)
                continue
            rendered.append(_render_preview_tile(
                img, box, crop_type=crop_type, enlarge_ratio=enlarge_ratio, target_dim=target_dim))

        last_frame_path = seq_imgs[-1]
        record, drive, frame_name = _extract_record_drive_frame(last_frame_path)
        conf = float(y_scores[i][int(y_pred[i])])
        pred_token = str(int(y_pred[i]))
        header = "idx={} true={} pred={} conf={:.3f} crop={} {} {} {}".format(
            int(i), int(y_true[i]), pred_token, conf, crop_type, record, drive, frame_name)
        mosaic = _mosaic_grid(rendered, rows=3, cols=3, tile_size=360)
        if draw_header:
            mosaic = _draw_label(mosaic, header)
        out_name = "test_sample_{:03d}_idx{:05d}_{}_{}_{}_t{}_p{}_c{:.3f}.jpg".format(
            int(out_i), int(i), record, drive, os.path.splitext(frame_name)[0],
            int(y_true[i]), pred_token, conf)
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, mosaic)
        print("  saved:", out_path)


def _evaluate_multiclass_motion_test(config_path, configs, model_opts, test_data, test_results,
                                     data_raw_test, weights_path, prefix):
    num_classes = model_opts.get('num_classes', 21)
    y_true = np.asarray(test_data['data'][1]).astype(int)
    y_scores = _softmax_scores(test_results)
    y_pred = np.argmax(y_scores, axis=1)

    if model_opts.get('print_class_distribution', False):
        print_class_distribution(y_true, num_classes, title='Test true class distribution')
        print_class_distribution(y_pred, num_classes, title='Test predicted class distribution')

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        y_true_one_hot = keras.utils.to_categorical(y_true, num_classes=num_classes)
        auc_macro = roc_auc_score(y_true_one_hot, y_scores, multi_class='ovr', average='macro')
    except ValueError:
        auc_macro = 0.0

    motion_line = format_motion_metrics_line(
        'motion', acc, auc_macro, f1_macro, f1_weighted, precision_macro, recall_macro)
    print(motion_line)
    motion_per_class = _print_per_class_metrics(
        y_true, y_pred, y_scores, num_classes, CLASS_ID_TO_NAME)

    labels = list(range(num_classes))
    target_names = [CLASS_ID_TO_NAME.get(i, str(i)) for i in labels]
    sk_report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0, digits=3)
    print("\nSklearn classification report:\n{}".format(sk_report))

    results_dir = default_results_dir(model_opts.get('model_path', './models'))
    test_out = test_log_path(results_dir)
    write_test_results(test_out, [
        "config_file: {}".format(config_path),
        "weights: {}".format(weights_path),
        motion_line,
        motion_per_class,
        sk_report,
    ])
    print("Test results written to:", test_out)

    sample_inference_count = model_opts.get('sample_inference_count', 5)
    _print_sample_inferences(y_true, y_pred, y_scores, sample_inference_count)

    inference_sample_count = model_opts.get('inference_sample_count', 0)
    if inference_sample_count > 0:
        save_inference_samples(
            test_data['data'][0],
            model_opts['obs_input_type'],
            os.path.join(image_output_dir(model_opts, prefix), 'inference_samples'),
            predictions=test_results,
            labels=y_true,
            sample_count=inference_sample_count)

    visual_sample_count = int(model_opts.get('visual_sample_count', 0))
    if visual_sample_count > 0:
        visual_crop_type = model_opts.get('visual_sample_crop_type', 'auto')
        if visual_crop_type == 'auto':
            visual_crop_type = _resolve_visual_crop_type(model_opts.get('obs_input_type', []))
        _save_visual_inference_samples(
            data_raw_test,
            y_true,
            y_pred,
            y_scores,
            out_dir=model_opts.get(
                'visual_sample_out_dir',
                os.path.join(model_opts.get('model_path', './models'), 'visual_samples')),
            sample_count=visual_sample_count,
            num_frames=int(model_opts.get('visual_sample_frames', 9)),
            crop_type=visual_crop_type,
            enlarge_ratio=float(model_opts.get('enlarge_ratio', 1.5)),
            target_dim=model_opts.get('target_dim', (224, 224)),
            draw_header=bool(model_opts.get('visual_sample_draw_header', False)),
        )


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

    if test:
        print('Test-only mode: skipping train/val preprocessing.')
        data_raw_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
        data_getter_test = DataGetter('test', data_raw_test, configs['model_opts'])
        test_data = data_getter_test.get_data()
        data_train = None
        val_data = None
    else:
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
    loaded_weights_path = model_name

    if test or resume:
        weights_to_load = checkpoint_path if checkpoint_path is not None else model_name
        loaded_weights_path = weights_to_load
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
        loaded_weights_path = checkpoint_path if checkpoint_path is not None else model_name
        tamformer.load_weights(loaded_weights_path)

    print("Testing ...")
    test_results = tamformer.predict(test_data['data'][0], verbose=1)
    if configs['model_opts'].get('label_format') == 'multiclass':
        _evaluate_multiclass_motion_test(
            config_path,
            configs,
            configs['model_opts'],
            test_data,
            test_results,
            data_raw_test,
            loaded_weights_path,
            prefix,
        )
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
