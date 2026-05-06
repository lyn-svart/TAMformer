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


def _auto_configure_feat_size(model_opts):
    """Auto-derive feat_size from obs_input_type and backbone when enabled."""
    if not model_opts.get('auto_feat_size', False):
        return

    obs_input_types = model_opts.get('obs_input_type', [])
    backbone = model_opts.get('backbone', 'vgg16')
    feat_size = []

    for d_type in obs_input_types:
        if d_type == 'box':
            feat_size.append(4)
        elif d_type == 'speed':
            feat_size.append(1)
        elif ('local' in d_type) or ('context' in d_type) or ('surround' in d_type):
            feat_size.append(DataGetter.spatial_backbone_vector_dim(backbone, 'max'))
        else:
            # Keep compatibility for unknown/custom modalities.
            configured = model_opts.get('feat_size', [])
            idx = len(feat_size)
            if isinstance(configured, list) and idx < len(configured):
                feat_size.append(configured[idx])
            else:
                raise ValueError(
                    "auto_feat_size cannot infer dim for obs_input_type '{}'".format(d_type)
                )

    model_opts['feat_size'] = feat_size
    print("Auto feat_size:", feat_size, "(backbone={})".format(backbone))


def _print_sample_inferences(y_true, y_pred, y_scores, sample_count=5):
    """Print a few sample inferences from the test split."""
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
            "  [{}] true={} pred={} conf={:.4f}".format(
                i, true_class, pred_class, confidence
            )
        )


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
            tile = images_bgr[i]
            tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
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
        out = img[cy1:cy2 + 1, cx1:cx2 + 1].copy()
        return out
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


def _sample_diverse_indices(total, count):
    """Pick evenly spread indices across the full test set."""
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    # Evenly distribute picks from start to end (deterministic, diverse).
    idxs = np.linspace(0, total - 1, num=count, dtype=int).tolist()
    # Guard against rare duplicate indices from integer rounding.
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
        enlarge_ratio=1.5):
    """
    Save visual grids for a few test samples.

    Expects data_raw to contain:
      - image: list of sequences, each a list of frame paths
      - bbox:  list of sequences, each a list of [x1,y1,x2,y2] (pixel coords)
    """
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
            img = _crop_for_visual_sample(img, box, crop_type=crop_type, enlarge_ratio=enlarge_ratio)
            record, drive, frame_name = _extract_record_drive_frame(frame_path)
            tile_label = "{}/{}/{}".format(record, drive, frame_name)
            cv2.putText(img, tile_label, (8, max(20, img.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            rendered.append(img)

        last_frame_path = seq_imgs[-1]
        record, drive, frame_name = _extract_record_drive_frame(last_frame_path)
        if (y_pred is None) or (y_scores is None):
            conf = None
            pred_token = "NA"
            header = "idx={} true={} pred={} conf={} crop={} {} {} {}".format(
                int(i), int(y_true[i]), pred_token, "NA", crop_type, record, drive, frame_name
            )
        else:
            conf = float(y_scores[i][int(y_pred[i])])
            pred_token = str(int(y_pred[i]))
            header = "idx={} true={} pred={} conf={:.3f} crop={} {} {} {}".format(
                int(i), int(y_true[i]), int(y_pred[i]), conf, crop_type, record, drive, frame_name
            )
        mosaic = _mosaic_grid(rendered, rows=3, cols=3, tile_size=360)
        mosaic = _draw_label(mosaic, header)

        if conf is None:
            out_name = "test_sample_{:03d}_idx{:05d}_{}_{}_{}_t{}_p{}.jpg".format(
                int(out_i), int(i), record, drive, os.path.splitext(frame_name)[0], int(y_true[i]), pred_token
            )
        else:
            out_name = "test_sample_{:03d}_idx{:05d}_{}_{}_{}_t{}_p{}_c{:.3f}.jpg".format(
                int(out_i), int(i), record, drive, os.path.splitext(frame_name)[0], int(y_true[i]), pred_token, conf
            )
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, mosaic)
        print("  saved:", out_path)
        print("    metadata: idx={} crop={} record={} drive={} frame={}".format(
            int(i), crop_type, record, drive, frame_name
        ))


def run(config_path, auxiliary_loss, test, resume):
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    print(configs['model_opts']['dataset'], '--------------------------------------')
    fps = max(1, int(configs['model_opts'].get('interval', 30) / max(1, configs['data_opts'].get('fstride', 1))))
    obs_seconds = configs['model_opts'].get('obs_seconds', 1)
    configs['model_opts']['obs_length'] = max(1, int(obs_seconds * fps))
    configs['model_opts']['seq_len'] = configs['model_opts']['obs_length']
    configs['model_opts']['fstride'] = configs['data_opts'].get('fstride', 1)
    _auto_configure_feat_size(configs['model_opts'])
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length']

    dataset_name = configs['model_opts']['dataset']
    if dataset_name == 'custom_json':
        chunk_dt = configs['data_opts'].get('chunk_dt', 10)
        frames_root = configs['data_opts'].get('path_to_frames_root')
        if chunk_dt is not None:
            chunk_dt = int(chunk_dt)
        train_json = configs['data_opts'].get('path_to_json_train')
        val_json = configs['data_opts'].get('path_to_json_val')
        test_json = configs['data_opts'].get('path_to_json_test')
        if train_json and val_json and test_json:
            print("Using explicit custom_json splits (train/val/test).")
            data_raw_train = TrackJSONAdapter(train_json, chunk_dt=chunk_dt, frames_root=frames_root).load()
            data_raw_val = TrackJSONAdapter(val_json, chunk_dt=chunk_dt, frames_root=frames_root).load()
            data_raw_test = TrackJSONAdapter(test_json, chunk_dt=chunk_dt, frames_root=frames_root).load()
        else:
            json_path = configs['data_opts']['path_to_json']
            adapter = TrackJSONAdapter(json_path, chunk_dt=chunk_dt, frames_root=frames_root)
            data_raw_train = adapter.load()
            # Backward-compatible single-file mode (not suitable for final evaluation).
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
    weights_stem = (
        configs['model_opts']['model_path']
        + '/tamformer_'
        + configs['model_opts']['dataset']
        + '_'
        + '_'.join(configs['model_opts']['obs_input_type'])
        + '_'
        + str(configs['model_opts']['lr'])
    )
    # Keras 3: save_weights_only requires filepath to end with `.weights.h5`
    model_name = weights_stem + '.weights.h5'
    legacy_weights_h5 = weights_stem + '.h5'

    def _weights_file_to_load():
        if os.path.isfile(model_name):
            return model_name
        if os.path.isfile(legacy_weights_h5):
            return legacy_weights_h5
        return model_name

    if test or resume:
        load_path = _weights_file_to_load()
        print("Lodaing " + load_path + " ...")
        # Keras 3 no longer accepts `by_name` for the new `.weights.h5` format.
        # Keep a legacy path for old `.h5` checkpoints.
        if load_path.endswith(".weights.h5"):
            tamformer.load_weights(load_path)
        else:
            tamformer.load_weights(load_path, by_name=False, skip_mismatch=False)
    if not test:
        if bool(configs['model_opts'].get('visual_sample_before_training', False)):
            preview_count = int(configs['model_opts'].get('visual_sample_count', 0))
            if preview_count > 0:
                preview_frames = int(configs['model_opts'].get('visual_sample_frames', 9))
                preview_out_dir = configs['model_opts'].get(
                    'visual_sample_out_dir',
                    os.path.join(configs['model_opts'].get('model_path', './models'), 'visual_samples'),
                )
                preview_crop_type = configs['model_opts'].get('visual_sample_crop_type', 'auto')
                if preview_crop_type == 'auto':
                    preview_crop_type = _resolve_visual_crop_type(configs['model_opts'].get('obs_input_type', []))
                preview_enlarge_ratio = float(configs['model_opts'].get('enlarge_ratio', 1.5))
                print("\nCreating visual input preview before training...")
                _save_visual_inference_samples(
                    data_raw_test,
                    np.asarray(test_data['data'][1]).astype(int),
                    y_pred=None,
                    y_scores=None,
                    out_dir=preview_out_dir,
                    sample_count=preview_count,
                    num_frames=preview_frames,
                    crop_type=preview_crop_type,
                    enlarge_ratio=preview_enlarge_ratio,
                )
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
    sample_inference_count = configs['model_opts'].get('sample_inference_count', 5)
    _print_sample_inferences(y_true, y_pred, test_results, sample_inference_count)

    visual_sample_count = int(configs['model_opts'].get('visual_sample_count', 0))
    if visual_sample_count > 0:
        visual_frames = int(configs['model_opts'].get('visual_sample_frames', 9))
        visual_out_dir = configs['model_opts'].get(
            'visual_sample_out_dir',
            os.path.join(configs['model_opts'].get('model_path', './models'), 'visual_samples'),
        )
        visual_crop_type = configs['model_opts'].get('visual_sample_crop_type', 'auto')
        if visual_crop_type == 'auto':
            visual_crop_type = _resolve_visual_crop_type(configs['model_opts'].get('obs_input_type', []))
        visual_enlarge_ratio = float(configs['model_opts'].get('enlarge_ratio', 1.5))
        _save_visual_inference_samples(
            data_raw_test,
            y_true,
            y_pred,
            test_results,
            out_dir=visual_out_dir,
            sample_count=visual_sample_count,
            num_frames=visual_frames,
            crop_type=visual_crop_type,
            enlarge_ratio=visual_enlarge_ratio,
        )

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
