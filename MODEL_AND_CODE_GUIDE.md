# TAMformer Model and Code Guide

This document explains how the model works, how data flows through the codebase, and how the **custom `trackID`-based JSON** pipeline is wired (including **images**, optional **motion + location** dual-head training, and **Keras 3**).

## 1) Big picture

TAMformer is a **sequence classification** model.

- **Input:** a short observation window (`obs_length`) of **per-frame features** per modality (e.g. box, speed, optional CNN visual features).
- **Output:** either a **single** softmax head over **`num_classes`** motion labels (config-driven; **21** classes from JSON `motion` strings is typical), or—when **`predict_location: true`**—**two** heads: **`motion`** (same `num_classes`) and **`location`** (`num_location_classes`, usually **3**: left / center / right from JSON `location`).
- **Core idea:** temporal transformers per modality with **learned attention masks**, then **cross-attention** using the **last timestep** as query, then a small MLP + **softmax** (one or two output layers sharing that trunk).

Each training sample is one **object track** (grouped by `trackID`), not a single frame.

---

## 2) Main files and responsibilities

| File | Role |
|------|------|
| `run.py` | Loads YAML, builds datasets, compiles model, trains, checkpoints, evaluates. TensorFlow import and GPU/thread config run **before** heavy imports. |
| `data_generator.py` | `TrackJSONAdapter`, `DataGetter`, `DataGenerator` (`Sequence`): JSON → track sequences → batches. Visual pipeline (VGG/ResNet/MobileNet crops + features). |
| `tamformer.py` | Functional Keras model: masking nets, `TransformerBlock`, classifier. Written for **Keras 3** (`keras.ops`, no `tf.*` on symbolic tensors in the graph). |
| `configs/configs_custom_json.yaml` | Example config for `dataset: custom_json` (single motion head; modalities, splits, frames root, optional no-disk visual mode). |
| `configs/configs_custom_json_motion_location.yaml` | Same stack with **`predict_location: true`**: dual heads; separate **`model_path`** so checkpoints do not overwrite single-head runs. |
| `scripts/split_custom_json_tracks.py` | Splits one frame-keyed JSON into **track-disjoint** `train.json` / `val.json` / `test.json`. |
| `scripts/visualize_custom_json_inputs.py` | Draws frames + bbox + motion for sanity checks (optional `RECORD`/`DRIVE` overlay). |
| `scripts/show_track_sequence_example.py` | Prints raw vs processed tensor shapes for one track. |

---

## 3) Custom JSON pipeline (`TrackJSONAdapter`)

### 3.1 JSON layout

Frame-keyed dict:

- **Key:** frame path string, e.g. `RECORD1/DRIVE1/frames/000123.png` (your repo often uses this `RECORDX/DRIVEX/frames` scheme).
- **Value:** object with `objs` list; each object typically has:
  - `trackID`
  - `xywh` (normalized in image space)
  - `motion` (string label)
  - `location` (optional string: **`left`**, **`center`**, **`right`**; case-insensitive). Used only when **`predict_location`** is enabled in config; missing/unknown values default to **center** (class id **1**).
  - `img_width`, `img_height`
  - `Vx`, `Vz` (used for speed magnitude)

### 3.2 Grouping and sorting

1. Scan all frames and objects.
2. Group by `trackID`.
3. Sort each track by frame index parsed from the path.

### 3.3 Features and labels

- **BBox:** `xywh` → absolute `[x1, y1, x2, y2]` using stored image dimensions.
- **Center:** derived from bbox.
- **Speed:** `sqrt(Vx² + Vz²)` with safe numeric handling.
- **Class (motion):** `motion` string → integer via `TrackJSONAdapter.MOTION_TO_CLASS` (extended set; unknown strings map to **`stopped`**-class id **3** by default).
- **Class (location):** when dual-head training is enabled, `location` string → integer via **`TrackJSONAdapter.LOCATION_TO_CLASS`** (`left`→0, `center`→1, `right`→2). Parallel per-frame lists are stored in **`location_activities`** (same length and chunking as **`activities`**).

Final sequence labels for both tasks use the **last timestep** in `DataGetter.get_data_sequence()` (`labels` = motion, `labels_location` = location when `predict_location` is on).

Adapter output keys used downstream: `image`, `pid`, `bbox`, `center`, `obd_speed`, `activities`, **`location_activities`** (present for `TrackJSONAdapter`; JAAD/PIE do not supply it—see §5), `image_dimension`.

### 3.4 Resolving frame paths on disk

JSON keys may not match your machine’s absolute paths. Set in config:

```yaml
data_opts:
  path_to_frames_root: /absolute/path/to/PreventionData   # parent of RECORD*/DRIVE*/frames
```

`TrackJSONAdapter` accepts `frames_root=` and resolves each frame to an existing file when possible (including `RECORD…/DRIVE…/frames/…` suffix join).

---

## 4) Train / val / test splits (`custom_json`)

In `run.py`, if **all three** are set, data is loaded from separate files (no leakage):

```yaml
data_opts:
  path_to_json_train: .../train.json
  path_to_json_val:   .../val.json
  path_to_json_test:  .../test.json
```

Otherwise `path_to_json` is used and val/test are copies of train (**not** for final benchmarking).

Generate splits:

```bash
python scripts/split_custom_json_tracks.py \
  --input_json /path/to/full.json \
  --output_dir /path/to/splits \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --seed 42
```

---

## 5) Sequence preparation (`DataGetter`)

`get_data_sequence()`:

1. Reads `bbox`, `pid`, `activities`, optional `speed`, etc.
2. If **`model_opts.predict_location`**: attaches **`location`** sequences from **`location_activities`** when present; otherwise fills **center** for every timestep and prints a one-time warning (JAAD/PIE).
3. **`obs_length`** = `obs_seconds * (interval / fstride)` (clamped to at least 1).
4. Pads short tracks from the **start** (repeat first frame), then keeps the **last** `obs_length` steps.
5. Builds **`labels`** from the last timestep’s motion class; if **`predict_location`**, also **`labels_location`** and **`class_count_location`** inside the returned **`count`** bundle.

When **`balance_data`** mirrors samples with horizontal flips, **`location`** labels are **left↔right swapped** (0↔2) so they stay consistent with flipped boxes.

`get_data()` loops `model_opts['obs_input_type']`:

- `box`, `speed`: numpy tensors from arrays.
- Names containing **`local`** or **`context`**: `get_context_data()` → `load_images_crop_and_process()` (see §6).

---

## 6) Images and visual features (`local_context`, etc.)

### 6.1 What the model actually sees

The TAMformer core expects **fixed-size vectors per timestep per modality** (not raw RGB inside the transformer). For **`local_context`**, the pipeline:

1. Loads the frame image (OpenCV).
2. Crops an **enlarged square context** around the bbox (`enlarge_ratio`), pads/resizes to **224×224**.
3. Runs a frozen **ImageNet** CNN trunk (`model_opts.backbone`: `vgg16` | `resnet50` | `mobilenet`).
4. **Global-pools** spatial maps to a vector (default **max** pooling → e.g. **512-D** for VGG16).

Set `feat_size` last dimension to match that vector (e.g. **`[4, 1, 512]`** for `box`, `speed`, `local_context` + VGG16).

### 6.2 Disk cache vs no cache (`visual_disk_cache`)

| `visual_disk_cache` | `generator` | Behavior |
|---------------------|---------------|----------|
| `true` (default) | `true` | Writes **`.pkl`** per frame under `data/features/...`. Training reads paths; **needs disk** for cache. First epoch prep can be slow. |
| `false` | `true` | **No** feature `.pkl` files. `load_images_crop_and_process()` stores small **live specs** (path + bbox + crop params) in an **`object`** numpy array. **`DataGenerator`** runs the CNN **each batch** → **zero extra feature storage**, **much slower** training, more GPU work. |
| `false` | `false` | **Unsupported** (raises): non-generator mode expects precomputed arrays. |

---

## 7) Batch generator (`DataGenerator`)

- Subclasses **`Sequence`** and calls **`super().__init__(**kwargs)`** (Keras 3 / `PyDataset` expectation).
- Yields **`(tuple(X), y)`** — Keras 3’s data adapter requires **`tuple`**, not a **`list`**, for multi-input.
- **`y`** is a **1D int32 vector** of motion class ids per batch row when **`predict_location`** is **false**. When **`predict_location`** is **true**, **`y`** is a **dict** `{'motion': ..., 'location': ...}` aligned with the model’s named outputs.
- **`_generate_X`**: for cached visual inputs, elements are **`.pkl` paths`** (`str`). For **`visual_disk_cache: false`**, each timestep is a **`__LIVE_VISUAL__`** tuple; the generator runs **`_live_visual_vector()`** using `opts['_live_visual_parent']` (the `DataGetter`) for **`jitter_bbox` / `squarify` / `img_pad`** consistency with the cache path.

---

## 8) Model architecture (`tamformer.py`) — Keras 3 notes

- Mask assembly uses **`keras.ops`** (`expand_dims`, `transpose`, `mean`, `square`, …) inside **`Lambda`**, not raw **`tf.*`** on **`KerasTensor`**.
- **`TransformerBlock.call(..., training=None, ...)`** — `training` must default to **`None`** so symbolic builds do not require a positional `training` argument.
- Attention mask **rounding** runs only when **`training is False`** (inference). Rounding when `training` was **`None`** previously disconnected mask subnets from the loss (no gradients).
- **Classifier:** default is one **`Dense(num_classes, softmax, name='output')`**. If **`model_opts.predict_location`** is **true**, that layer is replaced by two heads on the same features: **`Dense(..., name='motion')`** and **`Dense(num_location_classes, softmax, name='location')`** (`outputs=[motion, location]`).
- **`ModelCheckpoint`** with **`save_weights_only=True`** must use a filepath ending in **`.weights.h5`**; `run.py` uses that and can fall back to legacy **`.h5`** when **loading** old files. Dual-head checkpoints only load into a dual-head graph (and vice versa for single-head weights).

---

## 9) Training flow (`run.py`)

1. Load YAML.
2. **`custom_json`**: optional split JSONs + optional **`path_to_frames_root`** passed into **`TrackJSONAdapter`**.
3. **`DataGetter`** / **`DataGenerator`** for train, val, test.
4. Build **`TAMformer`**, compile:
   - **Single head:** one weighted sparse categorical loss + **`sparse_categorical_accuracy`**.
   - **Dual head (`predict_location`):** per-head losses (**`motion`**, **`location`**), optional **`class_weights`** / **`location_class_weights`** via **`class_weights(..., head='motion'|'location')`**, equal **`loss_weights`** by default, per-output **`sparse_categorical_accuracy`**.
5. **`fit`** with checkpoint on **`val_loss`** (sum of heads when dual).
6. Reload best weights, **`predict`** on test, print metrics (dual: motion + location metrics, joint accuracy, sample lines with string names; **`test_data['data'][1]`** is a **`dict`** with **`motion`** / **`location`** arrays).

---

## 10) Integration changelog (high level)

- **Keras 3 / TF 2.16+:** removed **`tensorflow.compat.v1.keras`** usage; **`tf.reduce_mean`** in losses; no **`Session` / `set_session`**.
- **Mask gradients:** round masks only when **`training is False`**.
- **Splits:** `path_to_json_train` / `val` / `test` supported for **`custom_json`**.
- **Frames on disk:** `path_to_frames_root` + adapter path resolution.
- **Visual:** `local_context` (etc.) + **`visual_disk_cache`** live vs disk modes; safe cache filenames (sanitize `trackID`); **`RECORD/DRIVE`**-aware cache folders.
- **Data adapter:** **`tuple(X)`**; **`Sequence`** **`super().__init__`**.
- **Dual-head motion + location:** JSON **`location`** field, **`location_activities`**, **`predict_location`** config, two softmax heads, dict **`y`** from the generator, extended **`run.py`** eval and visual sample headers/filenames.

---

## 11) How to run

**Single-head (motion only):**

```bash
python run.py --config_file configs/configs_custom_json.yaml
```

**Dual-head (motion + location):**

```bash
python run.py --config_file configs/configs_custom_json_motion_location.yaml
```

**`custom_json` checklist in YAML:**

- `model_opts.dataset: custom_json`
- `model_opts.obs_input_type` / `feat_size` aligned (include **`512`** if VGG16-pooled **`local_context`** is last).
- Prefer **`path_to_json_train` / `val` / `test`** over a single `path_to_json`.
- **`data_opts.path_to_frames_root`** if JSON keys are not absolute valid image paths.
- **`visual_disk_cache`**: `false` if you cannot store `data/features/...` (trade speed/disk).
- **Dual-head:** set **`predict_location: true`**, **`num_location_classes: 3`**, **`location_class_weights`** as needed; use a **separate `model_path`** (as in **`configs_custom_json_motion_location.yaml`**) so weights files do not collide with single-head training. Ensure each **`objs[]`** entry includes **`location`** when you care about that head’s supervision (otherwise the adapter defaults missing values to **center**).

**Sanity-check frames (optional):**

```bash
python scripts/visualize_custom_json_inputs.py \
  --json_path .../train.json \
  --frames_root .../PreventionData \
  --save_path input_sanity.png
```

---

## 12) Practical notes

- **Class imbalance:** `apply_class_weights` can produce very large weights; consider disabling for debugging or capping/smoothing weights if optimization is unstable.
- **No-disk visual mode:** expect **much longer** wall-clock per epoch; lower **batch size** if you hit GPU memory limits.
- **Metrics:** macro AUC needs sufficient class support in test; can be **`0.0`** if sklearn rejects the setup.

---

## 13) Mental model

- **`data_generator.py`:** tracks → fixed-length tensors (and optional CNN features per frame); optional parallel **`location`** sequences for dual-head training.
- **`tamformer.py`:** temporal attention + masks + cross-attention → one or two softmax heads (motion / optional location).
- **`run.py`:** configuration, training loop, checkpointing, evaluation (including joint motion+location accuracy when dual-head).

That is the end-to-end picture for this repository’s **`custom_json`** + **optional image** stack, with optional **motion + location** dual-head mode.
