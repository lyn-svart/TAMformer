# TAMformer Model and Code Guide

This document explains how the model works, how data flows through the codebase, and how the **custom `trackID`-based JSON** pipeline is wired (including **images** and **Keras 3**).

## 1) Big picture

TAMformer is a **sequence classification** model.

- **Input:** a short observation window (`obs_length`) of **per-frame features** per modality (e.g. box, speed, optional CNN visual features).
- **Output:** a class prediction over **`num_classes`** (config-driven; your setup often uses **21** motion classes from JSON `motion` strings).
- **Core idea:** temporal transformers per modality with **learned attention masks**, then **cross-attention** using the **last timestep** as query, then a small MLP + **softmax**.

Each training sample is one **object track** (grouped by `trackID`), not a single frame.

---

## 2) Main files and responsibilities

| File | Role |
|------|------|
| `run.py` | Loads YAML, builds datasets, compiles model, trains, checkpoints, evaluates. TensorFlow import and GPU/thread config run **before** heavy imports. |
| `data_generator.py` | `TrackJSONAdapter`, `DataGetter`, `DataGenerator` (`Sequence`): JSON → track sequences → batches. Visual pipeline (VGG/ResNet/MobileNet crops + features). |
| `tamformer.py` | Functional Keras model: masking nets, `TransformerBlock`, classifier. Written for **Keras 3** (`keras.ops`, no `tf.*` on symbolic tensors in the graph). |
| `configs/configs_custom_json.yaml` | Example config for `dataset: custom_json` (modalities, splits, frames root, optional no-disk visual mode). |
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
- **Class:** `motion` string → integer via `TrackJSONAdapter.MOTION_TO_CLASS` (extended set; unknown strings map to **`stopped`**-class id **3** by default). Final sequence label uses the **last timestep** class in `DataGetter.get_data_sequence()`.

Adapter output keys used downstream: `image`, `pid`, `bbox`, `center`, `obd_speed`, `activities`, `image_dimension`.

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
2. **`obs_length`** = `obs_seconds * (interval / fstride)` (clamped to at least 1).
3. Pads short tracks from the **start** (repeat first frame), then keeps the **last** `obs_length` steps.
4. Builds **`labels`** from the last timestep’s class.

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
- **`_generate_X`**: for cached visual inputs, elements are **`.pkl` paths`** (`str`). For **`visual_disk_cache: false`**, each timestep is a **`__LIVE_VISUAL__`** tuple; the generator runs **`_live_visual_vector()`** using `opts['_live_visual_parent']` (the `DataGetter`) for **`jitter_bbox` / `squarify` / `img_pad`** consistency with the cache path.

---

## 8) Model architecture (`tamformer.py`) — Keras 3 notes

- Mask assembly uses **`keras.ops`** (`expand_dims`, `transpose`, `mean`, `square`, …) inside **`Lambda`**, not raw **`tf.*`** on **`KerasTensor`**.
- **`TransformerBlock.call(..., training=None, ...)`** — `training` must default to **`None`** so symbolic builds do not require a positional `training` argument.
- Attention mask **rounding** runs only when **`training is False`** (inference). Rounding when `training` was **`None`** previously disconnected mask subnets from the loss (no gradients).
- **`ModelCheckpoint`** with **`save_weights_only=True`** must use a filepath ending in **`.weights.h5`**; `run.py` uses that and can fall back to legacy **`.h5`** when **loading** old files.

---

## 9) Training flow (`run.py`)

1. Load YAML.
2. **`custom_json`**: optional split JSONs + optional **`path_to_frames_root`** passed into **`TrackJSONAdapter`**.
3. **`DataGetter`** / **`DataGenerator`** for train, val, test.
4. Build **`TAMformer`**, compile (weighted sparse CE unless disabled).
5. **`fit`** with checkpoint on **`val_loss`**.
6. Reload best weights, **`predict`** on test, print metrics.

---

## 10) Integration changelog (high level)

- **Keras 3 / TF 2.16+:** removed **`tensorflow.compat.v1.keras`** usage; **`tf.reduce_mean`** in losses; no **`Session` / `set_session`**.
- **Mask gradients:** round masks only when **`training is False`**.
- **Splits:** `path_to_json_train` / `val` / `test` supported for **`custom_json`**.
- **Frames on disk:** `path_to_frames_root` + adapter path resolution.
- **Visual:** `local_context` (etc.) + **`visual_disk_cache`** live vs disk modes; safe cache filenames (sanitize `trackID`); **`RECORD/DRIVE`**-aware cache folders.
- **Data adapter:** **`tuple(X)`**; **`Sequence`** **`super().__init__`**.

---

## 11) How to run

```bash
python run.py --config_file configs/configs_custom_json.yaml
```

**`custom_json` checklist in YAML:**

- `model_opts.dataset: custom_json`
- `model_opts.obs_input_type` / `feat_size` aligned (include **`512`** if VGG16-pooled **`local_context`** is last).
- Prefer **`path_to_json_train` / `val` / `test`** over a single `path_to_json`.
- **`data_opts.path_to_frames_root`** if JSON keys are not absolute valid image paths.
- **`visual_disk_cache`**: `false` if you cannot store `data/features/...` (trade speed/disk).

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

- **`data_generator.py`:** tracks → fixed-length tensors (and optional CNN features per frame).
- **`tamformer.py`:** temporal attention + masks + cross-attention → class logits.
- **`run.py`:** configuration, training loop, checkpointing, evaluation.

That is the end-to-end picture for this repository’s **`custom_json`** + **optional image** stack.
