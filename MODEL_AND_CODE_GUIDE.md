# TAMformer Model and Code Guide

This document explains how the model works, how data flows through the codebase, and what was changed for your `trackID`-based JSON format.

## 1) Big Picture

TAMformer is a sequence classification model.

- **Input:** a short observation window (`obs_length`) of per-frame features.
- **Output:** a class prediction (`num_classes`, here 5 classes).
- **Core idea:** transformer attention over time and modalities, with learned masks.

In your current setup, each training sample is one **object track** (grouped by `trackID`) instead of frame-based grouping.

---

## 2) Main Files and Responsibilities

- `run.py`
  - Entry point for training/testing.
  - Loads config, builds dataset objects, builds model, trains, evaluates.
- `data_generator.py`
  - Converts raw dataset structures into fixed-length sequences.
  - Provides Keras `Sequence` generator for batched training.
  - Contains `TrackJSONAdapter` for your custom JSON format.
- `tamformer.py`
  - Defines the neural network (Transformer blocks + classifier head).
- `configs/configs_custom_json.yaml`
  - Your custom config for `dataset: custom_json`.

---

## 3) Your Custom JSON Pipeline (trackID-based)

## 3.1 Input JSON shape
Your JSON is frame-keyed:

- key: `"RECORDX/DRIVEY/frames/000123.png"`
- value: object with `objs`, and each object has fields like:
  - `trackID`
  - `xywh` (normalized)
  - `motion`
  - `img_width`, `img_height`
  - `Vx`, `Vz`, etc.

## 3.2 Regrouping by track
`TrackJSONAdapter` in `data_generator.py`:

1. Reads all frame entries.
2. Iterates through each object.
3. Groups them by `trackID`.
4. Sorts each track chronologically by frame number parsed from image path.

So each sample becomes a **track sequence**.

## 3.3 Feature/label creation
For each object instance in a track:

- `xywh` normalized -> absolute `[x1, y1, x2, y2]` using image size.
- `bbox center` is computed from corners.
- `speed` is derived from `sqrt(Vx^2 + Vz^2)` with NaN-safe conversion.
- `motion` string -> 5-class id:
  - `approaching: 0`
  - `leaving: 1`
  - `crossing: 2`
  - `stopped: 3`
  - `standing: 4`

The adapter returns fields compatible with existing TAMformer data code:

- `image`, `pid`, `bbox`, `center`, `obd_speed`, `activities`, `image_dimension`

---

## 4) Sequence Preparation (`DataGetter`)

`DataGetter.get_data_sequence()` converts variable-length tracks into model-ready fixed windows:

1. Reads raw arrays (`bbox`, `pid`, `activities`, etc.).
2. Computes `obs_length` from config:
   - `obs_length = obs_seconds * (interval / fstride)`
3. For each sequence:
   - if shorter than `obs_length`: pad by repeating first frame at start
   - keep only last `obs_length` frames
4. Converts lists to `numpy` arrays.
5. Builds final per-sample label from last frame class.

This output is then wrapped by `DataGenerator` when `generator: True`.

---

## 5) Batch Generator (`DataGenerator`)

`DataGenerator` is a Keras `Sequence` that yields `(X, y)`:

- `X` is a list of modality tensors based on `obs_input_type`:
  - `box`: shape `[batch, obs_length, 4]`
  - `speed`: shape `[batch, obs_length, 1]`
  - (and optionally visual context features if used)
- `y` is shape `[batch]`, class ids.

Normalization:

- Box coordinates are normalized by image dimensions.
- Context image features are normalized for VGG-like encoders when used.

---

## 6) Model Architecture (`tamformer.py`)

`TAMformer.tamformer()` builds the model:

1. **Inputs per modality**
   - One `Input((obs_length, feat_size_i))` per modality.
2. **PositionEmbedding**
   - Adds temporal position information.
3. **Per-modality temporal transformers**
   - Self-attention with learned temporal masks (`masks_obs`).
4. **Query-based cross-attention**
   - Uses the last timestep as query.
   - Attends to concatenated modality encodings.
5. **Classifier head**
   - Dense layers + dropout -> final softmax over classes.

Output:

- `Dense(num_classes, activation='softmax')`

---

## 7) Training and Evaluation Flow (`run.py`)

1. Load YAML config.
2. Build raw data:
   - `dataset: custom_json` -> use `TrackJSONAdapter`.
   - otherwise use PIE/JAAD loaders.
3. Build `DataGetter` for train/val/test.
4. Build TAMformer model.
5. Compile with weighted sparse categorical cross entropy.
6. Train with `ModelCheckpoint` saving best `val_loss`.
7. Reload best weights.
8. Run predictions on test split and print metrics:
   - accuracy, macro/weighted F1, precision, recall, macro AUC (if possible).

---

## 8) Important Fixes Applied During Integration

1. **Cross-attention mask None bug**
   - In `TransformerBlock.call()`, `tf.round(attention_mask)` now runs only when mask is not `None`.
2. **Label shape robustness**
   - `_generate_y()` now handles scalar or sequence-shaped labels.
3. **Checkpoint path creation**
   - `run.py` now creates `model_path` automatically before saving.

---

## 9) How to Run (custom JSON)

Use:

```bash
python run.py --config_file configs/configs_custom_json.yaml
```

Make sure in config:

- `model_opts.dataset: custom_json`
- `data_opts.path_to_json: /absolute/path/to/Train.json`

---

## 10) Practical Notes for Your Current Data

- Current class distribution is imbalanced (class `4` dominant).
- Class weights are already computed and applied.
- If training is unstable or accuracy is low:
  - reduce learning rate further,
  - increase epochs,
  - or provide balanced train/val/test splits by track.

---

## 11) Mental Model Summary

Think of each `trackID` as one short video snippet of one object.

- `data_generator.py` turns that snippet into fixed-length numeric sequences.
- `tamformer.py` learns temporal patterns from those sequences.
- `run.py` orchestrates train/validate/test and reporting.

That is the full end-to-end pipeline in this repository.
