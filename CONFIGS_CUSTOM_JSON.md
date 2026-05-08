# `configs/configs_custom_json.yaml` Field Guide

This file explains the meaning of each key in `configs/configs_custom_json.yaml` as used by this repository.

## Overview

- `model_opts`: Model architecture, training behavior, feature extraction, and debug/preview settings.
- `data_opts`: Dataset loading and split paths for `custom_json`.

---

## `model_opts`

### Core model/data settings

- `model`
  - Descriptive in current code path; not heavily used for branching.

- `backbone`
  - Visual CNN backbone (`vgg16`, `resnet50`, `mobilenet`) used for visual modalities.

- `obs_input_type`
  - Input modalities used by the model, in order.
  - Example: `[box, local_context]` means geometry + visual context features.

- `dataset`
  - Must be `custom_json` to use `TrackJSONAdapter` flow in `run.py`.

- `num_classes`
  - Number of motion classes for output softmax.

### Sequence/timing

- `obs_seconds`
  - Main observation window duration.
  - Runtime computes `obs_length = obs_seconds * fps`.

- `interval`
  - Used with `data_opts.fstride` to compute effective FPS.

- `obs_length`
  - Present in config, but runtime recalculates/overwrites from `obs_seconds` and FPS.

- `seq_len`
  - Also overwritten at runtime to match computed `obs_length`.

- `time_to_event`
  - Legacy field; not active in current custom_json training path.

- `min_encoding_len`
  - Legacy field; not active in current custom_json training path.

- `step`
  - Legacy field; currently unused in main flow.

### Geometry/crop behavior

- `enlarge_ratio`
  - Enlargement factor for `local_context` / `surround` crop generation.

- `normalize_boxes`
  - Exists for compatibility; current custom_json flow does not heavily branch on it.

### Training behavior

- `batch_size`
  - Train batch size.

- `val_batch_size`
  - Validation batch size.

- `epochs`
  - Number of training epochs.

- `optimizer`
  - `adam`, `sgd`, or `rmsprop`.

- `lr`
  - Learning rate.

- `dropout`
  - Config value exists, but current TAMformer dropout is mostly hardcoded in `tamformer.py`.

- `model_path`
  - Directory for model checkpoints and default visual sample output.

### Class balancing/loss

- `balance_data`
  - If `true`, train sequences are balanced by oversampling logic in `DataGetter`.

- `apply_class_weights`
  - Enables weighted sparse categorical loss in `run.py`.

- `class_weights`
  - Base class multipliers (length should match `num_classes`).
  - Combined with inverse-frequency weighting when `apply_class_weights=true`.

- `classifier_activation`
  - Descriptive; model head is currently softmax in code.

- `classifier_loss`
  - Descriptive; compile path uses weighted sparse categorical loss wrapper.

### Feature shape control

- `auto_feat_size`
  - If `true`, derive `feat_size` automatically from `obs_input_type` + backbone.

- `feat_size`
  - Used only when `auto_feat_size=false`.

### Generator/cache settings (important for runtime and speed)

- `generator`
  - `true`: use `DataGenerator` (stream batches, lower RAM, slower startup/IO path).
  - `false`: eager/full arrays in memory (higher RAM, faster per-step after preload).

- `visual_disk_cache`
  - If `true`, saves visual features as `.pkl` to disk.

- `visual_disk_cache_train_only`
  - If `true`, only training split writes/reads disk cache; val/test are live.

- `visual_cache_pooled`
  - If `true`, cache pooled vectors (e.g., 512-d for VGG16) instead of full feature maps.
  - Greatly reduces disk and read time.

### Debug/evaluation outputs

- `sample_inference_count`
  - Number of sample text predictions printed after test.

- `visual_sample_count`
  - Number of visual sample mosaics to save.

- `visual_sample_frames`
  - Frames per visual sample (up to 9 in current 3x3 layout).

- `visual_sample_out_dir`
  - Output directory for saved visual sample images.

- `visual_sample_crop_type`
  - Crop style for visual samples:
  - `auto`, `bbox`, `context`, `surround`, `none`.
  - `auto` resolves from `obs_input_type`.

- `visual_sample_draw_header`
  - If `true`, draw metadata header text on top of visual sample mosaics.
  - If `false`, keep visuals clean (no drawn text on image).

- `visual_sample_before_training`
  - If `true`, generate visual previews immediately before training starts.
  - Useful when you want to inspect inputs without waiting for full training.

- `data_augmentation`
  - Currently not a major active branch in this custom_json flow.

---

## `data_opts`

- `fstride`
  - Frame stride used in sequence timing; affects effective FPS and computed `obs_length`.

- `sample_type`
  - Legacy JAAD/PIE-oriented option; mostly non-critical for custom_json.

- `subset`
  - Legacy JAAD/PIE-oriented option.

- `data_split_type`
  - Legacy JAAD/PIE-oriented option.

- `seq_type`
  - Legacy JAAD/PIE-oriented option.

- `min_track_size`
  - Minimum accepted track length; runtime generally aligns this with computed `obs_length`.

- `path_to_json_train`
  - Training split JSON path (preferred custom_json split mode).

- `path_to_json_val`
  - Validation split JSON path.

- `path_to_json_test`
  - Test split JSON path.

- `path_to_frames_root`
  - Root directory used to resolve frame image files referenced by JSON keys.

- `path_to_json`
  - Legacy single-file fallback mode (used if explicit train/val/test paths are missing).

---

## Notes and recommendations

- If you changed crop behavior (e.g., `enlarge_ratio`) and use disk cache, clear relevant cached features before strict comparison experiments.
- For fast debug runs:
  - keep `visual_cache_pooled: true`
  - reduce `epochs`
  - optionally lower `visual_sample_count`.
- For custom_json, prioritize configuring:
  - `obs_input_type`, `num_classes`, `obs_seconds`, `fstride`
  - `path_to_json_train/val/test`, `path_to_frames_root`
  - cache/generator settings.
