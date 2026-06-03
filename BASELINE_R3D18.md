# R3D-18 baseline (aligned with dual-head TAMformer)

Compare **motion** test metrics only against:

```bash
python run.py --config_file configs/configs_custom_json_motion_location.yaml
```

## Dependencies (PyTorch baseline)

```bash
pip install torch torchvision tqdm
```

## Dataset paths

Use the same directory for both models (`SOURCE`):

- `Train.json`, `Validation.json`, `Test.json`
- Frame tree under `path_to_frames_root` / `--frames-root` (see YAML and `run_r3d18.sh`)

Update paths in:

- `configs/configs_custom_json_motion_location.yaml` → `data_opts.path_to_json_*` and `path_to_frames_root`
- `run_r3d18.sh` → `SOURCE` default or env override

## Train R3D-18

```bash
SOURCE=/path/to/PreventionData \
CLIP_LEN=10 FRAME_KEEP_MOD=1 \
BATCH=64 LR=1e-4 EPOCHS=20 PATIENCE=20 SEED=42 \
bash run_r3d18.sh
```

Performance-oriented run (same data semantics, faster pipeline):

```bash
SOURCE=/path/to/PreventionData \
BATCH=128 WORKERS=16 PREFETCH_FACTOR=4 CACHE_SIZE=50000 COMPILE=1 \
bash run_r3d18.sh
```

Checkpoints: `<SOURCE>/../checkpoints/best_r3d18.pt` (or `SAVE_DIR`).

## Result logs (txt)

Logs are written under the results directory (created at startup):

| File | R3D-18 | TAMformer |
|------|--------|-----------|
| Training | `<repo>/training_results.txt` (default) | `<results_log_dir>/training_results.txt` |
| Test | `<repo>/test_results.txt` (default) | `<results_log_dir>/test_results.txt` |

Default (next to the code):

```text
.../TAMformer/training_results.txt
.../TAMformer/test_results.txt
```

Checkpoints still go to `<SOURCE>/../checkpoints/best_r3d18.pt` unless you set `SAVE_DIR`.

The startup banner prints the **absolute path** as `training_log`. Watch it live:

```bash
tail -f /path/to/checkpoints/results/training_results.txt
```

Override R3D: `RESULTS_DIR=/path/to/logs bash run_r3d18.sh`  
TAMformer: set `model_opts.results_log_dir` in the YAML (default `./models_motion_location/results`).

### Mid-epoch progress lines

Every `LOG_EVERY_N_BATCHES` (default 50) during training:

```text
epoch=1 step=50/5033 sec/batch=10.20 samp/sec=6.27 loss=1.12
```

You do **not** need to finish an epoch to estimate speed.

### Throughput benchmark: skip bbox crop (not for final metrics)

Measures how much time bbox crop+resize costs. Uses **full frame** resized to 112×112 (still reads/decodes full images; still one resize per frame).

```bash
SKIP_CROP_RESIZE=1 FRACTION=0.05 MAX_TRAIN_BATCHES=200 LOG_EVERY_N_BATCHES=20 \
CHUNK_STRIDE=4 BATCH=64 WORKERS=8 bash run_r3d18.sh
```

Or: `python r3d18.py --source ... --skip-crop-resize ...`

Compare `sec/batch` in `training_results.txt` with and without this flag. **Do not** use for paper numbers vs TAMformer.

### Quick benchmark (minutes, not hours)

```bash
SOURCE=/path/to/PreventionData \
FRACTION=0.05 EPOCHS=1 CHUNK_STRIDE=4 \
BATCH=64 WORKERS=8 PREFETCH_FACTOR=2 CACHE_SIZE=64 \
MAX_TRAIN_BATCHES=200 LOG_EVERY_N_BATCHES=20 \
bash run_r3d18.sh
```

Then read `sec/batch` from the log and estimate:

```text
steps_per_epoch ~= ceil(train_samples / BATCH)
epoch_seconds ~= steps_per_epoch * sec_per_batch
```

## Diagnose epoch duration from logs

In `training_results.txt`, use:

- `building train...` / `train samples: N` (appears as each split loads)
- `samples train/val/test: X/Y/Z` for dataset scale
- `train steps/epoch (approx): ...`
- mid-epoch `sec/batch` and `samp/sec` lines (every N batches)
- end-of-epoch summary columns (`epoch`, `train_loss`, `val_loss`, ...)

If `train_samples` is very large and `sec/batch` is high, training is data-loader bound.

## Compare metrics

R3D-18 test output uses the same summary line as TAMformer motion eval:

`acc`, `auc_macro_ovr`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`

TAMformer dual-head run prints **motion** metrics in that format; ignore location / joint accuracy when comparing to R3D-18.

## Known differences (methods text)

| Item | TAMformer | R3D-18 |
|------|-----------|--------|
| Input | box + frozen VGG context @ 224 | RGB bbox crop @ 112 |
| Context crop | `enlarge_ratio` 1.5 | `--crop-pad` 0.10 |
| Heads | motion + location | motion only |
| Optimizer | Adam | Adam |
| Class weights | off (`apply_class_weights: false`) | off (plain CE) |

Sampling: both use **chunk_dt / clip_len = 10** sliding windows per track, label = **last frame** in the window.
