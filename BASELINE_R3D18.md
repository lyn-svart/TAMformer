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

Checkpoints: `<SOURCE>/../checkpoints/best_r3d18.pt` (or `SAVE_DIR`).

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
