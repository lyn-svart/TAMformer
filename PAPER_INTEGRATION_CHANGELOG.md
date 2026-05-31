# TAMformer Integration Changelog (for Paper Methods)

This document records **all changes on your fork** compared to the upstream TAMformer release, so you can describe them accurately in a paper (methods, experiments, and implementation notes).

**Comparison baseline**

| Item | Value |
|------|--------|
| Upstream repo | [NadaSOsman/TAMformer](https://github.com/NadaSOsman/TAMformer) (`upstream/master`) |
| Your fork | [lyn-svart/TAMformer](https://github.com/lyn-svart/TAMformer) (`origin/main`) |
| Common ancestor | `14ca97e` (last shared commit before your work) |
| Commits on your fork only | **46** (`87b04b4` … `18bedb2`, 2026-04-28 → 2026-05-12) |
| Diff size | **23 files**, ~**+3,098 / −313** lines (vs upstream) |

**How this was produced:** `git log upstream/master..main` and `git diff upstream/master..main`, plus reading commit messages and key source files. No GitHub API was required; the same history is on your local clone.

---

## 1. Executive summary (what changed at a high level)

The upstream codebase implements the **original TAMformer paper setup**: multi-step **binary crossing** prediction (40 sigmoid heads, fixed temporal length 136, `pose` + geometry + visual context, TensorFlow 1.x–style Keras usage).

Your fork turns TAMformer into a **flexible sequence classifier** suitable for your prevention / object-motion JSON data, with three major layers of work:

1. **Task refactor (JAAD / PIE path)** — From 40-step binary crossing to **single-step multiclass motion** (initially 5 canonical classes), dynamic `obs_length`, one softmax head, weighted sparse categorical loss, and expanded evaluation metrics.

2. **Custom JSON dataset path** — New `TrackJSONAdapter`, `dataset: custom_json`, train/val/test JSON splits, **21 motion classes** from string labels, VGG/ResNet/MobileNet visual features with disk/live caching, tooling scripts, and extensive docs.

3. **Dual-head extension** — Optional simultaneous prediction of **motion** (21 classes) and **location** (left / center / right), with separate config, losses, metrics, and checkpoint directory.

The **core TAMformer idea is preserved** (per-modality transformers with learned masks, cross-attention from the current timestep, shared trunk). What changed is the **output head**, **temporal sizing**, **data adapters**, and **training/eval plumbing**.

---

## 2. Original vs current behavior (paper framing)

| Aspect | Upstream (paper code) | Your fork (`main`) |
|--------|----------------------|-------------------|
| Primary task | Multi-horizon **binary crossing** (40 outputs) | **Single-step multiclass** motion (and optionally location) |
| Output activation | 40× sigmoid | 1× (or 2×) **softmax** |
| Loss | `weighted_binary_crossentropy` | `weighted_sparse_categorical_crossentropy` |
| Temporal length | Hardcoded **136** frames; subsampled masks for **40** steps | **`obs_length`** from `obs_seconds × fps` (typically 1 s window) |
| Input modalities | Included **`pose`** among others | **`box` + `local_context`** (pose removed from configs) |
| JAAD / PIE labels | Crossing-oriented binary pipeline | Canonical motion class mapping (5-class path in JAAD/PIE adapters) |
| Your data | Not supported | **`custom_json`**: frame-keyed JSON → per-`trackID` sequences |
| Motion classes (custom JSON) | — | **21** string labels (`MOTION_TO_CLASS` in `data_generator.py`) |
| Location | — | Optional **3-class** head (`left`, `center`, `right`) |
| Framework details | `tf.*` in model graph | **`keras.ops`** (Keras 3–friendly); mask rounding only at inference |
| Weights files | `.h5` | **`.weights.h5`** (Keras 3) with legacy `.h5` fallback |

---

## 3. Chronological commit log (grouped by theme)

Commits are listed oldest → newest. Hash + date + subject; details from commit bodies where available.

### Phase A — 5-class single-step refactor (2026-04-28)

| Commit | Summary |
|--------|---------|
| `87b04b4` | **Refactor to 5-class single-step motion task.** Removes `pose`; `box + local_context`; `obs_seconds: 1`; `num_classes: 5`; softmax + sparse CE; dynamic `obs_length`; single softmax head in `tamformer.py`; multiclass metrics in `run.py`; JAAD/PIE motion maps; docs: `ONBOARDING_QUICKSTART.md`, `IMPLEMENTATION_SUMMARY_TR_EN.md`, `PROJECT_OVERVIEW_FINAL.md`. |

### Phase B — Custom JSON pipeline (2026-04-30 → 2026-05-04)

| Commit | Summary |
|--------|---------|
| `55c4c95` | **`TrackJSONAdapter`**, `configs/configs_custom_json.yaml`, `requirements.txt`; `run.py` wires `dataset == custom_json`. |
| `c7f8c2c` | Path corrections. |
| `2c95334` | `data_generator` updates for custom JSON flow. |
| `6e98415` | `run.py`: `makedirs` for output dirs. |
| `4cdbaef` | `scripts/show_track_sequence_example.py` — debug one track’s tensor shapes. |
| `fed2bc2` | Track IDs scoped per video (`sequence_id::trackID`). |
| `1a8570f` | Sliding-window chunking in adapter (`chunk_dt`). |
| `51dc855` | Window size correction. |
| `e2ebbd2` | Expanded **motion class** vocabulary (toward 21 classes). |
| `461c8bc` | GPU setup in `run.py`. |
| `7595f89` | TensorFlow version pin (see `requirements.txt`). |
| `86b2f98` | **`keras.ops`**; mask rounding only at inference; simplified aux loss. |
| `6d545a4` | Further `tamformer.py` fixes. |
| `4aacad2` | Keras 3 **`.weights.h5`** + legacy `.h5` load. |
| `79deea6`, `061e04b`, `3e3a7f5` | `data_generator.py` iteration (batches, visuals, edge cases). |
| `2f5489b` | `scripts/visualize_custom_json_inputs.py`; mask rounding refinement. |
| `3028fcc` | Basename index + `frames_root` for visualization. |
| `23f40d9`, `10bfd72`, `f05e526` | Visualization script updates. |
| `edbd042` | Explicit **train / val / test** JSON paths in config + `run.py`. |
| `c0d1bb4` | **`path_to_frames_root`** / `frames_root` in adapter. |
| `246c337` | **Live visual features** (no disk cache). |
| `4b5b709` | Tuple sequences as object arrays (NumPy compatibility). |
| `db2c2f2` | **Visual disk cache** options + doc updates. |
| `7ac77c0` | **`auto_feat_size`**; MobileNet default in some configs. |
| `ee58ece` | **Pooled visual cache** (smaller/faster `.pkl` vectors). |
| `5464182` / `c0ef791` | Sharded cache experiment **reverted**. |
| `f361a1b`, `04d1f7a` | `run.py` / caching toggles. |
| `72c4abf` | VGG16 backbone; option to run **without speed** modality. |

### Phase C — Training UX, metrics, docs (2026-05-05 → 2026-05-12)

| Commit | Summary |
|--------|---------|
| `7938890` | **Visual sample grids** after inference (bbox + true/pred/conf). |
| `61b7443` | Diverse visual samples + metadata JSON sidecars. |
| `0dcbb21` | Crop types for previews (`bbox` / `context` / `surround`). |
| `af62816` | Preview tiles; optional header overlay toggle. |
| `30ba62f` | **Per-class accuracies** in evaluation. |
| `444eaeb` / `cf888da` | Fixed context window option added then **removed**. |
| `00ff822` | `CONFIGS_CUSTOM_JSON.md`; cache fixes. |
| `4e3e75f` | Faster caching path. |
| `93f7a3f` | `scripts/check_motion_labels.py` — validate JSON labels vs `MOTION_TO_CLASS`. |
| `f3d933b` | `configs_custom_json.yaml` updates. |
| `4397030` | Minor follow-up. |
| `d8210b6` | **Dual-head**: motion + location multiclass. |
| `18bedb2` | Document dual-head in `MODEL_AND_CODE_GUIDE.md`. |

---

## 4. Files added

| File | Purpose |
|------|---------|
| `configs/configs_custom_json.yaml` | Main experiment config: 21 motion classes, VGG16, visual cache, JSON split paths, visual sample settings. |
| `configs/configs_custom_json_motion_location.yaml` | Same as above + **`predict_location: true`**, separate `model_path`. |
| `requirements.txt` | Pinned stack: NumPy, TF 2.16.1+cuda, OpenCV, PyYAML, sklearn, Pillow. |
| `scripts/split_custom_json_tracks.py` | Track-disjoint train/val/test JSON split. |
| `scripts/visualize_custom_json_inputs.py` | Overlay bbox + motion on frames for sanity checks. |
| `scripts/show_track_sequence_example.py` | Print raw vs processed shapes for one track. |
| `scripts/check_motion_labels.py` | Scan JSON for unknown `motion` strings vs adapter map. |
| `CONFIGS_CUSTOM_JSON.md` | Field-by-field YAML guide. |
| `MODEL_AND_CODE_GUIDE.md` | End-to-end architecture + data flow (English, paper-ready). |
| `ONBOARDING_QUICKSTART.md` | Quick start (5-class JAAD/PIE focus). |
| `IMPLEMENTATION_SUMMARY_TR_EN.md` | TR/EN summary of Phase A refactor. |
| `PROJECT_OVERVIEW_FINAL.md` | High-level project snapshot (TR + EN). |
| `PAPER_INTEGRATION_CHANGELOG.md` | **This file.** |

**Note:** Three `__pycache__/*.pyc` files appear in the diff; they should not be committed long-term (artifact noise, not part of the method).

---

## 5. Files modified (by component)

### 5.1 `tamformer.py` (~101 lines changed)

- **Removed** hardcoded `136` / `40` mask banks and multi-sigmoid heads.
- **Added** `obs_length`-sized learned masks (one small MLP per timestep).
- **Cross-attention** query is the **last timestep** (`current_query`), not a subsampled window.
- **Classifier:** `Dense(num_classes, softmax)`; optional second head `location` when `predict_location`.
- **Portability:** `tf.range` / `tf.round` → `keras.ops`; attention masks rounded only when `training is False`.
- **Auxiliary loss:** simplified to match single (or dual) multiclass setup.

### 5.2 `data_generator.py` (+~800 lines net)

**New: `TrackJSONAdapter`**

- Reads frame-keyed JSON: `{ "path/to/frame.png": { "objs": [ ... ] } }`.
- Groups by `trackID` within each `RECORD*/DRIVE*/frames` sequence.
- Builds per-track lists: images, bbox, center, speed (√(Vx²+Vz²)), motion class, location class.
- **`MOTION_TO_CLASS`**: 21 labels (`opposite`, `crossing-tocenter`, …, `intent to cross`).
- **`LOCATION_TO_CLASS`**: `left`→0, `center`→1, `right`→2 (unknown → center).
- **`frames_root`**: resolves relative JSON paths to on-disk frames.
- **`chunk_dt`**: optional sliding windows of length `chunk_dt+1` per track.
- Skips `type == 'Human'` objects.

**Updated: `DataGetter` / `DataGenerator`**

- `obs_length` from `obs_seconds` and FPS; trim/pad sequences to window.
- Single sparse label = **last frame** in window (motion; + location if enabled).
- Visual pipeline: crop `local_context` / `surround` / `local_box`; VGG16/ResNet50/MobileNet features.
- **Caching:** `visual_disk_cache`, `visual_disk_cache_train_only`, `visual_cache_pooled`, live path without cache.
- **`auto_feat_size`**: sets `feat_size` from modalities + backbone.
- Batch `y` as `int32` vector or dict `{'motion', 'location'}` for dual-head.

### 5.3 `run.py` (+~900 lines net)

- TensorFlow/GPU env configured **before** heavy imports.
- **`custom_json` branch:** load train/val/test JSON via `TrackJSONAdapter`; pass `frames_root`, `chunk_dt`.
- **Compile:** `weighted_sparse_categorical_crossentropy`; dual loss/weights for motion + location.
- **Metrics:** accuracy, macro/weighted F1, precision, recall, macro OVR AUC; per-class accuracy.
- **Checkpointing:** Keras 3 weights naming; resume/test paths.
- **Visual samples:** save PNG grids before/after training (`visual_sample_*` config keys).
- **Class weights:** `class_weights(..., head='motion'|'location')` from batch counts.

### 5.4 `jaad_data.py` / `pie_data.py`

- Canonical **5-class motion** mapping for paper datasets (standing, walking, starting_to_move, running, stopping).
- Integrated into `activities` for the refactored multiclass path.
- Unchanged in role; not used when `dataset: custom_json`.

### 5.5 Config YAMLs (`configs_all`, `configs_beh`, `configs_pie`)

- Removed `pose` from `obs_input_type`.
- Added `num_classes: 5`, `obs_seconds: 1`, softmax + sparse CE, `class_weights`.
- Adjusted `feat_size` for `box + local_context`.

---

## 6. Custom JSON data format (for Methods / Dataset section)

**Input annotation file** (one or three split files):

```json
{
  "RECORD1/DRIVE1/frames/000123.png": {
    "objs": [
      {
        "trackID": 42,
        "xywh": [cx, cy, w, h],
        "motion": "approaching",
        "location": "left",
        "img_width": 1920,
        "img_height": 1080,
        "Vx": 0.1,
        "Vz": -0.2
      }
    ]
  }
}
```

**Model inputs (typical config):** normalized bbox sequence + CNN embeddings of enlarged pedestrian crop (`local_context`), 1 s window at dataset FPS.

**Targets:**

- **Motion:** integer in `[0, 20]` from `motion` string.
- **Location (optional):** integer in `{0,1,2}`.

**Splits:** `scripts/split_custom_json_tracks.py` ensures tracks do not leak across train/val/test.

---

## 7. Configuration knobs worth citing in the paper

From `configs/configs_custom_json.yaml` (adjust paths for your environment):

| Parameter | Typical value | Meaning |
|-----------|---------------|---------|
| `obs_input_type` | `[box, local_context]` | Geometry + visual CNN features |
| `backbone` | `vgg16` | Visual encoder |
| `obs_seconds` | `1` | Observation horizon |
| `num_classes` | `21` | Motion vocabulary size |
| `predict_location` | `false` / `true` | Single- vs dual-head |
| `num_location_classes` | `3` | Lane position |
| `visual_disk_cache` | `true` | Precompute CNN features to disk |
| `visual_cache_pooled` | `true` | Store pooled vectors (smaller cache) |
| `path_to_json_train/val/test` | — | Split JSON paths |
| `path_to_frames_root` | — | Root containing `RECORD*/DRIVE*/frames` |

Full key reference: `CONFIGS_CUSTOM_JSON.md`.

---

## 8. How to run (reproducibility paragraph)

**Single-head motion (21 classes):**

```bash
python run.py --config_file configs/configs_custom_json.yaml
python run.py --config_file configs/configs_custom_json.yaml --test
```

**Dual-head motion + location:**

```bash
python run.py --config_file configs/configs_custom_json_motion_location.yaml
```

**Utilities:**

```bash
python scripts/split_custom_json_tracks.py --input all.json --out_dir splits/
python scripts/check_motion_labels.py --json splits/train.json
python scripts/visualize_custom_json_inputs.py --json splits/train.json --frames_root /path/to/PreventionData
python scripts/show_track_sequence_example.py --json_path splits/train.json
```

**JAAD / PIE (5-class path, if still used):**

```bash
python run.py --config_file configs/configs_all.yaml
```

---

## 9. Suggested paper subsections mapped to this work

| Paper section | What to describe from this changelog |
|---------------|--------------------------------------|
| **Model** | TAMformer backbone unchanged in spirit; you use **one softmax** over the last second of observations; optional **location head** shares trunk. |
| **Inputs** | Bounding boxes + **VGG16** (or config backbone) crop features; 1 s temporal window. |
| **Outputs** | 21-way motion; optional 3-way location. |
| **Training** | Weighted sparse categorical cross-entropy; Adam; class weights optional. |
| **Evaluation** | Accuracy, macro F1, precision/recall, OVR AUC; per-class accuracy. |
| **Dataset** | Custom frame-keyed JSON; track-level samples; disjoint splits. |
| **Implementation** | TensorFlow 2.16 / Keras 3 weights; fork URL vs original TAMformer repo. |

---

## 10. Related docs already in the repo

| Document | Best for |
|----------|----------|
| `MODEL_AND_CODE_GUIDE.md` | Detailed pipeline + dual-head behavior |
| `IMPLEMENTATION_SUMMARY_TR_EN.md` | Phase A (5-class JAAD/PIE) only |
| `CONFIGS_CUSTOM_JSON.md` | Every YAML key |
| `ONBOARDING_QUICKSTART.md` | First run on JAAD/PIE configs |

---

## 11. Caveats for the paper author

1. **Two class regimes:** JAAD/PIE configs target **5** motion classes; **custom JSON** uses **21**. State clearly which experiments use which vocabulary.
2. **Upstream citation:** Cite the original TAMformer paper/repo for the architecture; describe this fork for task formulation, data adapter, and training changes.
3. **Config paths** in YAML still point to example machine paths (`/home/bkaradurak/...`); replace before publication/supplementary material.
4. **`__pycache__` in git history:** consider removing from the repo before release.
5. **Docs vs code:** `ONBOARDING_QUICKSTART.md` / `PROJECT_OVERVIEW_FINAL.md` describe the **5-class** milestone; `MODEL_AND_CODE_GUIDE.md` reflects the **latest** custom JSON + 21-class + dual-head state.

---

*Generated from local git history: `upstream/master` (NadaSOsman/TAMformer) → `main` (lyn-svart/TAMformer), 46 commits, clean working tree.*
