"""
R3D-18 motion classification — aligned with TAMformer custom_json / TrackJSONAdapter.

- Fixed 21-class motion vocabulary (motion_labels.py)
- Sliding windows of length clip_len (default 10), step 1 per track
- Label = motion class at last frame in window
- Test metrics match run.py motion-head printout
"""

import os
import re
import json
import random
import argparse
import time
from functools import partial
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter, OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from motion_labels import (
    NUM_MOTION_CLASSES,
    CLASS_ID_TO_NAME,
    motion_to_class,
)
from result_logging import (
    format_motion_metrics_line,
    format_per_class_metrics,
    test_log_path,
    training_log_path,
    write_run_header,
    write_test_results,
    append_epoch_line,
    append_progress_line,
)

PAT = re.compile(
    r"(?:^|[\\/])(RECORD[^\\/]+)[\\/](DRIVE[^\\/]+)[\\/]frames[\\/](\d+)\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)

torch.backends.cudnn.benchmark = True

# Directory containing this script (default location for training_results.txt)
CODE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "r3d18"


def parse_args():
    parser = argparse.ArgumentParser(
        description="R3D-18 motion classification (TAMformer-aligned custom_json)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_paths = parser.add_argument_group("Paths")
    g_paths.add_argument("--source", required=True,
                         help="Dataset root (JSON splits and default frames root)")
    g_paths.add_argument("--frames-root", default=None,
                         help="Frame files root (default: same as --source)")
    g_paths.add_argument("--save-dir", default=None,
                         help=f"Checkpoints dir (default: <repo>/checkpoints/{MODEL_NAME})")
    g_paths.add_argument("--results-dir", default=None,
                         help="Directory for training_results.txt and test_results.txt "
                              "(default: folder containing r3d18.py)")
    g_paths.add_argument("--train-json", default=None)
    g_paths.add_argument("--val-json", default=None)
    g_paths.add_argument("--test-json", default=None)

    g_clip = parser.add_argument_group("Clip & Crop")
    g_clip.add_argument("-T", "--clip-len", type=int, default=10,
                        help="Window length (matches TAMformer chunk_dt / obs_length)")
    g_clip.add_argument("--chunk-stride", type=int, default=1,
                        help="Sliding-window step on each track")
    g_clip.add_argument("--crop-pad", type=float, default=0.10,
                        help="Padding around bbox crop (R3D input; TAMformer uses enlarge_ratio 1.5 @ 224)")
    g_clip.add_argument("--input-size", type=int, default=112)
    g_clip.add_argument(
        "--skip-crop-resize",
        action="store_true",
        help="Benchmark only: skip bbox crop+pad; resize full frame to --input-size "
             "(not comparable to TAMformer/R3D bbox baseline)",
    )
    g_clip.add_argument(
        "--benchmark-dummy-read",
        action="store_true",
        help="Benchmark only: read/decode each frame from disk but feed zeros (3,T,H,W) "
             "to the model; isolates I/O vs crop/resize/GPU",
    )

    g_train = parser.add_argument_group("Training")
    g_train.add_argument("--batch-size", type=int, default=64)
    g_train.add_argument("--lr", type=float, default=1e-4)
    g_train.add_argument("--weight-decay", type=float, default=0.0,
                         help="Weight decay (TAMformer Adam config has none)")
    g_train.add_argument("--epochs", type=int, default=20)
    g_train.add_argument("--patience", type=int, default=20,
                         help="Early stopping patience (>= epochs disables early stop)")
    g_train.add_argument("--use-fraction", type=float, default=1.0)
    g_train.add_argument("--num-workers", type=int, default=8)
    g_train.add_argument("--prefetch-factor", type=int, default=4,
                         help="DataLoader prefetch factor when num_workers > 0")
    g_train.add_argument("--seed", type=int, default=42)
    g_train.add_argument("--weighted-sampler", action="store_true",
                         help="WeightedRandomSampler (off matches TAMformer balance_data/apply_class_weights)")
    g_train.add_argument("--cache-size", type=int, default=50000,
                         help="Per-worker LRU frame cache size (0 disables cache)")
    g_train.add_argument("--compile", action="store_true",
                         help="Use torch.compile(model) when available (PyTorch 2+)")
    g_train.add_argument("--log-every-n-batches", type=int, default=50,
                         help="Append throughput to training_results.txt every N batches (0=off)")
    g_train.add_argument("--max-train-batches", type=int, default=0,
                         help="Stop each train epoch after N batches (0=full epoch; for benchmarks)")
    g_train.add_argument(
        "--no-dedupe-frame-reads",
        action="store_true",
        help="Old loader: each sample reads frames in workers. Default: one read per "
             "unique frame path per batch (collate), then crop all boxes.",
    )

    g_train.add_argument("--no-resume", action="store_true",
                         help="Train from scratch even if a checkpoint exists in --save-dir")

    g_filter = parser.add_argument_group("Legacy filtering (unused when align-tamformer)")
    g_filter.add_argument("--frame-keep-mod", type=int, default=1,
                        help="Kept for CLI compatibility; track sliding windows ignore stride subsampling")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _frame_index(frame_path: str) -> int:
    matches = re.findall(r"(\d+)", frame_path)
    if not matches:
        return -1
    return int(matches[-1])


def _sequence_id(frame_path: str) -> str:
    normalized = str(frame_path).replace("\\", "/")
    if "/frames/" in normalized:
        return normalized.split("/frames/")[0]
    return os.path.dirname(normalized)


def _record_drive_suffix(frame_path: str) -> Optional[str]:
    normalized = str(frame_path).replace("\\", "/")
    match = re.search(r"(RECORD[^/]+/DRIVE[^/]+/frames/.+)$", normalized)
    return match.group(1) if match else None


def resolve_frame_path(frame_path: str, frames_root: str) -> str:
    normalized = str(frame_path).replace("\\", "/")
    if os.path.isfile(normalized):
        return normalized
    if not frames_root:
        return normalized
    root = str(frames_root)
    candidates = [os.path.join(root, normalized)]
    rd_suffix = _record_drive_suffix(normalized)
    if rd_suffix:
        candidates.append(os.path.join(root, rd_suffix))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return normalized


def parse_key_path(json_key: str) -> Optional[Tuple[str, str, int, str]]:
    m = PAT.search(json_key)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), m.group(4).lower()


def local_frame_path(frames_root: str, record: str, drive: str,
                     index: int, ext: str = "png") -> str:
    return str(Path(frames_root) / record / drive / "frames" / f"{index:06d}.{ext}")


def resolved_path_for_frame(frame_path: str, frames_root: str) -> str:
    parsed = parse_key_path(frame_path)
    if parsed is not None:
        record, drive, frame_idx, ext = parsed
        return local_frame_path(frames_root, record, drive, frame_idx, ext)
    return resolve_frame_path(frame_path, frames_root)


def load_img(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def denorm_xywh(cx, cy, w, h, img_w, img_h):
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    return [x, y, w * img_w, h * img_h]


def safe_crop(img, x, y, w, h, pad: float, input_size: int):
    H, W = img.shape[:2]
    x -= w * pad
    y -= h * pad
    w *= (1 + 2 * pad)
    h *= (1 + 2 * pad)
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(W, x + w))
    y2 = int(min(H, y + h))
    if x2 <= x1 or y2 <= y1:
        return cv2.resize(img, (input_size, input_size))
    return cv2.resize(img[y1:y2, x1:x2], (input_size, input_size))


def resize_full_frame(img, input_size: int):
    """Benchmark path: one resize of the full decoded frame (no bbox crop)."""
    return cv2.resize(img, (input_size, input_size))


def normalize_clip_hwc(frames_hwc: np.ndarray) -> np.ndarray:
    """frames_hwc: (T, H, W, 3) float-ready uint8/float from crops."""
    clip = frames_hwc.astype(np.float32) / 255.0
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    clip = (clip - mean) / std
    return np.transpose(clip, (3, 0, 1, 2))


def load_unique_frames(paths: List[str]) -> Dict[str, np.ndarray]:
    """Read each path at most once."""
    cache: Dict[str, np.ndarray] = {}
    for path in paths:
        if path in cache:
            continue
        img = load_img(path)
        if img is None:
            raise FileNotFoundError(f"Frame missing at runtime: {path}")
        cache[path] = img
    return cache


def build_clip_from_frames_meta(
    frames_meta: List[Dict],
    frame_cache: Dict[str, np.ndarray],
    args,
) -> np.ndarray:
    """Return clip array (C, T, H, W)."""
    pad = args.crop_pad
    sz = args.input_size
    skip_crop = bool(getattr(args, "skip_crop_resize", False))
    crops = []
    for fm in frames_meta:
        img = frame_cache[fm["path"]]
        if skip_crop:
            crops.append(resize_full_frame(img, sz))
        else:
            x, y, w, h = fm["bbox"]
            crops.append(safe_crop(img, x, y, w, h, pad=pad, input_size=sz))
    return normalize_clip_hwc(np.stack(crops, axis=0))


class CollateStats:
    """Updated each collate call (for logging unique imreads per batch)."""

    def __init__(self):
        self.last_unique_reads = 0
        self.last_total_frame_refs = 0


def collate_clips_batch(batch: List[Dict], args, stats: Optional[CollateStats] = None):
    """
    Batch-level frame dedupe (teacher's idea): collect unique paths in the batch,
    imread each once, crop all tracks/windows from cache.
    """
    if getattr(args, "benchmark_dummy_read", False):
        unique_paths = []
        seen = set()
        total_refs = 0
        for sample in batch:
            for fm in sample["frames"]:
                total_refs += 1
                if fm["path"] not in seen:
                    seen.add(fm["path"])
                    unique_paths.append(fm["path"])
        load_unique_frames(unique_paths)
        t = int(args.clip_len)
        sz = int(args.input_size)
        clips = [np.zeros((3, t, sz, sz), dtype=np.float32) for _ in batch]
        labels = [int(s["label"]) for s in batch]
        if stats is not None:
            stats.last_unique_reads = len(unique_paths)
            stats.last_total_frame_refs = total_refs
        x = torch.from_numpy(np.stack(clips, axis=0))
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    unique_paths = []
    seen = set()
    total_refs = 0
    for sample in batch:
        for fm in sample["frames"]:
            total_refs += 1
            p = fm["path"]
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

    frame_cache = load_unique_frames(unique_paths)

    clips = [
        build_clip_from_frames_meta(sample["frames"], frame_cache, args)
        for sample in batch
    ]
    labels = [int(s["label"]) for s in batch]

    if stats is not None:
        stats.last_unique_reads = len(unique_paths)
        stats.last_total_frame_refs = total_refs

    x = torch.from_numpy(np.stack(clips, axis=0).astype(np.float32))
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def make_collate_fn(args, stats: Optional[CollateStats] = None):
    return partial(collate_clips_batch, args=args, stats=stats)


class PreventionClipsFromFrames(Dataset):
    """Track-centric sliding windows (TrackJSONAdapter chunk_dt semantics)."""

    def __init__(self, ann_path: str, frames_root: str, args, frac: float = 1.0):
        print(f"\nLoading dataset: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            frame_dict = json.load(f)

        self.frames_root = frames_root
        self.args = args
        self.cache_size = max(0, int(getattr(args, "cache_size", 0)))
        self._frame_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        window = int(args.clip_len)
        stride = max(1, int(args.chunk_stride))

        tracks: Dict[str, List[Tuple[int, str, dict]]] = {}
        for frame_path, frame_data in frame_dict.items():
            frame_idx = _frame_index(frame_path)
            sequence_id = _sequence_id(frame_path)
            for obj in frame_data.get("objs", []):
                if obj.get("type") == "Human":
                    continue
                track_id = obj.get("trackID", None)
                if track_id is None:
                    continue
                tid = "{}::{}".format(sequence_id, str(track_id))
                tracks.setdefault(tid, []).append((frame_idx, frame_path, obj))

        raw_samples = []
        skipped_short = 0
        skipped_missing = 0

        for tid, samples in tracks.items():
            samples.sort(key=lambda x: x[0])
            n = len(samples)
            if n < window:
                skipped_short += 1
                continue

            for start in range(0, n - window + 1, stride):
                win = samples[start:start + window]
                paths_ok = True
                frames_meta = []
                for _, fpath, obj in win:
                    rpath = resolved_path_for_frame(fpath, frames_root)
                    if not os.path.isfile(rpath):
                        paths_ok = False
                        break
                    img_w = int(obj.get("img_width", 1920))
                    img_h = int(obj.get("img_height", 1080))
                    cx, cy, w, h = obj["xywh"]
                    frames_meta.append({
                        "path": rpath,
                        "bbox": denorm_xywh(cx, cy, w, h, img_w, img_h),
                    })
                if not paths_ok:
                    skipped_missing += 1
                    continue

                label = motion_to_class(win[-1][2].get("motion"))
                raw_samples.append({"frames": frames_meta, "label": label})

        print(f"Tracks shorter than T={window}: {skipped_short}")
        print(f"Skipped windows (missing frames): {skipped_missing}")
        print(f"Raw samples: {len(raw_samples)}")

        if 0 < frac < 1.0:
            random.shuffle(raw_samples)
            raw_samples = raw_samples[:max(1, int(len(raw_samples) * frac))]

        self.samples = raw_samples
        label_dist = Counter(s["label"] for s in self.samples)
        print(f"Final dataset size: {len(self.samples)} samples")
        print(f"Class distribution (ids): {dict(sorted(label_dist.items()))}")
        if getattr(self.args, "benchmark_dummy_read", False):
            print(
                "WARNING: --benchmark-dummy-read enabled (read frames, model gets zeros; "
                "throughput benchmark only)."
            )
        elif getattr(self.args, "skip_crop_resize", False):
            print(
                "WARNING: --skip-crop-resize enabled (full-frame resize only; "
                "for throughput benchmarks, not fair R3D/TAMformer comparison)."
            )
        elif not getattr(self.args, "no_dedupe_frame_reads", False):
            print(
                "Frame dedupe enabled: each batch reads every unique frame path once, "
                "then crops all boxes (collate_fn)."
            )

    def __len__(self):
        return len(self.samples)

    def _get_frame(self, path: str):
        if self.cache_size <= 0:
            return load_img(path)
        cached = self._frame_cache.get(path)
        if cached is not None:
            self._frame_cache.move_to_end(path, last=True)
            return cached
        img = load_img(path)
        if img is None:
            return None
        self._frame_cache[path] = img
        if len(self._frame_cache) > self.cache_size:
            self._frame_cache.popitem(last=False)
        return img

    def _dummy_clip_tensor(self, label: int):
        """Zeros with correct (C, T, H, W) for R3D-18; labels unchanged (benchmark only)."""
        t = int(self.args.clip_len)
        sz = int(self.args.input_size)
        clip = np.zeros((3, t, sz, sz), dtype=np.float32)
        return torch.from_numpy(clip), label

    def __getitem__(self, idx):
        s = self.samples[idx]
        if not getattr(self.args, "no_dedupe_frame_reads", False):
            return s

        label = s["label"]

        if getattr(self.args, "benchmark_dummy_read", False):
            for fm in s["frames"]:
                img = self._get_frame(fm["path"])
                if img is None:
                    raise FileNotFoundError(f"Frame missing at runtime: {fm['path']}")
            return self._dummy_clip_tensor(label)

        pad = self.args.crop_pad
        sz = self.args.input_size
        skip_crop = bool(getattr(self.args, "skip_crop_resize", False))
        frames = []
        for fm in s["frames"]:
            img = self._get_frame(fm["path"])
            if img is None:
                raise FileNotFoundError(f"Frame missing at runtime: {fm['path']}")
            if skip_crop:
                frames.append(resize_full_frame(img, sz))
            else:
                x, y, w, h = fm["bbox"]
                frames.append(safe_crop(img, x, y, w, h, pad=pad, input_size=sz))

        return torch.from_numpy(normalize_clip_hwc(np.stack(frames, axis=0))), label


def make_model(num_classes: int = NUM_MOTION_CLASSES) -> nn.Module:
    print(f"Creating R3D-18 model for {num_classes} classes…")
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_train_sampler(train_set: PreventionClipsFromFrames):
    labels = [s["label"] for s in train_set.samples]
    cnt = Counter(labels)
    total = sum(cnt.values())
    class_w = {c: total / max(1, n) for c, n in cnt.items()}
    weights = torch.DoubleTensor([class_w[l] for l in labels])
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def make_scaler(device: str):
    try:
        return torch.amp.GradScaler(device_type=device, enabled=(device == "cuda"))
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


def autocast_ctx(device: str, enabled: bool = True):
    try:
        return torch.amp.autocast(
            device_type=device, enabled=enabled and (device == "cuda")
        )
    except TypeError:
        return torch.cuda.amp.autocast(enabled=enabled and (device == "cuda"))


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    epoch=1,
    log_file=None,
    log_every_n_batches=0,
    max_train_batches=0,
    collate_stats: Optional[CollateStats] = None,
):
    model.train()
    ce = nn.CrossEntropyLoss()
    scaler = make_scaler(device)
    total = 0.0
    n_samples = 0
    n_batches_total = len(loader)
    n_batches_target = (
        min(n_batches_total, max_train_batches) if max_train_batches > 0 else n_batches_total
    )
    start = time.perf_counter()
    interval_start = start
    interval_samples = 0
    batch_idx = 0

    pbar = tqdm(loader, leave=False, desc="Training", total=n_batches_target)
    for batch_idx, (x, y) in enumerate(pbar, start=1):
        if max_train_batches > 0 and batch_idx > max_train_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            loss = ce(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        batch_n = x.size(0)
        total += loss.item() * batch_n
        n_samples += batch_n
        interval_samples += batch_n
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (
            log_file
            and log_every_n_batches > 0
            and batch_idx % log_every_n_batches == 0
        ):
            interval_elapsed = max(1e-9, time.perf_counter() - interval_start)
            sec_per_batch = interval_elapsed / log_every_n_batches
            samp_sec = interval_samples / interval_elapsed
            line = "epoch={} step={}/{} sec/batch={:.2f} samp/sec={:.2f} loss={:.4f}".format(
                epoch,
                batch_idx,
                n_batches_target,
                sec_per_batch,
                samp_sec,
                loss.item(),
            )
            if collate_stats is not None:
                line += " unique_frames={}/{}".format(
                    collate_stats.last_unique_reads,
                    collate_stats.last_total_frame_refs,
                )
            append_progress_line(log_file, line)
            interval_start = time.perf_counter()
            interval_samples = 0

    elapsed = max(1e-9, time.perf_counter() - start)
    if batch_idx == 0:
        return 0.0, 0.0, 0.0

    n_batches_done = batch_idx
    sec_per_batch = elapsed / n_batches_done
    samples_per_sec = n_samples / elapsed
    avg_loss = total / max(1, n_samples)

    if log_file and log_every_n_batches > 0:
        append_progress_line(
            log_file,
            "epoch={} step={}/{} END-TRAIN sec/batch={:.2f} samp/sec={:.2f} avg_loss={:.4f}".format(
                epoch,
                n_batches_done,
                n_batches_target,
                sec_per_batch,
                samples_per_sec,
                avg_loss,
            ),
        )

    return avg_loss, sec_per_batch, samples_per_sec


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, num_classes: int = NUM_MOTION_CLASSES, desc: str = "Evaluating"):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False, desc=desc):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx(device):
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * x.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_scores.append(probs)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())

    if not y_true:
        return 0.0, {}, 0.0

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_scores = np.vstack(y_scores)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        y_true_oh = np.eye(num_classes, dtype=np.float32)[y_true]
        auc_macro = roc_auc_score(y_true_oh, y_scores, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = 0.0

    metrics = {
        "acc": acc,
        "auc_macro_ovr": auc_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }
    avg_loss = total_loss / max(1, len(loader.dataset))
    return acc, metrics, avg_loss


def motion_metrics_line(metrics: dict, header: str = "motion") -> str:
    return format_motion_metrics_line(
        header,
        metrics["acc"],
        metrics["auc_macro_ovr"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
        metrics["precision_macro"],
        metrics["recall_macro"],
    )


def print_motion_metrics(metrics: dict, header: str = "motion"):
    print(motion_metrics_line(metrics, header=header))


def sklearn_classification_report(y_true, y_pred, num_classes: int = NUM_MOTION_CLASSES) -> str:
    labels = list(range(num_classes))
    target_names = [CLASS_ID_TO_NAME.get(i, str(i)) for i in labels]
    return classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        zero_division=0, digits=3,
    )


def print_per_class_report(y_true, y_pred, num_classes: int = NUM_MOTION_CLASSES):
    print(sklearn_classification_report(y_true, y_pred, num_classes))


def default_save_dir(model_name: str = MODEL_NAME) -> str:
    return str(CODE_DIR / "checkpoints" / model_name)


def find_resume_checkpoint(save_dir: str, best_path: str, no_resume: bool) -> Optional[str]:
    if no_resume:
        return None
    latest_path = os.path.join(save_dir, "checkpoint.pt")
    if os.path.isfile(latest_path):
        return latest_path
    if os.path.isfile(best_path):
        return best_path
    return None


def load_training_checkpoint(path: str, model: nn.Module, device: str) -> Dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


def save_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    no_improve: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "no_improve": no_improve,
        },
        path,
    )


def main():
    cv2.setNumThreads(0)
    args = parse_args()
    set_seed(args.seed)

    if args.benchmark_dummy_read and args.skip_crop_resize:
        print(
            "Note: --benchmark-dummy-read ignores --skip-crop-resize "
            "(frames are read then discarded)."
        )

    source = args.source
    frames_root = args.frames_root or source
    save_dir = args.save_dir or default_save_dir()
    train_json = args.train_json or os.path.join(source, "Train.json")
    val_json = args.val_json or os.path.join(source, "Validation.json")
    test_json = args.test_json or os.path.join(source, "Test.json")

    os.makedirs(save_dir, exist_ok=True)
    if args.results_dir:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = str(CODE_DIR)
    train_log_file = os.path.abspath(training_log_path(results_dir))
    test_log_file = os.path.abspath(test_log_path(results_dir))
    best_model_path = os.path.join(save_dir, f"best_{MODEL_NAME}.pt")
    latest_checkpoint_path = os.path.join(save_dir, "checkpoint.pt")

    assert Path(source).exists(), f"--source not found: {source}"
    for p in [train_json, val_json, test_json]:
        assert Path(p).exists(), f"Missing JSON: {p}"

    print("=" * 60)
    print("R3D-18 Motion Classification (TAMformer-aligned)")
    print(f"  source       : {source}")
    print(f"  frames_root  : {frames_root}")
    print(f"  save_dir     : {save_dir}")
    print(f"  clip_len T   : {args.clip_len}")
    print(f"  chunk_stride : {args.chunk_stride}")
    print(f"  skip_crop    : {args.skip_crop_resize}")
    print(f"  dummy_read   : {args.benchmark_dummy_read}")
    print(f"  dedupe_reads : {not args.no_dedupe_frame_reads}")
    print(f"  num_classes  : {NUM_MOTION_CLASSES}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  lr           : {args.lr}")
    print(f"  epochs       : {args.epochs}")
    print(f"  patience     : {args.patience}")
    print(f"  prefetch     : {args.prefetch_factor}")
    print(f"  cache_size   : {args.cache_size}")
    print(f"  compile      : {args.compile}")
    print(f"  results_dir  : {results_dir}")
    print(f"  training_log : {train_log_file}")
    print(f"  test_log     : {test_log_file}")
    print("=" * 60)

    write_run_header(
        train_log_file,
        "R3D-18 training (TAMformer-aligned)",
        [
            f"source: {source}",
            f"frames_root: {frames_root}",
            f"train_json: {train_json}",
            f"val_json: {val_json}",
            f"test_json: {test_json}",
            f"clip_len: {args.clip_len}",
            f"chunk_stride: {args.chunk_stride}",
            f"skip_crop_resize: {args.skip_crop_resize}",
            f"benchmark_dummy_read: {args.benchmark_dummy_read}",
            f"dedupe_frame_reads: {not args.no_dedupe_frame_reads}",
            f"batch_size: {args.batch_size}",
            f"lr: {args.lr}",
            f"epochs: {args.epochs}",
            f"patience: {args.patience}",
            f"seed: {args.seed}",
            f"prefetch_factor: {args.prefetch_factor}",
            f"cache_size: {args.cache_size}",
            f"compile: {args.compile}",
            f"log_every_n_batches: {args.log_every_n_batches}",
            f"max_train_batches: {args.max_train_batches}",
            f"device: pending",
        ],
    )
    print("Training log (created now):", train_log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    append_epoch_line(train_log_file, f"device: {device}")

    append_progress_line(train_log_file, "building train...")
    train_set = PreventionClipsFromFrames(train_json, frames_root, args, args.use_fraction)
    append_progress_line(train_log_file, "train samples: {}".format(len(train_set)))

    append_progress_line(train_log_file, "building val...")
    val_set = PreventionClipsFromFrames(val_json, frames_root, args, args.use_fraction)
    append_progress_line(train_log_file, "val samples: {}".format(len(val_set)))

    append_progress_line(train_log_file, "building test...")
    test_set = PreventionClipsFromFrames(test_json, frames_root, args, args.use_fraction)
    append_progress_line(train_log_file, "test samples: {}".format(len(test_set)))

    append_epoch_line(
        train_log_file,
        "samples train/val/test: {}/{}/{}".format(
            len(train_set), len(val_set), len(test_set),
        ),
    )
    steps_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    if args.max_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
    append_progress_line(
        train_log_file,
        "train steps/epoch (approx): {} (batch_size={})".format(
            steps_per_epoch, args.batch_size,
        ),
    )
    append_epoch_line(train_log_file, "-" * 72)
    append_epoch_line(
        train_log_file,
        "{:>5}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}  {}".format(
            "epoch", "train_loss", "val_loss", "val_acc", "sec/batch", "samp/sec", "notes",
        ),
    )

    collate_stats = CollateStats()
    if args.no_dedupe_frame_reads:
        collate_fn = None
        print("Loader: per-sample frame reads in workers (--no-dedupe-frame-reads).")
    else:
        collate_fn = make_collate_fn(args, collate_stats)
        print(
            "Loader: batch frame dedupe (one imread per unique path per batch, "
            "then crop all boxes)."
        )

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = max(2, int(args.prefetch_factor))

    if args.weighted_sampler:
        sampler = build_train_sampler(train_set)
        train_loader = DataLoader(train_set, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_kw)

    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)

    model = make_model().to(device)
    resume_path = find_resume_checkpoint(save_dir, best_model_path, args.no_resume)
    resume_ckpt = None
    if resume_path:
        resume_ckpt = load_training_checkpoint(resume_path, model, device)
        print(f"Resuming from checkpoint: {resume_path}")
        print(
            f"  last epoch   : {resume_ckpt['epoch']}/{args.epochs}"
            f"  best_val_loss: {resume_ckpt.get('best_val_loss', float('inf')):.4f}"
        )
        append_epoch_line(
            train_log_file,
            "resume: {} (epoch {}, best_val_loss={:.6f})".format(
                resume_path,
                resume_ckpt["epoch"],
                resume_ckpt.get("best_val_loss", float("inf")),
            ),
        )

    if args.compile:
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("torch.compile enabled.")
                append_epoch_line(train_log_file, "torch.compile: enabled")
            except Exception as exc:
                print("torch.compile failed, continuing without compile:", exc)
                append_epoch_line(train_log_file, f"torch.compile: failed ({exc})")
        else:
            print("torch.compile not available in this PyTorch version.")
            append_epoch_line(train_log_file, "torch.compile: unavailable")
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if resume_ckpt is not None:
        opt.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_loss = resume_ckpt.get("best_val_loss", float("inf"))
        no_improve = resume_ckpt.get("no_improve", 0)
    else:
        start_epoch = 1
        best_loss, no_improve = float("inf"), 0

    ce = nn.CrossEntropyLoss()

    if start_epoch > args.epochs:
        print(
            f"\nCheckpoint already at epoch {start_epoch - 1}/{args.epochs} — skipping training."
        )
        append_epoch_line(
            train_log_file,
            "training skipped: checkpoint epoch {} >= target epochs {}".format(
                start_epoch - 1, args.epochs,
            ),
        )
    else:
        print(f"\nStarting training from epoch {start_epoch}/{args.epochs}…")
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss, sec_per_batch, samples_per_sec = train_one_epoch(
                model,
                train_loader,
                opt,
                device,
                epoch=epoch,
                log_file=train_log_file,
                log_every_n_batches=args.log_every_n_batches,
                max_train_batches=args.max_train_batches,
                collate_stats=collate_stats if not args.no_dedupe_frame_reads else None,
            )
            val_acc, val_metrics, val_loss = evaluate(
                model,
                val_loader,
                device,
                criterion=ce,
                desc="Validating",
            )
            print(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.3f}")
            print_motion_metrics(val_metrics, header="val motion")

            notes = ""
            if val_loss < best_loss:
                best_loss, no_improve = val_loss, 0
                save_training_checkpoint(
                    best_model_path, model, opt, epoch, best_loss, no_improve,
                )
                notes = "saved best"
                print(f"New best model saved (val_loss={best_loss:.4f})")
            else:
                no_improve += 1
                notes = "patience {}/{}".format(no_improve, args.patience)
                print(f"No improvement - patience {no_improve}/{args.patience}")
                if args.patience < args.epochs and no_improve >= args.patience:
                    print("Early stopping triggered.")
                    notes = "early stop"
                    save_training_checkpoint(
                        latest_checkpoint_path, model, opt, epoch, best_loss, no_improve,
                    )
                    append_epoch_line(
                        train_log_file,
                        "{:>5}  {:>12.6f}  {:>12.6f}  {:>10.4f}  {:>10.4f}  {:>10.2f}  {}".format(
                            epoch, train_loss, val_loss, val_acc, sec_per_batch, samples_per_sec, notes,
                        ),
                    )
                    append_epoch_line(train_log_file, motion_metrics_line(val_metrics, "val motion"))
                    break

            save_training_checkpoint(
                latest_checkpoint_path, model, opt, epoch, best_loss, no_improve,
            )
            append_epoch_line(
                train_log_file,
                "{:>5}  {:>12.6f}  {:>12.6f}  {:>10.4f}  {:>10.4f}  {:>10.2f}  {}".format(
                    epoch, train_loss, val_loss, val_acc, sec_per_batch, samples_per_sec, notes,
                ),
            )
            append_epoch_line(train_log_file, motion_metrics_line(val_metrics, "val motion"))

    print("\nTraining done — loading best model for test…")
    assert os.path.isfile(best_model_path), (
        f"No best checkpoint found at {best_model_path}. "
        "Train at least one epoch or pass --no-resume to start fresh."
    )
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc, test_metrics, _ = evaluate(model, test_loader, device, desc="Testing")

    print(f"\n{'=' * 60}")
    print("TEST (compare to TAMformer motion head):")
    print_motion_metrics(test_metrics, header="motion")
    print("\nPer-class classification report:")
    print_per_class_report(test_metrics["y_true"], test_metrics["y_pred"])

    test_motion_line = motion_metrics_line(test_metrics, header="motion")
    test_per_class = format_per_class_metrics(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        test_metrics["y_scores"],
        NUM_MOTION_CLASSES,
        CLASS_ID_TO_NAME,
    )
    sk_report = sklearn_classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
    )
    write_test_results(
        test_log_file,
        [
            f"source: {source}",
            f"frames_root: {frames_root}",
            f"test_json: {test_json}",
            f"best_checkpoint: {best_model_path}",
            f"best_val_loss: {best_loss}",
            test_motion_line,
            test_per_class,
            "Sklearn classification report:",
            sk_report,
        ],
    )

    print(f"Best model saved to: {best_model_path}")
    print("Training log written to:", train_log_file)
    print("Test results written to:", test_log_file)


if __name__ == "__main__":
    main()
