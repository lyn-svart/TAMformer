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
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter
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

PAT = re.compile(
    r"(?:^|[\\/])(RECORD[^\\/]+)[\\/](DRIVE[^\\/]+)[\\/]frames[\\/](\d+)\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)

torch.backends.cudnn.benchmark = True


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
                         help="Checkpoints dir (default: <source>/../checkpoints)")
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
    g_train.add_argument("--seed", type=int, default=42)
    g_train.add_argument("--weighted-sampler", action="store_true",
                         help="WeightedRandomSampler (off matches TAMformer balance_data/apply_class_weights)")

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


class PreventionClipsFromFrames(Dataset):
    """Track-centric sliding windows (TrackJSONAdapter chunk_dt semantics)."""

    def __init__(self, ann_path: str, frames_root: str, args, frac: float = 1.0):
        print(f"\nLoading dataset: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            frame_dict = json.load(f)

        self.frames_root = frames_root
        self.args = args
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pad = self.args.crop_pad
        sz = self.args.input_size
        frames = []
        for fm in s["frames"]:
            img = load_img(fm["path"])
            if img is None:
                raise FileNotFoundError(f"Frame missing at runtime: {fm['path']}")
            x, y, w, h = fm["bbox"]
            frames.append(safe_crop(img, x, y, w, h, pad=pad, input_size=sz))

        clip = np.stack(frames, axis=0).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
        clip = (clip - mean) / std
        clip = np.transpose(clip, (3, 0, 1, 2))
        return torch.from_numpy(clip), s["label"]


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


def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    scaler = make_scaler(device)
    total = 0.0

    pbar = tqdm(loader, leave=False, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            loss = ce(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = NUM_MOTION_CLASSES):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    for x, y in tqdm(loader, leave=False, desc="Evaluating"):
        with autocast_ctx(device):
            logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_scores.append(probs)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())

    if not y_true:
        return 0.0, {}

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
    return acc, metrics


def print_motion_metrics(metrics: dict, header: str = "motion"):
    print(
        f"{header} acc:", metrics["acc"],
        "- auc_macro_ovr:", metrics["auc_macro_ovr"],
        "- f1_macro:", metrics["f1_macro"],
        "- f1_weighted:", metrics["f1_weighted"],
        "- precision_macro:", metrics["precision_macro"],
        "- recall_macro:", metrics["recall_macro"],
    )


def print_per_class_report(y_true, y_pred, num_classes: int = NUM_MOTION_CLASSES):
    labels = list(range(num_classes))
    target_names = [CLASS_ID_TO_NAME.get(i, str(i)) for i in labels]
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            zero_division=0, digits=3,
        )
    )


def main():
    cv2.setNumThreads(0)
    args = parse_args()
    set_seed(args.seed)

    source = args.source
    frames_root = args.frames_root or source
    save_dir = args.save_dir or str(Path(source).parent / "checkpoints")
    train_json = args.train_json or os.path.join(source, "Train.json")
    val_json = args.val_json or os.path.join(source, "Validation.json")
    test_json = args.test_json or os.path.join(source, "Test.json")

    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_r3d18.pt")

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
    print(f"  num_classes  : {NUM_MOTION_CLASSES}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  lr           : {args.lr}")
    print(f"  epochs       : {args.epochs}")
    print(f"  patience     : {args.patience}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_set = PreventionClipsFromFrames(train_json, frames_root, args, args.use_fraction)
    val_set = PreventionClipsFromFrames(val_json, frames_root, args, args.use_fraction)
    test_set = PreventionClipsFromFrames(test_json, frames_root, args, args.use_fraction)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.weighted_sampler:
        sampler = build_train_sampler(train_set)
        train_loader = DataLoader(train_set, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_kw)

    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)

    model = make_model().to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_loss, no_improve = float("inf"), 0
    ce = nn.CrossEntropyLoss()

    print("\nStarting training…")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, opt, device)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += ce(model(x), y).item() * x.size(0)
        val_loss /= max(1, len(val_set))

        val_acc, val_metrics = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.3f}")
        print_motion_metrics(val_metrics, header="val motion")

        if val_loss < best_loss:
            best_loss, no_improve = val_loss, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_loss,
                },
                best_model_path,
            )
            print(f"New best model saved (val_loss={best_loss:.4f})")
        else:
            no_improve += 1
            print(f"No improvement - patience {no_improve}/{args.patience}")
            if args.patience < args.epochs and no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    print("\nTraining done — loading best model for test…")
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc, test_metrics = evaluate(model, test_loader, device)

    print(f"\n{'=' * 60}")
    print("TEST (compare to TAMformer motion head):")
    print_motion_metrics(test_metrics, header="motion")
    print("\nPer-class classification report:")
    print_per_class_report(test_metrics["y_true"], test_metrics["y_pred"])
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
