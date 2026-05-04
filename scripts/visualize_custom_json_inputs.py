import argparse
import os
import random
import re

import cv2
import matplotlib.pyplot as plt


def _frame_index(frame_path):
    matches = re.findall(r"(\d+)", str(frame_path))
    return int(matches[-1]) if matches else -1


def _normalize(path_str):
    return str(path_str).replace("\\", os.sep).replace("/", os.sep)


def _resolve_frame_path(frame_path, dataset_root=None):
    normalized = _normalize(frame_path)
    candidates = [normalized]
    if dataset_root:
        candidates.append(os.path.join(dataset_root, normalized))
        # Common case: JSON stores ".../frames/..." while root points above that.
        rel = normalized.split("frames" + os.sep, 1)[-1] if ("frames" + os.sep) in normalized else normalized
        candidates.append(os.path.join(dataset_root, "frames", rel))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _build_tracks(json_path):
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        frame_dict = json.load(f)

    tracks = {}
    for frame_path, frame_data in frame_dict.items():
        for obj in frame_data.get("objs", []):
            tid = obj.get("trackID")
            xywh = obj.get("xywh")
            if tid is None or xywh is None or len(xywh) != 4:
                continue
            tid = str(tid)
            tracks.setdefault(tid, [])
            tracks[tid].append((frame_path, obj))

    for tid in tracks:
        tracks[tid].sort(key=lambda x: _frame_index(x[0]))
    return tracks


def _xywh_to_xyxy(xywh, img_w, img_h):
    cx = float(xywh[0]) * img_w
    cy = float(xywh[1]) * img_h
    bw = float(xywh[2]) * img_w
    bh = float(xywh[3]) * img_h
    x1 = max(0, int(round(cx - bw / 2.0)))
    y1 = max(0, int(round(cy - bh / 2.0)))
    x2 = min(img_w - 1, int(round(cx + bw / 2.0)))
    y2 = min(img_h - 1, int(round(cy + bh / 2.0)))
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(description="Visual sanity-check for custom_json TAMformer inputs.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to frame-keyed JSON.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Optional root folder to resolve frame paths.")
    parser.add_argument("--track_id", type=str, default=None, help="Track ID to inspect (default: random).")
    parser.add_argument("--num_frames", type=int, default=6, help="How many timesteps to render.")
    parser.add_argument("--save_path", type=str, default="input_sanity.png", help="Output image path.")
    args = parser.parse_args()

    tracks = _build_tracks(args.json_path)
    if not tracks:
        raise RuntimeError("No valid tracks found in JSON.")

    if args.track_id and args.track_id in tracks:
        tid = args.track_id
    else:
        tid = random.choice(list(tracks.keys()))

    entries = tracks[tid]
    if not entries:
        raise RuntimeError(f"Track {tid} has no entries.")

    k = min(max(1, args.num_frames), len(entries))
    sampled = entries[-k:]

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]

    rendered = 0
    for ax, (frame_path, obj) in zip(axes, sampled):
        resolved = _resolve_frame_path(frame_path, args.dataset_root)
        if not resolved:
            ax.set_title("missing frame")
            ax.axis("off")
            continue

        img = cv2.imread(resolved)
        if img is None:
            ax.set_title("unreadable frame")
            ax.axis("off")
            continue

        h, w = img.shape[:2]
        x1, y1, x2, y2 = _xywh_to_xyxy(obj["xywh"], w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        speed = float(obj.get("Vx", 0.0))
        motion = str(obj.get("motion", "unknown"))
        label = f"id={tid} motion={motion} speed={speed:.2f}"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(os.path.basename(_normalize(frame_path)))
        ax.axis("off")
        rendered += 1

    fig.suptitle(f"Track {tid} | rendered {rendered}/{k} frames", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.save_path, dpi=140)
    print(f"Saved visualization: {args.save_path}")


if __name__ == "__main__":
    main()
