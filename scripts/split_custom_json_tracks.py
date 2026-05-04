import argparse
import json
import os
import random
from collections import defaultdict


def _collect_track_to_frames(frame_dict):
    track_to_frames = defaultdict(set)
    for frame_path, frame_data in frame_dict.items():
        for obj in frame_data.get("objs", []):
            tid = obj.get("trackID")
            if tid is None:
                continue
            track_to_frames[str(tid)].add(frame_path)
    return track_to_frames


def _build_split_ids(track_ids, train_ratio, val_ratio, test_ratio, seed):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    ids = list(track_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:n_train + n_val + n_test])
    return train_ids, val_ids, test_ids


def _filter_frame_dict(frame_dict, allowed_track_ids):
    out = {}
    for frame_path, frame_data in frame_dict.items():
        objs = frame_data.get("objs", [])
        filtered = [o for o in objs if str(o.get("trackID")) in allowed_track_ids]
        if filtered:
            copied = dict(frame_data)
            copied["objs"] = filtered
            out[frame_path] = copied
    return out


def _count_unique_tracks(frame_dict):
    ids = set()
    for frame_data in frame_dict.values():
        for obj in frame_data.get("objs", []):
            tid = obj.get("trackID")
            if tid is not None:
                ids.add(str(tid))
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description="Split custom frame-keyed JSON into track-disjoint train/val/test.")
    parser.add_argument("--input_json", required=True, type=str, help="Path to source JSON.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save split JSON files.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Track ratio for train split.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Track ratio for val split.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Track ratio for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        frame_dict = json.load(f)

    track_to_frames = _collect_track_to_frames(frame_dict)
    all_track_ids = sorted(track_to_frames.keys())
    if not all_track_ids:
        raise RuntimeError("No trackID found in input_json.")

    train_ids, val_ids, test_ids = _build_split_ids(
        all_track_ids, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    train_json = _filter_frame_dict(frame_dict, train_ids)
    val_json = _filter_frame_dict(frame_dict, val_ids)
    test_json = _filter_frame_dict(frame_dict, test_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    test_path = os.path.join(args.output_dir, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_json, f)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f)

    print("Split done.")
    print(f"tracks train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(
        "frames train/val/test: "
        f"{len(train_json)}/{len(val_json)}/{len(test_json)}"
    )
    print(
        "unique tracks train/val/test (sanity): "
        f"{_count_unique_tracks(train_json)}/{_count_unique_tracks(val_json)}/{_count_unique_tracks(test_json)}"
    )
    print("Saved:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)


if __name__ == "__main__":
    main()
