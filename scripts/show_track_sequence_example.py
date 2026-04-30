import argparse
import json
import re
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_generator import TrackJSONAdapter, DataGetter


def frame_index(frame_path):
    matches = re.findall(r"(\d+)", frame_path)
    if not matches:
        return -1
    return int(matches[-1])


def build_raw_tracks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        frame_dict = json.load(f)

    tracks = {}
    for frame_path, frame_data in frame_dict.items():
        for obj in frame_data.get("objs", []):
            tid = obj.get("trackID")
            if tid is None:
                continue
            tid = str(tid)
            tracks.setdefault(tid, [])
            tracks[tid].append((frame_index(frame_path), frame_path, obj))

    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])
    return tracks


def print_track_example(track_id, entries, max_frames=8):
    print("\n=== RAW TRACK EXAMPLE ===")
    print("trackID:", track_id)
    print("track_length:", len(entries))
    print("showing first", min(max_frames, len(entries)), "frames")
    print("-" * 90)
    for i, (_, frame_path, obj) in enumerate(entries[:max_frames]):
        motion = obj.get("motion", "N/A")
        xywh = obj.get("xywh", [])
        vx = obj.get("Vx", "N/A")
        vz = obj.get("Vz", "N/A")
        print(f"[{i:02d}] frame={frame_path}")
        print(f"     motion={motion}, xywh={xywh}, Vx={vx}, Vz={vz}")


def main():
    parser = argparse.ArgumentParser(description="Show track-sequence examples from custom JSON.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to frame-keyed JSON file.")
    parser.add_argument("--track_id", type=str, default=None, help="Specific trackID to inspect.")
    parser.add_argument("--obs_length", type=int, default=30, help="Observation length for padded window example.")
    parser.add_argument("--max_frames", type=int, default=8, help="How many raw frames to print.")
    args = parser.parse_args()

    raw_tracks = build_raw_tracks(args.json_path)
    if not raw_tracks:
        print("No tracks found.")
        return

    if args.track_id is not None and args.track_id in raw_tracks:
        selected_id = args.track_id
    else:
        # deterministic choice: longest track
        selected_id = max(raw_tracks.keys(), key=lambda tid: len(raw_tracks[tid]))

    print_track_example(selected_id, raw_tracks[selected_id], max_frames=args.max_frames)

    adapter = TrackJSONAdapter(args.json_path)
    data_raw = adapter.load()

    model_opts = {
        "generator": False,
        "process": True,
        "dataset": "custom_json",
        "obs_input_type": ["box", "speed"],
        "batch_size": 32,
        "val_batch_size": 32,
        "balance_data": False,
        "obs_seconds": 1,
        "interval": args.obs_length,
        "fstride": 1,
    }
    # Force exact obs_length for demonstration by setting interval to obs_length and obs_seconds=1.
    getter = DataGetter("train", data_raw, model_opts)
    seq_data, _ = getter.get_data_sequence()

    idx = None
    for i, pid_seq in enumerate(seq_data["ped_id"]):
        if str(pid_seq[-1][0]) == str(selected_id):
            idx = i
            break
    if idx is None:
        idx = 0

    print("\n=== PROCESSED MODEL INPUT EXAMPLE ===")
    print("selected trackID:", seq_data["ped_id"][idx][-1][0])
    print("obs_length used:", seq_data["box"].shape[1])
    print("box tensor shape for one sample:", seq_data["box"][idx].shape)
    print("speed tensor shape for one sample:", seq_data["speed"][idx].shape)
    print("activity tensor shape for one sample:", seq_data["crossing"][idx].shape)
    print("final class label:", int(seq_data["labels"][idx]))
    print("\nLast 5 timesteps [x1, y1, x2, y2] + speed + class:")

    start = max(0, seq_data["box"].shape[1] - 5)
    for t in range(start, seq_data["box"].shape[1]):
        b = seq_data["box"][idx][t]
        s = seq_data["speed"][idx][t][0]
        c = int(seq_data["crossing"][idx][t][0])
        print(f"t={t:02d} box={[round(float(x), 2) for x in b]} speed={float(s):.4f} class={c}")


if __name__ == "__main__":
    main()
