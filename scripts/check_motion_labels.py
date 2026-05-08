import argparse
import ast
import json
import os
from collections import Counter


def _load_motion_map(repo_root):
    """Read TrackJSONAdapter.MOTION_TO_CLASS from data_generator.py without importing tensorflow."""
    data_gen_path = os.path.join(repo_root, "data_generator.py")
    with open(data_gen_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=data_gen_path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrackJSONAdapter":
            for child in node.body:
                if not isinstance(child, ast.Assign):
                    continue
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == "MOTION_TO_CLASS":
                        motion_map = ast.literal_eval(child.value)
                        if not isinstance(motion_map, dict):
                            raise RuntimeError("MOTION_TO_CLASS is not a dict.")
                        return motion_map
    raise RuntimeError("Could not find TrackJSONAdapter.MOTION_TO_CLASS in data_generator.py.")


def _scan_json(json_path, motion_map):
    with open(json_path, "r", encoding="utf-8") as f:
        frame_dict = json.load(f)

    expected = set(motion_map.keys())
    normalized_counter = Counter()
    raw_counter = Counter()
    unknown_counter = Counter()
    missing_motion = 0
    non_human_objects = 0

    for frame_data in frame_dict.values():
        objs = frame_data.get("objs", [])
        for obj in objs:
            if obj.get("type") == "Human":
                continue
            non_human_objects += 1
            motion_raw = obj.get("motion", None)
            if motion_raw is None:
                missing_motion += 1
                unknown_counter["<missing>"] += 1
                continue
            motion_norm = str(motion_raw).strip().lower()
            raw_counter[str(motion_raw)] += 1
            normalized_counter[motion_norm] += 1
            if motion_norm not in expected:
                unknown_counter[motion_norm] += 1

    return {
        "non_human_objects": non_human_objects,
        "missing_motion": missing_motion,
        "raw_counter": raw_counter,
        "normalized_counter": normalized_counter,
        "unknown_counter": unknown_counter,
    }


def _print_split_report(name, stats, motion_map, top_k):
    unknown_total = sum(stats["unknown_counter"].values())
    known_total = stats["non_human_objects"] - unknown_total
    print("\n=== {} ===".format(name))
    print("non-human objects: {}".format(stats["non_human_objects"]))
    print("known motions: {}".format(known_total))
    print("unknown/missing motions: {}".format(unknown_total))
    print("missing motion field: {}".format(stats["missing_motion"]))

    if unknown_total > 0:
        print("\nUnknown labels (normalized) -> count")
        for label, count in stats["unknown_counter"].most_common():
            print("  {} -> {}".format(label, count))

    print("\nKnown class distribution (normalized) -> class_id -> count")
    for motion, count in stats["normalized_counter"].most_common():
        class_id = motion_map.get(motion, "UNKNOWN")
        print("  {} -> {} -> {}".format(motion, class_id, count))

    print("\nTop raw label variants (helps detect typos/casing)")
    for raw, count in stats["raw_counter"].most_common(top_k):
        print("  {} -> {}".format(raw, count))


def main():
    parser = argparse.ArgumentParser(
        description="Check custom_json motion labels against TrackJSONAdapter.MOTION_TO_CLASS."
    )
    parser.add_argument("--json_path", type=str, default=None, help="Single JSON file to inspect.")
    parser.add_argument("--train_json", type=str, default=None, help="Train split JSON file.")
    parser.add_argument("--val_json", type=str, default=None, help="Validation split JSON file.")
    parser.add_argument("--test_json", type=str, default=None, help="Test split JSON file.")
    parser.add_argument("--top_k_raw", type=int, default=25, help="Show top-k raw motion strings.")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    motion_map = _load_motion_map(repo_root)
    print("Loaded {} motion classes from data_generator.py".format(len(motion_map)))

    jobs = []
    if args.json_path:
        jobs.append(("json", args.json_path))
    if args.train_json:
        jobs.append(("train", args.train_json))
    if args.val_json:
        jobs.append(("val", args.val_json))
    if args.test_json:
        jobs.append(("test", args.test_json))

    if not jobs:
        raise SystemExit("Provide --json_path or one/more of --train_json --val_json --test_json.")

    for split_name, path in jobs:
        if not os.path.isfile(path):
            print("\n=== {} ===".format(split_name))
            print("File not found: {}".format(path))
            continue
        stats = _scan_json(path, motion_map)
        _print_split_report(split_name, stats, motion_map, args.top_k_raw)


if __name__ == "__main__":
    main()
