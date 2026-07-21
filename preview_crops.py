"""Preview annotation boxes and context crops without training or inference."""

import argparse
import os

import cv2
import numpy as np
import yaml

from car_motion_data import CarMotion
from motion_labels import CLASS_ID_TO_NAME


def sample_indices(total, count):
    if total <= 0 or count <= 0:
        return []
    count = min(int(count), total)
    return np.linspace(0, total - 1, num=count, dtype=int).tolist()


def safe_box(box, image):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
    x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
    y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
    return x1, y1, x2, y2


def context_region(box, image, enlarge_ratio):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = safe_box(box, image)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1, 1) * float(enlarge_ratio)
    cx1 = max(0, int(round(center_x - side / 2.0)))
    cy1 = max(0, int(round(center_y - side / 2.0)))
    cx2 = min(width - 1, int(round(center_x + side / 2.0)))
    cy2 = min(height - 1, int(round(center_y + side / 2.0)))
    return cx1, cy1, max(cx1 + 1, cx2), max(cy1 + 1, cy2)


def draw_text(image, text, origin=(8, 24), color=(255, 255, 255), scale=0.6):
    x, y = origin
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(
        image,
        (max(0, x - 4), max(0, y - text_height - 5)),
        (min(image.shape[1] - 1, x + text_width + 4), min(image.shape[0] - 1, y + 5)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA)


def fit_tile(image, size):
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = min(target_width / float(width), target_height / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    tile = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    tile[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized
    return tile


def render_frame(image_path, box, frame_number, view, enlarge_ratio):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    x1, y1, x2, y2 = safe_box(box, image)
    if view == "full":
        rendered = image.copy()
        relative_box = (x1, y1, x2, y2)
        tile_size = (480, 270)
    else:
        cx1, cy1, cx2, cy2 = context_region(box, image, enlarge_ratio)
        rendered = image[cy1:cy2 + 1, cx1:cx2 + 1].copy()
        relative_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        tile_size = (360, 360)

    bx1, by1, bx2, by2 = relative_box
    thickness = max(2, int(round(min(rendered.shape[:2]) / 120.0)))
    cv2.rectangle(rendered, (bx1, by1), (bx2, by2), (0, 255, 255), thickness)
    draw_text(rendered, "t{:02d}".format(frame_number), origin=(8, 24))
    return fit_tile(rendered, tile_size)


def make_mosaic(tiles, tile_size):
    width, height = tile_size
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    padded = list(tiles[:9]) + [blank] * max(0, 9 - len(tiles))
    return np.vstack([
        np.hstack(padded[0:3]),
        np.hstack(padded[3:6]),
        np.hstack(padded[6:9]),
    ])


def save_preview(data, index, output_number, output_dir, frames, view, enlarge_ratio):
    image_sequence = data["image"][index]
    box_sequence = data["bbox"][index]
    frame_count = min(max(1, int(frames)), 9, len(image_sequence), len(box_sequence))
    start = len(image_sequence) - frame_count

    tiles = []
    for offset, (image_path, box) in enumerate(zip(
            image_sequence[-frame_count:], box_sequence[-frame_count:])):
        tile = render_frame(
            image_path, box, start + offset + 1, view, enlarge_ratio)
        if tile is not None:
            tiles.append(tile)

    tile_size = (480, 270) if view == "full" else (360, 360)
    mosaic = make_mosaic(tiles, tile_size)
    label = int(data["label"][index]) if "label" in data else -1
    label_name = CLASS_ID_TO_NAME.get(label, str(label))
    last_path = str(image_sequence[-1]).replace("\\", "/")
    frame_name = os.path.splitext(os.path.basename(last_path))[0]
    path_parts = last_path.split("/")
    record = next((part for part in path_parts if part.startswith("RECORD")), "RECORD")
    drive = next((part for part in path_parts if part.startswith("DRIVE")), "DRIVE")

    header = "idx={} true={} ({}) view={} {} {} {}".format(
        index, label, label_name, view, record, drive, frame_name)
    header_height = 36
    output = np.zeros(
        (mosaic.shape[0] + header_height, mosaic.shape[1], 3), dtype=np.uint8)
    output[header_height:] = mosaic
    draw_text(output, header, origin=(8, 25), color=(255, 255, 0), scale=0.55)

    filename = "crop_preview_{:03d}_idx{:06d}_{}_{}_{}_{}.jpg".format(
        output_number, index, record, drive, frame_name, view)
    output_path = os.path.join(output_dir, filename)
    if not cv2.imwrite(output_path, output):
        raise IOError("Could not write preview: {}".format(output_path))
    print("  saved:", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preview car-motion bbox/context crops without a model")
    parser.add_argument(
        "--config_file", default="configs/configs_car_motion.yaml",
        help="YAML configuration file")
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test",
        help="Dataset split to preview")
    parser.add_argument("--count", type=int, default=None, help="Number of samples")
    parser.add_argument(
        "--indices", type=str, default=None,
        help="Comma-separated exact sample indices, e.g. 0,175200")
    parser.add_argument("--frames", type=int, default=None, help="Frames per mosaic (max 9)")
    parser.add_argument(
        "--view", choices=("full", "context", "both"), default="both",
        help="full shows annotation alignment; context shows the resulting crop")
    parser.add_argument("--out_dir", default=None, help="Preview output directory")
    args = parser.parse_args()

    with open(args.config_file, "r") as config_file:
        configs = yaml.safe_load(config_file)
    model_opts = configs["model_opts"]
    data_opts = dict(configs["data_opts"])
    data_opts["min_track_size"] = (
        int(model_opts["obs_length"]) + 2 * int(model_opts["time_to_event"]))

    if model_opts.get("dataset") != "car_motion":
        raise ValueError("preview_crops.py currently supports dataset=car_motion only")

    count = args.count if args.count is not None else int(
        model_opts.get("visual_sample_count", 3))
    frames = args.frames if args.frames is not None else int(
        model_opts.get("visual_sample_frames", 9))
    output_dir = args.out_dir or model_opts.get(
        "crop_preview_out_dir", "./models/crop_previews")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading {} annotations only (no model, training, or inference)...".format(args.split))
    dataset = CarMotion(data_path=data_opts["path_to_dataset"])
    data = dataset.generate_data_trajectory_sequence(args.split, **data_opts)
    total = min(len(data.get("image", [])), len(data.get("bbox", [])))
    if args.indices:
        indices = [int(value.strip()) for value in args.indices.split(",") if value.strip()]
        invalid = [index for index in indices if index < 0 or index >= total]
        if invalid:
            raise IndexError(
                "Sample indices out of range (total={}): {}".format(total, invalid))
    else:
        indices = sample_indices(total, count)
    print("Samples: {} / {} | indices: {}".format(len(indices), total, indices))
    print("Saving crop previews to:", output_dir)

    views = ("full", "context") if args.view == "both" else (args.view,)
    enlarge_ratio = float(model_opts.get("enlarge_ratio", 1.5))
    for output_number, index in enumerate(indices):
        for view in views:
            save_preview(
                data, index, output_number, output_dir,
                frames, view, enlarge_ratio)


if __name__ == "__main__":
    main()
