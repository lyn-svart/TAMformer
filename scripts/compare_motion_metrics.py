#!/usr/bin/env python3
"""
Print TAMformer-style motion metrics from saved numpy arrays or a one-off eval.

Usage (after you have y_true.npy, y_pred.npy, y_scores.npy from a run):
  python scripts/compare_motion_metrics.py --y-true y_true.npy --y-scores y_scores.npy

R3D-18 prints the same summary line at the end of training; use this script only
if you export arrays for offline comparison.
"""

import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from motion_labels import NUM_MOTION_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y-true", required=True, help="Path to int labels .npy")
    parser.add_argument("--y-scores", required=True, help="Path to (N, C) softmax logits/scores .npy")
    parser.add_argument("--num-classes", type=int, default=NUM_MOTION_CLASSES)
    args = parser.parse_args()

    y_true = np.load(args.y_true).astype(int)
    y_scores = np.load(args.y_scores)
    y_pred = np.argmax(y_scores, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        y_true_oh = np.eye(args.num_classes, dtype=np.float32)[y_true]
        auc_macro = roc_auc_score(y_true_oh, y_scores, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = 0.0

    print(
        "motion acc:", acc,
        "- auc_macro_ovr:", auc_macro,
        "- f1_macro:", f1_macro,
        "- f1_weighted:", f1_weighted,
        "- precision_macro:", precision_macro,
        "- recall_macro:", recall_macro,
    )


if __name__ == "__main__":
    main()
