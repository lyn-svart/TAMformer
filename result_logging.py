"""Write training and test metrics to plain-text log files."""

import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np


def default_results_dir(base_dir: str) -> str:
    out = os.path.join(base_dir, "results")
    os.makedirs(out, exist_ok=True)
    return out


def training_log_path(results_dir: str) -> str:
    return os.path.join(results_dir, "training_results.txt")


def test_log_path(results_dir: str) -> str:
    return os.path.join(results_dir, "test_results.txt")


def _write_lines(path: str, lines: Iterable[str], mode: str = "w") -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
        f.flush()


def write_run_header(path: str, title: str, config_lines: List[str]) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 72,
        title,
        "Started: {}".format(stamp),
        "=" * 72,
    ] + config_lines + [""]
    _write_lines(path, lines, mode="w")


def append_epoch_line(path: str, line: str) -> None:
    _write_lines(path, [line], mode="a")


def write_keras_training_history(path: str, history) -> None:
    """Append Keras History epoch table after fit()."""
    if history is None or not getattr(history, "history", None):
        append_epoch_line(path, "(no training history — test-only or resume without fit)")
        return

    hist = history.history
    keys = sorted(hist.keys())
    lines = ["", "Keras fit() history (per epoch):", "-" * 72]
    header = "{:>5}  ".format("epoch") + "  ".join("{:>22}".format(k) for k in keys)
    lines.append(header)
    n_epochs = len(next(iter(hist.values())))
    for i in range(n_epochs):
        row = "{:>5}  ".format(i + 1)
        row += "  ".join("{:>22.6f}".format(hist[k][i]) for k in keys)
        lines.append(row)
    lines.append("")
    _write_lines(path, lines, mode="a")


def format_motion_metrics_line(
    prefix,
    acc,
    auc_macro,
    f1_macro,
    f1_weighted,
    precision_macro,
    recall_macro,
):
    return (
        "{} acc: {}".format(prefix, acc)
        + " - auc_macro_ovr: {}".format(auc_macro)
        + " - f1_macro: {}".format(f1_macro)
        + " - f1_weighted: {}".format(f1_weighted)
        + " - precision_macro: {}".format(precision_macro)
        + " - recall_macro: {}".format(recall_macro)
    )


def format_per_class_metrics(
    y_true,
    y_pred,
    y_scores,
    num_classes,
    class_names=None,
):
    lines = [
        "",
        "Per-class metrics:",
        "{:>7} {:>8} {:>10} {:>10} {:>10} {:>16} {:>16}".format(
            "class", "support", "acc/recall", "precision", "f1",
            "avg_true_conf", "avg_pred_conf",
        ),
        "-" * 82,
    ]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)

    for class_id in range(int(num_classes)):
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        support = int(np.sum(true_mask))
        pred_count = int(np.sum(pred_mask))
        tp = int(np.sum(true_mask & pred_mask))

        recall = (tp / support) if support else 0.0
        precision = (tp / pred_count) if pred_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        avg_true_conf = float(np.mean(y_scores[true_mask, class_id])) if support else 0.0
        if pred_count:
            pred_idx = np.where(pred_mask)[0]
            avg_pred_conf = float(np.mean(y_scores[pred_idx, class_id]))
        else:
            avg_pred_conf = 0.0

        label = class_names.get(class_id, str(class_id)) if class_names else str(class_id)
        lines.append(
            "{:>7} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>16.4f} {:>16.4f}  ({})".format(
                class_id, support, recall, precision, f1, avg_true_conf, avg_pred_conf, label,
            )
        )
    return "\n".join(lines)


def write_test_results(path: str, sections: List[str]) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 72,
        "Test results",
        "Written: {}".format(stamp),
        "=" * 72,
        "",
    ]
    for section in sections:
        if section:
            lines.append(section.rstrip())
            lines.append("")
    _write_lines(path, lines, mode="w")
