"""
plots.py
---------
Generate the paper's figures from results/summary.csv and raw_predictions.csv.

Figures:
  - fig_ece_vs_severity.png      : ECE rising with corruption severity
  - fig_acc_vs_severity.png      : accuracy falling with severity
  - fig_reliability_<cond>.png   : reliability diagram (confidence vs accuracy)

Run:
    python -m src.plots
"""

from __future__ import annotations
import os
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import reliability_diagram_data


def _read_summary(path="results/summary.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _read_raw(path="results/raw_predictions.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["confidence"] == "" or r["correct"] == "":
                continue
            rows.append(r)
    return rows


def plot_metric_vs_severity(metric="ece", out="results/fig_ece_vs_severity.png"):
    rows = _read_summary()
    # group by (model, degradation) -> list of (severity, value)
    series = defaultdict(list)
    for r in rows:
        if r["degradation"] == "clean":
            continue
        key = (r["model"], r["degradation"])
        series[key].append((int(r["severity"]), float(r[metric])))
    plt.figure(figsize=(7, 4.5))
    for (model, deg), pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{deg}")
    plt.xlabel("Corruption severity")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs. severity")
    plt.xticks([1, 2, 3])
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_reliability(model=None, degradation="jpeg", severity=3,
                     out=None, n_bins=10):
    raw = _read_raw()
    sel = [r for r in raw
           if r["degradation"] == degradation and int(r["severity"]) == severity
           and (model is None or r["model"] == model)]
    if not sel:
        print(f"No rows for {degradation} sev={severity}")
        return None
    correct = np.array([int(r["correct"]) for r in sel])
    conf = np.array([float(r["confidence"]) for r in sel])
    bins = reliability_diagram_data(correct, conf, n_bins=n_bins)

    centers = [(b["low"] + b["high"]) / 2 for b in bins]
    accs = [b["acc"] if b["count"] > 0 else 0 for b in bins]

    out = out or f"results/fig_reliability_{degradation}_s{severity}.png"
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    plt.bar(centers, accs, width=1.0 / n_bins * 0.9, alpha=0.7,
            edgecolor="black", label="model")
    plt.xlabel("Stated confidence")
    plt.ylabel("Actual accuracy")
    plt.title(f"Reliability: {degradation} (severity {severity})")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    os.makedirs("results", exist_ok=True)
    outs = []
    outs.append(plot_metric_vs_severity("ece", "results/fig_ece_vs_severity.png"))
    outs.append(plot_metric_vs_severity("accuracy", "results/fig_acc_vs_severity.png"))
    outs.append(plot_reliability(degradation="jpeg", severity=3))
    outs.append(plot_reliability(degradation="clean", severity=0,
                                 out="results/fig_reliability_clean.png"))
    print("Wrote figures:")
    for o in outs:
        if o:
            print("  ", o)


if __name__ == "__main__":
    main()
