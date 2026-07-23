"""
metrics.py
-----------
Calibration and reliability metrics. This is the scientific core of the paper.

Given, for a set of items:
  - correct:    array of 0/1 (was the model's answer right?)
  - confidence: array in [0,1] (model's stated confidence)

we compute:
  - accuracy
  - Expected Calibration Error (ECE)  -- headline calibration metric
  - Maximum Calibration Error (MCE)
  - Brier score                       -- proper scoring rule
  - reliability-diagram bin data      -- for the money figure
  - error-detection AUROC             -- can confidence separate right/wrong?

All confidences are expected in [0,1]. If a model verbalizes 0-100, divide by 100
before calling these.
"""

from __future__ import annotations
import numpy as np


def accuracy(correct: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=float)
    return float(correct.mean()) if len(correct) else float("nan")


def brier_score(correct: np.ndarray, confidence: np.ndarray) -> float:
    """Mean squared error between confidence and correctness. Lower = better."""
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    return float(np.mean((confidence - correct) ** 2))


def _bin_stats(correct, confidence, n_bins=10):
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bins[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            rows.append(dict(bin=b, low=bins[b], high=bins[b + 1],
                             count=0, conf=np.nan, acc=np.nan))
            continue
        rows.append(dict(
            bin=b, low=float(bins[b]), high=float(bins[b + 1]),
            count=count,
            conf=float(confidence[mask].mean()),
            acc=float(correct[mask].mean()),
        ))
    return rows


def expected_calibration_error(correct, confidence, n_bins=10) -> float:
    """ECE: weighted average |confidence - accuracy| across confidence bins."""
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    if n == 0:
        return float("nan")
    ece = 0.0
    for row in _bin_stats(correct, confidence, n_bins):
        if row["count"] == 0:
            continue
        ece += (row["count"] / n) * abs(row["conf"] - row["acc"])
    return float(ece)


def maximum_calibration_error(correct, confidence, n_bins=10) -> float:
    gaps = [abs(r["conf"] - r["acc"]) for r in _bin_stats(correct, confidence, n_bins)
            if r["count"] > 0]
    return float(max(gaps)) if gaps else float("nan")


def error_detection_auroc(correct, confidence) -> float:
    """
    AUROC for using confidence to distinguish correct (positive) from wrong.
    1.0 = confidence perfectly separates; 0.5 = no better than chance.
    Implemented without sklearn (rank-based Mann-Whitney formulation).
    """
    correct = np.asarray(correct, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    pos = confidence[correct == 1]
    neg = confidence[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUROC
    all_scores = np.concatenate([pos, neg])
    order = all_scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    avg_rank = tie_sum / counts
    ranks = avg_rank[inv]
    r_pos = ranks[:len(pos)].sum()
    auroc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auroc)


def reliability_diagram_data(correct, confidence, n_bins=10):
    """Return per-bin (confidence, accuracy, count) for plotting."""
    return _bin_stats(correct, confidence, n_bins)


def compute_all(correct, confidence, n_bins=10) -> dict:
    """Convenience: all headline metrics in one dict."""
    return dict(
        n=int(len(correct)),
        accuracy=accuracy(correct),
        ece=expected_calibration_error(correct, confidence, n_bins),
        mce=maximum_calibration_error(correct, confidence, n_bins),
        brier=brier_score(correct, confidence),
        err_auroc=error_detection_auroc(correct, confidence),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500

    # Case 1: a WELL-calibrated model (confidence ~ matches accuracy)
    true_p = rng.uniform(0, 1, n)
    correct = (rng.uniform(0, 1, n) < true_p).astype(int)
    conf = true_p
    print("Well-calibrated model:")
    print(" ", compute_all(correct, conf))

    # Case 2: an OVERCONFIDENT model (always says ~0.9 but only ~0.6 accurate)
    correct2 = (rng.uniform(0, 1, n) < 0.6).astype(int)
    conf2 = np.clip(rng.normal(0.9, 0.03, n), 0, 1)
    print("Overconfident model (the problem we expect to find):")
    print(" ", compute_all(correct2, conf2))
    print("\nNote how the overconfident model has much higher ECE and Brier.")
