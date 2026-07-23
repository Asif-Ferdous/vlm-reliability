"""
calibrate.py
-------------
The training-free calibration fix (RQ3) — the paper's actual contribution.

Approach 1: Temperature scaling of verbalized confidence.
    We fit a single scalar T on a small held-out set to rescale confidences so
    that stated confidence better matches empirical accuracy. Training-free in
    the sense that the VLM is NOT retrained; we only post-process its outputs.

    Confidence rescaling used here (logit-space temperature):
        p' = sigmoid( logit(p) / T )
    T > 1 softens overconfident probabilities toward 0.5.

Usage in the paper:
    - Fit T on a small calibration split (e.g. 20% of clean items).
    - Apply the SAME T to all degraded conditions.
    - Report ECE/Brier before vs. after. Improvement = contribution.
"""

from __future__ import annotations
import numpy as np
from src.metrics import expected_calibration_error, brier_score


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(conf: np.ndarray, T: float) -> np.ndarray:
    return _sigmoid(_logit(np.asarray(conf, dtype=float)) / T)


def fit_temperature(correct: np.ndarray, conf: np.ndarray,
                    grid=None) -> float:
    """
    Fit temperature T by minimizing ECE on a held-out calibration set.
    Simple, robust grid search (no gradients needed).
    """
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(conf, dtype=float)
    if grid is None:
        grid = np.linspace(0.5, 5.0, 46)  # T from 0.5 to 5.0
    best_T, best_ece = 1.0, float("inf")
    for T in grid:
        scaled = apply_temperature(conf, T)
        ece = expected_calibration_error(correct, scaled)
        if ece < best_ece:
            best_ece, best_T = ece, T
    return float(best_T)


def evaluate_fix(correct, conf, T):
    """Return before/after metrics for reporting."""
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(conf, dtype=float)
    scaled = apply_temperature(conf, T)
    return dict(
        T=T,
        ece_before=expected_calibration_error(correct, conf),
        ece_after=expected_calibration_error(correct, scaled),
        brier_before=brier_score(correct, conf),
        brier_after=brier_score(correct, scaled),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    # simulate an overconfident model: 60% accurate, states ~0.9
    correct = (rng.uniform(0, 1, n) < 0.6).astype(int)
    conf = np.clip(rng.normal(0.9, 0.03, n), 0.01, 0.99)

    # split: fit T on first half, evaluate on second half
    T = fit_temperature(correct[:200], conf[:200])
    res = evaluate_fix(correct[200:], conf[200:], T)
    print(f"Fitted temperature T = {T:.2f}")
    print(f"ECE   {res['ece_before']:.3f} -> {res['ece_after']:.3f}")
    print(f"Brier {res['brier_before']:.3f} -> {res['brier_after']:.3f}")
    print("\nIf ECE drops after scaling, the training-free fix works. "
          "That is the paper's headline result.")
