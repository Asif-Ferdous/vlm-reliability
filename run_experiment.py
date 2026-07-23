"""
run_experiment.py
------------------
The main experiment runner. Ties everything together:

  models × degradations × severities  ->  per-condition calibration metrics

Outputs:
  - results/raw_predictions.csv   (one row per item per condition)
  - results/summary.csv           (one row per model×condition×severity)

Run (no GPU, validates whole pipeline):
    python -m src.run_experiment --config configs/pilot_mock.yaml

Swap the config to a real backend later; nothing else changes.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import yaml
from PIL import Image

# Allow running as `python src/run_experiment.py` or `python -m src.run_experiment`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.degradations import apply_degradation, SEVERITIES
from src.models import build_model, CONFIDENCE_PROMPT, CONFIDENCE_PROMPT_QUALITY_AWARE
from src.metrics import compute_all
from src.toy_dataset import make_dataset


def _attach_severity(img: Image.Image, severity: int) -> Image.Image:
    """Mock backend reads severity from image.info; harmless for real models."""
    img = img.copy()
    img.info["severity"] = severity
    return img


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)

    # ---- data ----
    if cfg["dataset"]["type"] == "toy":
        items = make_dataset(n=cfg["dataset"].get("n", 60),
                             seed=cfg["dataset"].get("seed", 0))
    else:
        raise NotImplementedError(
            "Only 'toy' dataset wired in. Add real dataset loading in toy_dataset.load_real_dataset."
        )

    # ---- conditions ----
    conditions = cfg["degradations"]  # list of names; 'clean' allowed
    model_cfgs = cfg["models"]

    raw_rows = []
    summary_rows = []

    for mcfg in model_cfgs:
        model = build_model(mcfg)
        mname = mcfg.get("name", mcfg.get("backend", "model"))

        for deg in conditions:
            sev_list = [0] if deg == "clean" else SEVERITIES
            for sev in sev_list:
                correct_flags, confidences = [], []
                for idx, it in enumerate(items):
                    dimg = apply_degradation(it.image, deg, max(sev, 1))
                    dimg = _attach_severity(dimg, sev)
                    ans, conf = model.answer(dimg, it.question, it.options)
                    if conf is None:
                        # model failed to state confidence; skip in calibration
                        # but record so we can report parse-failure rate
                        raw_rows.append(dict(model=mname, degradation=deg,
                                             severity=sev, item=idx,
                                             gold=it.gold, answer=ans,
                                             confidence="", correct=""))
                        continue
                    is_correct = int(str(ans).strip().lower() ==
                                     str(it.gold).strip().lower())
                    correct_flags.append(is_correct)
                    confidences.append(conf)
                    raw_rows.append(dict(model=mname, degradation=deg,
                                         severity=sev, item=idx,
                                         gold=it.gold, answer=ans,
                                         confidence=round(conf, 4),
                                         correct=is_correct))

                m = compute_all(correct_flags, confidences,
                                n_bins=cfg.get("n_bins", 10))
                m.update(dict(model=mname, degradation=deg, severity=sev))
                summary_rows.append(m)
                print(f"[{mname:10s}] {deg:12s} sev={sev} "
                      f"acc={m['accuracy']:.3f} ece={m['ece']:.3f} "
                      f"brier={m['brier']:.3f} n={m['n']}")

    # ---- write outputs ----
    raw_path = "results/raw_predictions.csv"
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "degradation", "severity",
                                          "item", "gold", "answer",
                                          "confidence", "correct"])
        w.writeheader()
        w.writerows(raw_rows)

    sum_path = "results/summary.csv"
    fields = ["model", "degradation", "severity", "n", "accuracy",
              "ece", "mce", "brier", "err_auroc"]
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"\nWrote {raw_path} ({len(raw_rows)} rows) and "
          f"{sum_path} ({len(summary_rows)} rows).")
    return summary_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
