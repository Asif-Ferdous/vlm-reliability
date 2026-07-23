# VLM Reliability: Stated vs. Internal Confidence Under Image Degradation

Code, data, and figures for the paper:

**Small Vision-Language Models Know When They Are Wrong But Cannot Say So:
A Two-Model Study of Stated versus Internal Confidence Under Realistic Image Degradation**

M M Asif Ferdous · [arXiv link — add once posted]

---

## Summary

Small open-weight vision-language models are increasingly deployed on consumer
hardware, where images arrive compressed, blurred, or badly lit. In these
settings a usable uncertainty signal matters more than raw accuracy, because it
determines when a system should defer instead of answering.

We compare two confidence signals on the same predictions:

- **Verbalized confidence** — the number the model states in text
- **Internal confidence** — the model's own mean token probability

Across 3,800 predictions from Qwen2-VL-2B-Instruct and SmolVLM-Instruct under
six realistic photographic degradations at three severity levels:

| Finding | Detail |
|---|---|
| Verbalized confidence is uninformative | Near-constant 0.87–0.90 in Qwen2-VL; error-detection AUROC ≈ 0.50 (chance) |
| Internal confidence works | AUROC 0.92–0.99 on the same predictions |
| Replicates, more weakly | SmolVLM internal AUROC 0.54–0.92 |
| Verbalized may be unobtainable | SmolVLM produced a parseable confidence in 1/5 pilot attempts across 3 templates |
| Both fail under low light | Accuracy 0.99 → 0.22 (Qwen2-VL); internal AUROC falls to chance |
| Temperature scaling fails where needed | ECE improves in most conditions but worsens 0.650 → 0.756 at low light s3 |

**Practical takeaway:** use mean token probability, not stated confidence, as a
deferral signal — but do not treat it as a safety guarantee under severe
underexposure.

---

## Repository layout

```
├── notebook/
│   └── VLM_Reliability_Colab.ipynb   full pipeline, runs on a free Colab T4
├── code/
│   ├── degradations.py               six corruption families, three severities
│   ├── metrics.py                    ECE, MCE, Brier, error-detection AUROC
│   ├── calibrate.py                  post-hoc temperature scaling
│   ├── models.py                     model interface + confidence elicitation
│   ├── run_experiment.py             experiment runner
│   ├── toy_dataset.py                synthetic data for pipeline validation
│   ├── plots.py                      plotting utilities
│   └── make_figures_colab.py         generates the paper figures
├── data/
│   ├── summary_dual_Qwen2-VL-2B-Instruct_dual.csv
│   ├── summary_smolvlm.csv
│   ├── summary_with_CIs.csv          bootstrap 95% intervals
│   ├── calibration_fix_*.csv         temperature scaling results
│   └── checkpoints/                  raw per-prediction records (3,800 rows)
└── figures/                          the four paper figures
```

The CSVs in `data/` are the single source of truth for every number in the
paper.

---

## Reproducing

The full study runs in roughly 90 minutes of GPU time on a single free-tier
NVIDIA T4.

1. Open `notebook/VLM_Reliability_Colab.ipynb` in Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**

The notebook installs nothing beyond Colab's defaults, streams the dataset
(≈100 images rather than the full 800MB), and checkpoints results after every
condition so a disconnect does not lose work.

**Configuration.** `N_ITEMS` controls sample size (100 for the pilot reported in
the paper). `MODEL_ID` selects the model. Seeds are fixed throughout.

---

## Experimental setup

**Models.** Qwen2-VL-2B-Instruct, SmolVLM-Instruct — both open-weight, fp16,
greedy decoding.

**Data.** 100-item subset of Food101 (test split, seed 0), posed as four-option
multiple choice.

**Degradations.** JPEG compression, motion blur, low light, glare, rotation,
resampling — each at three severities, plus a clean baseline. 19 conditions per
model.

**Metrics.** Accuracy, ECE (10 bins), Brier score, and error-detection AUROC
with percentile bootstrap intervals (B = 2000).

---

## Known limitations

Stated plainly, because they matter for interpreting the results:

- **n ≈ 100 per condition.** Several SmolVLM confidence intervals span 0.3–0.4
  and are not distinguishable from chance. Larger n is the highest-value
  extension.
- **Undefined AUROC cells.** Conditions at ~100% accuracy have too few errors
  for AUROC to be defined; these are reported `n/a` and excluded from
  aggregates. They are not evidence of good calibration.
- **One dataset, one task format.** Generality to open-ended VQA is untested.
- **Prompt sensitivity.** One template per model, selected by pilot. The
  near-constant 0.90 in Qwen2-VL may be partly template-induced; an elicitation
  ablation is needed.
- **Synthetic degradations.** Corruptions are applied programmatically and
  approximate, rather than reproduce, real phone-camera artifacts.
- **No refusal behaviour.** Multiple choice forces an answer, so this design
  cannot capture refusal — which prior work identifies as a meaningful
  calibration mechanism.

---

## Relation to prior work

The closest prior study is Borszukovszki, de Jong, and Valdenegro-Toro (2025),
[arXiv:2504.03440](https://arxiv.org/abs/2504.03440), which examined verbalized
uncertainty in three proprietary VLMs under image corruption. That work
motivates its use of verbalized confidence by noting that for proprietary
models, internal token probabilities are not accessible.

By using open-weight models we recover that signal, enabling the direct
comparison their setting excludes. We replicate their central findings
(overconfidence, confidence clustering in the 80–100 range, ECE rising with
severity) and extend them with the verbalized/internal contrast, error-detection
AUROC, bootstrap intervals, and a direct test of temperature scaling — which
they identified as future work.

---

## Citation

```bibtex
@article{ferdous2026vlmreliability,
  title   = {Small Vision-Language Models Know When They Are Wrong But Cannot
             Say So: A Two-Model Study of Stated versus Internal Confidence
             Under Realistic Image Degradation},
  author  = {Ferdous, M M Asif},
  journal = {arXiv preprint},
  year    = {2026}
}
```

*(Update with the arXiv ID once posted.)*

---

## License

MIT for the code. Food101 is subject to its own terms; model weights are subject
to their respective licenses.
