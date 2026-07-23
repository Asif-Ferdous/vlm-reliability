# ============================================================
#  PUBLICATION FIGURES — standalone version
#
#  Reads results directly from the CSVs in Google Drive.
#  Does NOT require the model, the notebook variables, or a prior run
#  to still be in memory — so it survives a Colab session reset.
#
#  Paste as a new cell at the bottom of the Colab notebook and run.
#  Takes about 30 seconds.
# ============================================================
import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- mount Drive if not already mounted ----
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

OUT_DIR = '/content/drive/MyDrive/vlm_reliability'
FIGDIR  = f"{OUT_DIR}/figures_paper"
os.makedirs(FIGDIR, exist_ok=True)
print("OUT_DIR:", OUT_DIR)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(v):
    """Tolerant float conversion: empty / None / 'nan' -> nan."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


# ---- load the two summary files ----
qwen_rows = load_csv(f"{OUT_DIR}/summary_dual_Qwen2-VL-2B-Instruct_dual.csv")
smol_rows = load_csv(f"{OUT_DIR}/summary_smolvlm.csv")
print(f"loaded {len(qwen_rows)} Qwen rows, {len(smol_rows)} SmolVLM rows")

qw = {(r["degradation"], int(r["severity"])): r for r in qwen_rows}
sm = {(r["degradation"], int(r["severity"])): r for r in smol_rows}

plt.rcParams.update({"font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})

CONDS = ["jpeg", "motion_blur", "low_light", "glare", "rotation", "resample"]
NICE = {"jpeg": "JPEG", "motion_blur": "Motion blur", "low_light": "Low light",
        "glare": "Glare", "rotation": "Rotation", "resample": "Resample"}


# ------------------------------------------------------------
# FIGURE 1 — headline: verbalized vs internal AUROC (Qwen)
# ------------------------------------------------------------
labels, av, ai = [], [], []
for c in CONDS:
    for sev in [1, 2, 3]:
        r = qw.get((c, sev))
        if not r:
            continue
        v, i_ = num(r["auroc_verb"]), num(r["auroc_int"])
        if np.isnan(v) or np.isnan(i_):
            continue                       # undefined AUROC -> omit
        labels.append(f"{NICE[c][:4]} s{sev}")
        av.append(v); ai.append(i_)

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w/2, av, w, label="Verbalized", color="#c44e52",
       edgecolor="black", lw=0.5)
ax.bar(x + w/2, ai, w, label="Internal", color="#4c72b0",
       edgecolor="black", lw=0.5)
ax.axhline(0.5, ls="--", c="gray", lw=1, label="Chance")
ax.set_ylabel("Error-detection AUROC"); ax.set_ylim(0, 1.05)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_title("Qwen2-VL-2B: verbalized confidence is at chance; internal is not")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig1_auroc_comparison.png"); plt.close()
print("saved fig1_auroc_comparison.png")


# ------------------------------------------------------------
# FIGURE 2 — accuracy vs severity, both models
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (src, name) in zip(axes, [(qw, "Qwen2-VL-2B"), (sm, "SmolVLM")]):
    for c in CONDS:
        xs, ys = [], []
        for sev in [1, 2, 3]:
            r = src.get((c, sev))
            if r:
                xs.append(sev); ys.append(num(r["accuracy"]))
        if xs:
            ax.plot(xs, ys, marker="o", label=NICE[c])
    ax.axhline(0.25, ls=":", c="red", lw=1, label="Chance (4-way)")
    ax.set_xlabel("Corruption severity"); ax.set_xticks([1, 2, 3])
    ax.set_title(name)
axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0, 1.05)
axes[1].legend(fontsize=8, ncol=2, loc="lower left")
plt.suptitle("Accuracy collapses under severe low light in both models", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_accuracy_both.png", bbox_inches="tight"); plt.close()
print("saved fig2_accuracy_both.png")


# ------------------------------------------------------------
# FIGURE 3 — divergence under low light (Qwen)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
xs = [0, 1, 2, 3]
acc = [num(qw[("clean", 0)]["accuracy"])]  + [num(qw[("low_light", s)]["accuracy"])  for s in [1, 2, 3]]
vb  = [num(qw[("clean", 0)]["verb_mean"])] + [num(qw[("low_light", s)]["verb_mean"]) for s in [1, 2, 3]]
it  = [num(qw[("clean", 0)]["int_mean"])]  + [num(qw[("low_light", s)]["int_mean"])  for s in [1, 2, 3]]
ax.plot(xs, acc, marker="o", lw=2.5, label="Accuracy", color="black")
ax.plot(xs, vb, marker="s", lw=2, ls="--", label="Verbalized confidence",
        color="#c44e52")
ax.plot(xs, it, marker="^", lw=2, ls="-.", label="Internal confidence",
        color="#4c72b0")
ax.set_xlabel("Low-light severity (0 = clean)"); ax.set_xticks([0, 1, 2, 3])
ax.set_ylabel("Value"); ax.set_ylim(0, 1.05)
ax.set_title("Accuracy collapses; neither confidence signal follows")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig3_divergence.png"); plt.close()
print("saved fig3_divergence.png")


# ------------------------------------------------------------
# FIGURE 4 — reliability diagrams on clean images
#
#  NOTE: on clean images both signals are concentrated in [0.8, 1.0]
#  and both look well calibrated. That is the POINT of this figure:
#  reliability diagrams cannot distinguish the two signals even when
#  their AUROC differs by >0.4. It is a methodological caution, not
#  evidence for the headline claim. Caption it accordingly.
# ------------------------------------------------------------
ck = f"{OUT_DIR}/checkpoints/Qwen2-VL-2B-Instruct_dual__clean__s0.csv"
if os.path.exists(ck):
    rows = load_csv(ck)

    def bins10(correct, conf, nb=10):
        edges = np.linspace(0, 1, nb + 1)
        idx = np.digitize(conf, edges[1:-1])
        out = []
        for b in range(nb):
            m = idx == b
            out.append(((edges[b] + edges[b + 1]) / 2,
                        float(correct[m].mean()) if m.sum() else 0.0))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (k, title, col) in zip(axes, [
            ("verbalized", "Verbalized confidence", "#c44e52"),
            ("internal",   "Internal confidence",   "#4c72b0")]):
        rs = [r for r in rows if r[k] != "" and r["correct"] != ""]
        if len(rs) >= 10:
            c = np.array([int(r["correct"]) for r in rs])
            p = np.array([float(r[k]) for r in rs])
            data = bins10(c, p)
            ax.bar([d[0] for d in data], [d[1] for d in data], width=0.09,
                   alpha=0.8, color=col, edgecolor="black", lw=0.5)
        ax.plot([0, 1], [0, 1], "--", c="gray", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_title(title)
    axes[0].set_ylabel("Actual accuracy")
    plt.suptitle("Clean images: reliability diagrams cannot separate the two signals",
                 y=1.0)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig4_reliability.png", bbox_inches="tight"); plt.close()
    print("saved fig4_reliability.png")
else:
    print("SKIPPED fig4 - checkpoint not found at:", ck)


print(f"\nDone. Figures in {FIGDIR}/")
for f in sorted(glob.glob(f"{FIGDIR}/*.png")):
    print("  ", os.path.basename(f))
print("\nDownload these and place them in paper/figures/")
