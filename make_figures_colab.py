# ============================================================
#  PUBLICATION FIGURES — both models on shared axes
#  Paste as a new cell at the bottom of your Colab notebook and run.
#  Requires: summary2 (Qwen), smol_rows (SmolVLM), groups2, groups3
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

FIGDIR = f"{OUT_DIR}/figures_paper"
os.makedirs(FIGDIR, exist_ok=True)

# consistent styling
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

CONDS = ["jpeg", "motion_blur", "low_light", "glare", "rotation", "resample"]
NICE = {"jpeg": "JPEG", "motion_blur": "Motion blur", "low_light": "Low light",
        "glare": "Glare", "rotation": "Rotation", "resample": "Resample"}


# ------------------------------------------------------------
# FIGURE 1 — the headline: verbalized vs internal AUROC (Qwen)
# ------------------------------------------------------------
qw = {(s["degradation"], s["severity"]): s for s in summary2}
labels, av, ai = [], [], []
for c in CONDS:
    for sev in [1, 2, 3]:
        s = qw.get((c, sev))
        if not s: continue
        if np.isnan(s["auroc_verb"]) or np.isnan(s["auroc_int"]): continue
        labels.append(f"{NICE[c][:4]} s{sev}")
        av.append(s["auroc_verb"]); ai.append(s["auroc_int"])

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w/2, av, w, label="Verbalized", color="#c44e52", edgecolor="black", lw=0.5)
ax.bar(x + w/2, ai, w, label="Internal",   color="#4c72b0", edgecolor="black", lw=0.5)
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
sm = {(s["degradation"], s["severity"]): s for s in smol_rows}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (src, name) in zip(axes, [(qw, "Qwen2-VL-2B"), (sm, "SmolVLM")]):
    for c in CONDS:
        xs, ys = [], []
        for sev in [1, 2, 3]:
            s = src.get((c, sev))
            if s: xs.append(sev); ys.append(s["accuracy"])
        if xs: ax.plot(xs, ys, marker="o", label=NICE[c])
    ax.axhline(0.25, ls=":", c="red", lw=1, label="Chance (4-way)")
    ax.set_xlabel("Corruption severity"); ax.set_xticks([1, 2, 3])
    ax.set_title(name)
axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0, 1.05)
axes[1].legend(fontsize=8, ncol=2, loc="lower left")
plt.suptitle("Accuracy collapses under severe low light in both models", y=1.02)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig2_accuracy_both.png",
                                bbox_inches="tight"); plt.close()
print("saved fig2_accuracy_both.png")


# ------------------------------------------------------------
# FIGURE 3 — the divergence: accuracy vs both confidences (Qwen)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in ["low_light"]:
    xs = [0, 1, 2, 3]
    acc = [qw[("clean", 0)]["accuracy"]] + [qw[(c, s)]["accuracy"] for s in [1,2,3]]
    vb  = [qw[("clean", 0)]["verb_mean"]] + [qw[(c, s)]["verb_mean"] for s in [1,2,3]]
    it  = [qw[("clean", 0)]["int_mean"]]  + [qw[(c, s)]["int_mean"]  for s in [1,2,3]]
    ax.plot(xs, acc, marker="o", lw=2.5, label="Accuracy", color="black")
    ax.plot(xs, vb,  marker="s", lw=2, ls="--", label="Verbalized confidence",
            color="#c44e52")
    ax.plot(xs, it,  marker="^", lw=2, ls="-.", label="Internal confidence",
            color="#4c72b0")
ax.set_xlabel("Low-light severity (0 = clean)"); ax.set_xticks([0,1,2,3])
ax.set_ylabel("Value"); ax.set_ylim(0, 1.05)
ax.set_title("Accuracy collapses; neither confidence signal follows")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig3_divergence.png"); plt.close()
print("saved fig3_divergence.png")


# ------------------------------------------------------------
# FIGURE 4 — reliability diagrams, verbalized vs internal (clean)
# ------------------------------------------------------------
def rel_data(rows_sel, key, nbins=10):
    rs = [r for r in rows_sel if r[key] != "" and r["correct"] != ""]
    if len(rs) < 10: return None
    c = np.array([int(r["correct"]) for r in rs])
    p = np.array([float(r[key]) for r in rs])
    return _bin_stats(c, p, nbins)

sel = groups2[("clean", 0)]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, (k, title, col) in zip(axes, [
        ("verbalized", "Verbalized confidence", "#c44e52"),
        ("internal",   "Internal confidence",   "#4c72b0")]):
    bins = rel_data(sel, k)
    if bins:
        centers = [(b["low"]+b["high"])/2 for b in bins]
        accs = [b["acc"] if b["count"] > 0 else 0 for b in bins]
        ax.bar(centers, accs, width=0.09, alpha=0.8, color=col,
               edgecolor="black", lw=0.5)
    ax.plot([0,1],[0,1],"--",c="gray",lw=1)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence"); ax.set_title(title)
axes[0].set_ylabel("Actual accuracy")
plt.suptitle("Qwen2-VL, clean images: verbalized confidence occupies one bin", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig4_reliability.png",
                                bbox_inches="tight"); plt.close()
print("saved fig4_reliability.png")

print(f"\nAll figures in {FIGDIR}/")
print("Download them and place in your paper/figures/ folder.")
