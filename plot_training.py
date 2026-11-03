"""Report figure: learned-reward log p and policy entropy over training, per run."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [  # (dir, label, colour) — fixed categorical order, validated palette
    ("pilot_a", "Ours (likelihood + mask + RND)", "#2a78d6"),
    ("ours_nomask", "Ours, no mask (scored by its own no-mask model)", "#eb6834"),
    ("curiosity", "Curiosity (−log p)", "#1baf7a"),
    ("hand_final", "Hand-engineered reward", "#eda100"),
    ("ours", "Ours, entropy 0.01 (collapsed)", "#e87ba4"),
]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def series(run, key):
    L = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
    x = np.array([r["step"] for r in L if r.get(key) is not None]) / 1e3
    y = np.array([r[key] for r in L if r.get(key) is not None], float)
    k = 5  # rolling mean over 5 PPO updates
    if len(y) > k: y = np.convolve(y, np.ones(k) / k, mode="valid"); x = x[k - 1:]
    return x, y


plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), facecolor=SURF)
for ax, key, title in ((axes[0], "raw_logp", "A. Learned reward: Σ log p per step (higher = more expert-like)"),
                       (axes[1], "entropy", "B. Policy entropy (nats)")):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for run, label, c in RUNS:
        try: x, y = series(run, key)
        except (FileNotFoundError, KeyError): continue
        if len(y) == 0: continue
        ax.plot(x, y, color=c, lw=2, label=label)
    if key == "raw_logp":
        ax.axhline(-12.4, color=MUTED, lw=1.2, ls="--")
        ax.text(2, -10.5, "scripted expert (−12.4)", color=MUTED, fontsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel("environment steps (thousands)"); ax.set_xlim(0, 310)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3, labelcolor=INK)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("runs/fig_training.png", dpi=150, facecolor=SURF)
print("saved runs/fig_training.png")
