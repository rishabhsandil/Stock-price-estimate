"""Render v2 results chart for the case study page."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(__file__)
RES  = os.path.join(ROOT, "results")
OUT  = os.path.join(ROOT, "..", "..", "portfolio", "img")
os.makedirs(OUT, exist_ok=True)

summary = pd.read_csv(os.path.join(RES, "summary.csv"), index_col=0)
# Sort by accuracy descending so the best model leads visually
summary = summary.sort_values("accuracy_mean", ascending=False)

LIME = "#c6ff3d"; VIOLET = "#7c5cff"; CORAL = "#ff7849"
TEXT  = "#ededf0"; MUTED = "#8a8a93"; BG = "#0a0a0b"; SURFACE = "#15151a"; LINE = "#26262d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": LINE,
    "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "axes.axisbelow": True,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
labels = list(summary.index)
xs = list(range(len(labels)))
best = labels[0]  # top accuracy model = XGBoost

# ---- Left: accuracy vs market baseline ----
ax = axes[0]
acc = summary["accuracy_mean"].values
err = summary["accuracy_std"].values
colors = [LIME if name == best else VIOLET for name in labels]
alphas = [1.0 if name == best else 0.55 for name in labels]
bars = ax.bar(xs, acc, color=colors, edgecolor="none", width=0.66)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)
ax.errorbar(xs, acc, yerr=err, fmt="none", ecolor=MUTED, capsize=4, alpha=0.7, linewidth=1.1)
ax.axhline(0.459, color=CORAL, linestyle="--", linewidth=1.4,
           label="market baseline  ·  45.9%")
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15)
ax.set_ylim(0.30, 0.72)
ax.set_ylabel("Share of stocks correctly tagged as beating S&P", color=TEXT)
ax.set_title("How often each model picks a market-beater",
             color=TEXT, pad=14, fontsize=12.5, fontweight="bold", loc="left")
ax.legend(facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT, fontsize=9,
          loc="upper right", framealpha=0.9)
for i, v in enumerate(acc):
    weight = "bold" if labels[i] == best else "normal"
    color = LIME if labels[i] == best else TEXT
    ax.text(i, v + err[i] + 0.012, f"{v*100:.1f}%", ha="center", va="bottom",
            color=color, fontsize=10, fontweight=weight)

# ---- Right: strategy return vs SPY ----
ax = axes[1]
strat = summary["strat_ret"].values * 100  # to percent
spy_ret = float(summary["spy_ret"].iloc[0]) * 100  # benchmark line
colors2 = [LIME if name == best else VIOLET for name in labels]
alphas2 = [1.0 if name == best else 0.55 for name in labels]
bars2 = ax.bar(xs, strat, color=colors2, edgecolor="none", width=0.66)
for bar, a in zip(bars2, alphas2):
    bar.set_alpha(a)
ax.axhline(spy_ret, color=CORAL, linestyle="--", linewidth=1.4,
           label=f"S&P 500 return  ·  {spy_ret:.1f}%")
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Avg 90-day return on top-tertile picks (after costs)", color=TEXT)
ax.set_title("What you'd earn buying each model's high-conviction picks",
             color=TEXT, pad=14, fontsize=12.5, fontweight="bold", loc="left")
ax.legend(facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT, fontsize=9,
          loc="upper right", framealpha=0.9)
ymax = max(strat.max(), spy_ret) * 1.25
ax.set_ylim(0, ymax)
for i, v in enumerate(strat):
    weight = "bold" if labels[i] == best else "normal"
    color = LIME if labels[i] == best else TEXT
    ax.text(i, v + ymax * 0.02, f"{v:.1f}%", ha="center", va="bottom",
            color=color, fontsize=10, fontweight=weight)

# Footer
fig.text(0.005, 0.005,
         "5-fold walk-forward, 64 large-cap tickers  ·  trained only on the past, tested only on the future",
         color=MUTED, fontsize=8.5, ha="left")

plt.tight_layout(rect=[0, 0.03, 1, 1])
out_path = os.path.join(OUT, "stock-results.png")
plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
print(f"Wrote {out_path}")
