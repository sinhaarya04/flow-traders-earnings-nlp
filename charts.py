"""Generate the two one-pager charts."""
import json, pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "onepager"; OUT.mkdir(exist_ok=True)
res = json.loads((ROOT / "results.json").read_text())
fin = pd.read_csv(ROOT / "data" / "financials.csv")

INK = "#0f1b2d"; ACCENT = "#0b6bcb"; WARM = "#e07a1f"; GREY = "#8a94a3"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK})

# ---- Chart 1: NTI vs Average VIX (dual axis) -- the punchline ----
fig, ax1 = plt.subplots(figsize=(6.4, 3.5), dpi=150)
x = np.arange(len(fin))
ax1.bar(x, fin["nti_eur_m"], color=ACCENT, alpha=0.85, width=0.6, label="Net Trading Income (€m)")
ax1.set_ylabel("Net Trading Income (€m)", color=ACCENT, fontsize=10, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(fin["quarter"], rotation=45, ha="right", fontsize=8)
ax1.tick_params(axis="y", labelcolor=ACCENT)

ax2 = ax1.twinx()
ax2.plot(x, fin["avg_vix"], color=WARM, marker="o", lw=2.4, label="Average VIX")
ax2.set_ylabel("Average VIX", color=WARM, fontsize=10, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=WARM)

# annotate the honest outlier
i = fin.index[fin["quarter"] == "2024Q1"][0]
ax1.annotate("2024Q1: low VIX,\nhigh NTI (crypto/BTC-ETF)",
             xy=(i, fin["nti_eur_m"][i]), xytext=(2.4, 138),
             fontsize=7.2, color=GREY, ha="left",
             arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))

r = res["nti_vix_correlation"]
ax1.set_title(f"Flow's P&L tracks volatility — but imperfectly  (r = {r})",
              fontsize=11, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(OUT / "chart_nti_vix.png", bbox_inches="tight")
plt.close(fig)

# ---- Chart 2: recurring words, colored by how many quarters they appear in ----
tw = res["top_words"][:10][::-1]
words = [w["word"] for w in tw]
totals = [w["total"] for w in tw]
inq = [w["in_quarters"] for w in tw]
colors = [ACCENT if q == 8 else (WARM if q == 7 else GREY) for q in inq]

fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=150)
bars = ax.barh(words, totals, color=colors)
for b, q in zip(bars, inq):
    ax.text(b.get_width() + 0.6, b.get_y() + b.get_height()/2,
            f"{q}/8", va="center", fontsize=8, fontweight="bold", color=INK)
ax.set_xlabel("Total mentions across 8 letters", fontsize=9)
ax.set_title("The vocabulary barely changes\n(blue = in every letter, orange = 7/8)",
             fontsize=11, fontweight="bold", pad=10)
ax.margins(x=0.12)
fig.tight_layout()
fig.savefig(OUT / "chart_words.png", bbox_inches="tight")
plt.close(fig)

# ---- Chart 3: vocabulary recycling by quarter (NO words revealed) -- for the handout ----
rec = res["recycled_pct_by_quarter"]
qs = [q for q in res["quarters"] if rec[q] is not None]
vals = [rec[q] for q in qs]
avg = res["avg_recycled_pct"]

fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=150)
x = np.arange(len(qs))
ax.bar(x, vals, color=ACCENT, alpha=0.85, width=0.6)
ax.axhline(avg, color=WARM, lw=2, ls="--", label=f"average {avg:.0f}%")
ax.set_xticks(x); ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("% of vocabulary seen in a prior letter", fontsize=9)
ax.set_ylim(0, 80)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Each letter recycles more of the last one\n(themes converge; the story stops being new)",
             fontsize=11, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(OUT / "chart_recycling.png", bbox_inches="tight")
plt.close(fig)

print("Saved:", [p.name for p in OUT.glob("*.png")])
