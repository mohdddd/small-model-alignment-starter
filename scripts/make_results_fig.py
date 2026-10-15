"""Results figure for the report: (a) HH-DPO vs SFT per category, (b) UF-DPO vs SFT by length bucket.
Counts are swap-consistent (win only if the model wins in both A/B orders)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
WIN, TIE, LOSS, INK, MUTED, GRID = "#2a78d6", "#c9c9c9", "#eb6834", "#222222", "#6b6b6b", "#e6e6e6"
cats = json.load(open(OUT + "/eval/hh_dpo_vs_sft_hh_swap_consistency.json"))["by_category"]
final = json.load(open(OUT + "/eval/final_stats.json"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), dpi=220, gridspec_kw={"width_ratios": [1.55, 1]})
order = sorted(cats, key=lambda k: (cats[k].get("win", 0) - cats[k].get("loss", 0)))
labels = [k.replace("_", " ") for k in order]
for i, k in enumerate(order):
    w, t, l = cats[k].get("win", 0), cats[k].get("tie", 0), cats[k].get("loss", 0)
    ax1.barh(i, w, color=WIN, edgecolor="white", linewidth=1.5, label="DPO wins" if i == 0 else None)
    ax1.barh(i, t, left=w, color=TIE, edgecolor="white", linewidth=1.5, label="tie / order-inconsistent" if i == 0 else None)
    ax1.barh(i, l, left=w + t, color=LOSS, edgecolor="white", linewidth=1.5, label="SFT wins" if i == 0 else None)
    ax1.text(31, i, f"{w}/{t}/{l}", va="center", fontsize=7.5, color=MUTED)
ax1.set_yticks(range(len(order)), labels, fontsize=8, color=INK)
ax1.set_xlim(0, 36); ax1.set_xticks([0, 10, 20, 30])
ax1.set_xlabel("prompts (30 per category); W/T/L at right", fontsize=8, color=MUTED)
ax1.set_title("(a) HH-RLHF DPO vs SFT, by category", fontsize=9.5, loc="left", color=INK)
ax1.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.45, -0.38), ncol=3)

lb = final["uf_dpo_vs_sft_alpaca"]["by_length"]
groups = [("longer", "DPO >1.2x longer"), ("similar", "within ±20%"), ("shorter", "DPO >1.2x shorter")]
for i, (g, name) in enumerate(groups):
    w, l, t = lb[g]
    n = w + l + t
    ax2.bar(i, w / (w + l), color=WIN, width=0.6, edgecolor="white")
    ax2.text(i, w / (w + l) - 0.04, f"{w / (w + l):.0%}\n{w}–{l}\nn={n}", ha="center", va="top", fontsize=7.5, color="white")
ax2.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
ax2.set_xticks(range(3), [n for _, n in groups], fontsize=8, color=INK)
ax2.set_ylim(0, 0.7); ax2.set_ylabel("DPO win rate (excl. ties)", fontsize=8, color=MUTED)
ax2.set_title("(b) UltraFeedback DPO vs SFT, by length ratio", fontsize=9.5, loc="left", color=INK)
for ax in (ax1, ax2):
    ax.grid(True, axis="x" if ax is ax1 else "y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbdbd")
    ax.tick_params(colors=MUTED, labelsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], bbox_inches="tight", facecolor="white")
print("wrote", sys.argv[2])
