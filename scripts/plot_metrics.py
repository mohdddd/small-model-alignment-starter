"""Training plots for the report (static PNG, one figure per run).

  python scripts/plot_metrics.py --run_dir outputs/sft --kind sft
  python scripts/plot_metrics.py --run_dir outputs/dpo_hh --kind dpo --title "HH-RLHF DPO"
  python scripts/plot_metrics.py --run_dir outputs/lora_persona --kind lora

Train curves: thin raw line + EMA (smoothing stated in the legend). Validation:
markers + line. The selected (best) checkpoint is a dashed vertical line.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pa2.utils import read_jsonl  # noqa: E402

TRAIN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#222222", "#6b6b6b", "#e6e6e6"
EMA = 0.9


def ema(values, alpha=EMA):
    """Bias-corrected EMA (no lag toward the first value)."""
    out, s = [], 0.0
    for t, v in enumerate(values, 1):
        s = alpha * s + (1 - alpha) * v
        out.append(s / (1 - alpha ** t))
    return out


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("optimizer step", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbdbd")
    ax.tick_params(colors=MUTED, labelsize=8)


def panel(ax, train_x, train_y, val_x, val_y, title, ylabel, best_step=None, train_label="train"):
    if train_x:
        ax.plot(train_x, train_y, color=TRAIN, alpha=0.25, linewidth=1)
        ax.plot(train_x, ema(train_y), color=TRAIN, linewidth=2, label=f"{train_label} (EMA {EMA})")
    if val_x:
        ax.plot(val_x, val_y, color=VAL, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.5, label="validation")
    if best_step is not None:
        ax.axvline(best_step, color=MUTED, linestyle="--", linewidth=1, label=f"selected ckpt (step {best_step})")
    style(ax, title, ylabel)
    ax.legend(fontsize=8, frameon=False)


def plot_sft(rows, title):
    tr = [r for r in rows if r.get("kind") == "train"]
    ev = [r for r in rows if r.get("kind") == "eval"]
    best = min(ev, key=lambda r: r["val_loss"])["step"] if ev else None
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    panel(ax, [r["step"] for r in tr], [r["train_loss"] for r in tr], [r["step"] for r in ev],
          [r["val_loss"] for r in ev], title or "SFT: response-token NLL", "loss (nats / token)", best)
    return fig


def plot_dpo(rows, title):
    tr = [r for r in rows if r.get("kind") == "train"]
    ev = [r for r in rows if r.get("kind") == "eval"]
    best_rows = [r for r in ev if r.get("best")]
    best = best_rows[-1]["step"] if best_rows else None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=200)
    specs = [("train_loss", "val_loss", "DPO loss", "loss"),
             ("train_reward_acc", "val_reward_acc", "Reward accuracy (r_w > r_l)", "accuracy"),
             ("train_reward_margin", "val_reward_margin", "Reward margin (r_w - r_l)", "margin")]
    for ax, (tk, vk, t, yl) in zip(axes, specs):
        panel(ax, [r["step"] for r in tr], [r[tk] for r in tr], [r["step"] for r in ev], [r[vk] for r in ev],
              t, yl, best)
    fig.suptitle(title or "DPO", fontsize=11, color=INK, x=0.01, ha="left")
    return fig


def plot_lora(rows, title):
    tr = [r for r in rows if "loss" in r and "eval_loss" not in r and "step" in r]
    ev = [r for r in rows if "eval_loss" in r]
    xs_ev = [r.get("step", 0) for r in ev]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=200)
    specs = [("loss", "eval_loss", "DPO loss", "loss"),
             ("rewards/accuracies", "eval_rewards/accuracies", "Reward accuracy", "accuracy"),
             ("rewards/margins", "eval_rewards/margins", "Reward margin", "margin")]
    for ax, (tk, vk, t, yl) in zip(axes, specs):
        t_rows = [r for r in tr if tk in r]
        panel(ax, [r["step"] for r in t_rows], [r[tk] for r in t_rows],
              [x for x, r in zip(xs_ev, ev) if vk in r], [r[vk] for r in ev if vk in r], t, yl)
    fig.suptitle(title or "LoRA-DPO persona (TRL)", fontsize=11, color=INK, x=0.01, ha="left")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--kind", choices=["sft", "dpo", "lora"], required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows = read_jsonl(os.path.join(args.run_dir, "metrics.jsonl"))
    fig = {"sft": plot_sft, "dpo": plot_dpo, "lora": plot_lora}[args.kind](rows, args.title)
    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, f"{args.kind}_curves.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
