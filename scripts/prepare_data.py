"""Download + filter the datasets for each track and write fixed JSONL/JSON files.

Usage:
  python scripts/prepare_data.py --which sft alpaca --out_dir outputs/data
  python scripts/prepare_data.py --which hh --out_dir outputs/data
  python scripts/prepare_data.py --which uf --out_dir outputs/data
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from pa2 import data as D  # noqa: E402
from pa2.utils import read_json, write_json, write_jsonl  # noqa: E402

SPECS = {
    "sft": ("HuggingFaceH4/ultrachat_200k", "train_sft", "test_sft", D.ultrachat_rows),
    "hh": ("Anthropic/hh-rlhf", "train", "test", D.hh_rows),
    "uf": ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs", D.uf_rows),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", required=True, choices=["sft", "hh", "uf", "alpaca"])
    ap.add_argument("--out_dir", default="outputs/data")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--sft_train", type=int, default=30000)
    ap.add_argument("--pref_train", type=int, default=15000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--max_prompt_tokens", type=int, default=256)
    ap.add_argument("--alpaca_n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, "data_stats.json")
    all_stats = read_json(stats_path) if os.path.exists(stats_path) else {}

    for which in args.which:
        if which == "alpaca":
            out = os.path.join(args.out_dir, f"alpaca_eval_{args.alpaca_n}_seed{args.seed}.json")
            if os.path.exists(out):
                print("exists:", out)
                continue
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
            subset = D.alpaca_subset(read_json(path), args.alpaca_n, args.seed)
            write_json(out, subset)
            all_stats["alpaca"] = {"source_rows": len(read_json(path)), "kept": len(subset), "seed": args.seed}
            print("wrote", out, len(subset))
            continue

        repo, train_split, val_split, row_fn = SPECS[which]
        n_train = args.sft_train if which == "sft" else args.pref_train
        train_out = os.path.join(args.out_dir, f"{which}_train.jsonl")
        val_out = os.path.join(args.out_dir, f"{which}_val.jsonl")
        if os.path.exists(train_out) and os.path.exists(val_out):
            print("exists:", train_out, val_out)
            continue
        tr_rows, tr_stats = D.build_split(load_dataset(repo, split=train_split), row_fn, tok, n_train,
                                          args.max_prompt_tokens, args.seed, "train")
        va_rows, va_stats = D.build_split(load_dataset(repo, split=val_split), row_fn, tok, args.n_val,
                                          args.max_prompt_tokens, args.seed, "val")
        write_jsonl(train_out, tr_rows)
        write_jsonl(val_out, va_rows)
        all_stats[which] = {"dataset": repo, "train_split": train_split, "val_split": val_split,
                            "train": tr_stats, "val": va_stats, "seed": args.seed,
                            "max_prompt_tokens": args.max_prompt_tokens}
        print(which, "train", len(tr_rows), "val", len(va_rows), tr_stats, va_stats)

    write_json(stats_path, all_stats)


if __name__ == "__main__":
    main()
