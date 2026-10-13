"""Length-stratified win rates (verbosity check for the report).

For every judged pair, compare candidate vs baseline response length (words) and
report the candidate's win/loss/tie counts when it is longer (>+20%), similar
(within ±20%), or shorter (<-20%) than the baseline.

  python scripts/analyze_judgments.py --pairs P.jsonl --judged J.jsonl --out A.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pa2.utils import read_jsonl, write_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    pairs = {p["pair_id"]: p for p in read_jsonl(args.pairs)}
    judged = read_jsonl(args.judged)
    buckets = {"candidate_longer": Counter(), "similar_length": Counter(), "candidate_shorter": Counter()}
    ratios = []
    for j in judged:
        p = pairs[j["pair_id"]]
        cand_is_a = p["meta"]["A"] == p["meta"]["candidate"]
        cand, base = (p["response_a"], p["response_b"]) if cand_is_a else (p["response_b"], p["response_a"])
        lc, lb = len(cand.split()), max(1, len(base.split()))
        r = lc / lb
        ratios.append(r)
        key = "candidate_longer" if r > 1.2 else "candidate_shorter" if r < 1 / 1.2 else "similar_length"
        w = j["winner_model"]
        buckets[key]["win" if w == p["meta"]["candidate"] else "loss" if w == p["meta"]["baseline"]
                     else "tie" if w == "tie" else "unknown"] += 1

    out = {"pairs": os.path.basename(args.pairs), "n": len(judged),
           "median_len_ratio_candidate_over_baseline": sorted(ratios)[len(ratios) // 2] if ratios else None}
    for k, c in buckets.items():
        decided = c["win"] + c["loss"]
        out[k] = dict(c) | {"n": sum(c.values()), "win_rate_excl_ties": c["win"] / decided if decided else None}
    write_json(args.out, out)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
