"""Position-bias control: judge every pair a second time with A/B swapped.

  make    : write a swapped copy of a pairs file (response_a <-> response_b, meta A <-> B)
  combine : merge original + swapped judgments. A model wins only if it wins in BOTH
            orders; disagreement counts as a tie. Reports the position-consistency rate.

  python scripts/swap_consistency.py make --pairs P.jsonl --out P_swapped.jsonl
  python scripts/swap_consistency.py combine --orig J.jsonl --swapped J_swapped.jsonl --out S.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pa2.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def make(args):
    rows = []
    for p in read_jsonl(args.pairs):
        q = dict(p, response_a=p["response_b"], response_b=p["response_a"])
        q["meta"] = dict(p["meta"], A=p["meta"]["B"], B=p["meta"]["A"])
        rows.append(q)
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} swapped pairs")


def combine(args):
    orig = {j["pair_id"]: j for j in read_jsonl(args.orig)}
    swap = {j["pair_id"]: j for j in read_jsonl(args.swapped)}
    common = sorted(set(orig) & set(swap))
    cand = next(iter(orig.values()))["meta"]["candidate"]
    base = next(iter(orig.values()))["meta"]["baseline"]

    def outcome(w):
        return "win" if w == cand else "loss" if w == base else "tie" if w == "tie" else "unknown"

    overall, by_cat, consistent = Counter(), defaultdict(Counter), 0
    for pid in common:
        a, b = outcome(orig[pid]["winner_model"]), outcome(swap[pid]["winner_model"])
        consistent += a == b
        final = a if a == b else "tie"  # disagreement across orders -> tie
        overall[final] += 1
        by_cat[orig[pid].get("category", "")][final] += 1

    def rates(c):
        w, l, t = c["win"], c["loss"], c["tie"]
        return dict(c) | {"n": sum(c.values()), "win_rate_excl_ties": w / (w + l) if w + l else None,
                          "win_rate_ties_half": (w + 0.5 * t) / (w + l + t) if w + l + t else None}

    out = {"candidate": cand, "baseline": base, "n_pairs_both_orders": len(common),
           "position_consistency_rate": consistent / len(common) if common else None,
           "overall": rates(overall), "by_category": {k: rates(v) for k, v in sorted(by_cat.items())}}
    write_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("candidate", "baseline", "n_pairs_both_orders",
                                          "position_consistency_rate", "overall")}, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--pairs", required=True)
    m.add_argument("--out", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--orig", required=True)
    c.add_argument("--swapped", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return make(args) if args.cmd == "make" else combine(args)


if __name__ == "__main__":
    main()
