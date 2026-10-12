"""Build blinded pairwise files: candidate vs baseline, A/B order randomized per
item with a fixed seed. Model identities live only in the hidden `meta` field;
the judge prompt uses instruction + response_a + response_b only."""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.utils import read_json, write_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="generation JSON of the model under test (e.g. DPO)")
    ap.add_argument("--baseline", required=True, help="generation JSON of the reference model (e.g. SFT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["general", "safety", "persona"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cand, base = read_json(args.candidate), read_json(args.baseline)
    base_by_id = {r["id"]: r for r in base}
    rng = random.Random(args.seed)
    rows, n_a = [], 0
    for c in cand:
        b = base_by_id.get(c["id"])
        if b is None:
            continue
        assert c["instruction"] == b["instruction"], c["id"]
        cand_is_a = rng.random() < 0.5
        n_a += cand_is_a
        a, bb = (c, b) if cand_is_a else (b, c)
        rows.append({
            "pair_id": f"{c['id']}",
            "instruction": c["instruction"],
            "category": c.get("category", c.get("dataset", "")),
            "response_a": a["output"],
            "response_b": bb["output"],
            "mode": args.mode,
            "meta": {"A": a["generator"], "B": bb["generator"], "candidate": c["generator"],
                     "baseline": b["generator"]},
        })
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} pairs, candidate in position A for {n_a}")


if __name__ == "__main__":
    main()
