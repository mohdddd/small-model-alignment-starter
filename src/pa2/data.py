"""Dataset builders. Every builder: shuffle with a fixed seed, keep single-turn
examples, drop empty/identical/over-long items, take the first N survivors.

Prompt length filtering uses the *formatted* prompt (same template as training),
so no training prompt is ever truncated.
"""

from __future__ import annotations

import random

from .prompting import format_prompt

HH_HUMAN = "\n\nHuman:"
HH_ASSISTANT = "\n\nAssistant:"


def _prompt_len(tokenizer, instruction):
    return len(tokenizer.encode(format_prompt(instruction), add_special_tokens=False))


def _take(rows_iter, n, stats):
    out = []
    for row in rows_iter:
        stats["seen"] += 1
        if row is None:
            continue
        out.append(row)
        if len(out) >= n:
            break
    stats["kept"] = len(out)
    return out


# ------------------------------- SFT --------------------------------------- #
def ultrachat_rows(ds, tokenizer, max_prompt_tokens, split_tag, stats):
    for i, ex in enumerate(ds):
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            stats["bad_roles"] = stats.get("bad_roles", 0) + 1
            yield None
            continue
        instr, resp = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not instr or not resp:
            yield None
            continue
        if _prompt_len(tokenizer, instr) > max_prompt_tokens:
            stats["prompt_too_long"] = stats.get("prompt_too_long", 0) + 1
            yield None
            continue
        yield {"id": f"ultrachat-{split_tag}-{ex.get('prompt_id', i)}", "instruction": instr, "response": resp}


# ------------------------------- HH-RLHF ----------------------------------- #
def parse_hh(chosen_text, rejected_text):
    """Return (instruction, chosen, rejected) for single-turn pairs, else None."""
    for t in (chosen_text, rejected_text):
        if t.count(HH_HUMAN) != 1 or t.count(HH_ASSISTANT) != 1:
            return None
    c_idx, r_idx = chosen_text.rfind(HH_ASSISTANT), rejected_text.rfind(HH_ASSISTANT)
    c_prompt, r_prompt = chosen_text[:c_idx], rejected_text[:r_idx]
    if c_prompt.strip() != r_prompt.strip():
        return None
    instruction = c_prompt.replace(HH_HUMAN, "", 1).strip()
    chosen = chosen_text[c_idx + len(HH_ASSISTANT):].strip()
    rejected = rejected_text[r_idx + len(HH_ASSISTANT):].strip()
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return instruction, chosen, rejected


def hh_rows(ds, tokenizer, max_prompt_tokens, split_tag, stats):
    for i, ex in enumerate(ds):
        parsed = parse_hh(ex["chosen"], ex["rejected"])
        if parsed is None:
            stats["multi_turn_or_invalid"] = stats.get("multi_turn_or_invalid", 0) + 1
            yield None
            continue
        instr, chosen, rejected = parsed
        if _prompt_len(tokenizer, instr) > max_prompt_tokens:
            stats["prompt_too_long"] = stats.get("prompt_too_long", 0) + 1
            yield None
            continue
        yield {"id": f"hh-{split_tag}-{i}", "instruction": instr, "chosen": chosen, "rejected": rejected}


# ----------------------------- UltraFeedback ------------------------------- #
def uf_rows(ds, tokenizer, max_prompt_tokens, split_tag, stats):
    for i, ex in enumerate(ds):
        ch, rj = ex["chosen"], ex["rejected"]
        ok = (len(ch) == 2 and len(rj) == 2 and ch[0]["role"] == "user"
              and ch[1]["role"] == "assistant" and rj[1]["role"] == "assistant")
        if not ok:
            stats["multi_turn_or_invalid"] = stats.get("multi_turn_or_invalid", 0) + 1
            yield None
            continue
        instr = (ex.get("prompt") or ch[0]["content"]).strip()
        chosen, rejected = ch[1]["content"].strip(), rj[1]["content"].strip()
        if not instr or not chosen or not rejected or chosen == rejected:
            yield None
            continue
        if ex.get("score_chosen") is not None and ex["score_chosen"] <= ex["score_rejected"]:
            stats["score_tie"] = stats.get("score_tie", 0) + 1
            yield None
            continue
        if _prompt_len(tokenizer, instr) > max_prompt_tokens:
            stats["prompt_too_long"] = stats.get("prompt_too_long", 0) + 1
            yield None
            continue
        yield {"id": f"uf-{split_tag}-{ex.get('prompt_id', i)}", "instruction": instr,
               "chosen": chosen, "rejected": rejected}


def build_split(ds, row_fn, tokenizer, n, max_prompt_tokens, seed, split_tag):
    stats = {"source_rows": len(ds), "seen": 0}
    ds = ds.shuffle(seed=seed)
    rows = _take(row_fn(ds, tokenizer, max_prompt_tokens, split_tag, stats), n, stats)
    if len(rows) < n:
        print(f"[warn] {split_tag}: only {len(rows)} rows available (wanted {n})")
    return rows, stats


# ------------------------------ AlpacaEval --------------------------------- #
def alpaca_subset(records, n=300, seed=42):
    """random.Random(seed).sample over all prompts, kept in original order."""
    idx = sorted(random.Random(seed).sample(range(len(records)), n))
    return [{"id": f"alpaca-{i:04d}", "instruction": records[i]["instruction"],
             "dataset": "alpaca_eval", "source_subset": records[i].get("dataset", "")} for i in idx]
