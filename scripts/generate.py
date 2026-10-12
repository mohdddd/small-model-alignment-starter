"""Batched deterministic generation on a fixed eval set.

Writes a JSON list of {id, instruction, output, generator, dataset, ...}.
Skips work if --out already exists (restartable). Use --limit for sanity checks.

Templates:
  alpaca : the plain-text template used for Qwen SFT/DPO training (also base model)
  chat   : tokenizer.apply_chat_template (SmolLM2-Instruct / LoRA persona model)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from pa2.prompting import clean_response, format_prompt  # noqa: E402
from pa2.utils import read_json, write_json  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint")
    ap.add_argument("--adapter", default=None, help="optional PEFT/LoRA adapter dir")
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generator", required=True, help="name stored in each record")
    ap.add_argument("--template", choices=["alpaca", "chat"], default="alpaca")
    ap.add_argument("--system_prompt", default=None, help="chat template only")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def build_prompt(tok, instruction, template, system_prompt):
    if template == "alpaca":
        return format_prompt(instruction)
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
        [{"role": "user", "content": instruction}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(args.out) and not args.overwrite and args.limit is None:
        print("exists, skipping:", args.out)
        return
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(device).eval()

    items = read_json(args.eval_set)[: args.limit]
    prompts = [build_prompt(tok, it["instruction"], args.template, args.system_prompt) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))  # length-sorted batches
    outputs, n_new, t0 = [None] * len(items), [0] * len(items), time.time()
    gen_kwargs = dict(do_sample=False, max_new_tokens=args.max_new_tokens,
                      repetition_penalty=args.repetition_penalty, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id)
    if args.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    for b in range(0, len(order), args.batch_size):
        idx = order[b: b + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            ids = new[j].tolist()
            stop = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else None
            n_new[i] = (stop + 1) if stop is not None else len(ids)  # len == max_new_tokens -> no EOS
            text = tok.decode(ids[:stop] if stop is not None else ids, skip_special_tokens=True)
            outputs[i] = clean_response(text) if args.template == "alpaca" else text.strip()
        print(f"[{args.generator}] {min(b + args.batch_size, len(order))}/{len(order)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    records = []
    for i, it in enumerate(items):
        rec = {"id": it["id"], "instruction": it["instruction"], "output": outputs[i],
               "generator": args.generator, "dataset": it.get("dataset", os.path.basename(args.eval_set)),
               "new_tokens": n_new[i], "hit_max_new_tokens": n_new[i] >= args.max_new_tokens}
        for k in ("category", "expected_behavior"):
            if k in it:
                rec[k] = it[k]
        records.append(rec)

    settings = {k: v for k, v in gen_kwargs.items() if k not in ("eos_token_id", "pad_token_id")}
    settings.update(template=args.template, system_prompt=args.system_prompt, model=args.model, adapter=args.adapter)
    out_path = args.out if args.limit is None else args.out.replace(".json", f".sanity{args.limit}.json")
    write_json(out_path, records)
    write_json(out_path.replace(".json", ".settings.json"), settings)
    for r in records[: min(3, len(records))]:
        print("=" * 80, "\nQ:", r["instruction"][:200], "\nA:", r["output"][:600])
    print("wrote", out_path, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
