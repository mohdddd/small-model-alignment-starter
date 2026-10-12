"""From-scratch SFT loop (no HF Trainer): response-only loss, fp16 autocast with
fp32 master weights + GradScaler, grad accumulation, clipping, cosine schedule
with warmup, validation loss, best + final checkpoints, metrics.jsonl.

Loss reduction: exact mean NLL over all response tokens in each effective batch
(each micro-batch's summed NLL is divided by the token count of its whole
accumulation group).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from pa2.core import IGNORE_INDEX, build_lm_batch, sequence_logprobs  # noqa: E402
from pa2.prompting import PROMPT_TEMPLATE, format_prompt  # noqa: E402
from pa2.utils import MetricLogger, peak_gpu_memory_gb, read_jsonl, set_seed, write_json  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_val_examples", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_hours", type=float, default=10.5, help="stop + save final before Kaggle's limit")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_save", action="store_true", help="smoke tests: skip checkpoints")
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_nll, total_tok = 0.0, 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logps, counts = sequence_logprobs(model, batch)
        total_nll += -logps.sum().item()
        total_tok += counts.sum().item()
    model.train()
    return total_nll / max(total_tok, 1)


def save(model, tok, path, meta):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    write_json(os.path.join(path, "checkpoint_meta.json"), meta)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    train = read_jsonl(args.train_file)[: args.max_train_examples]
    val = read_jsonl(args.val_file)[: args.max_val_examples]

    def collate(rows):
        return build_lm_batch(tok, [format_prompt(r["instruction"]) for r in rows],
                              [r["response"] for r in rows], args.max_length)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=gen,
                              collate_fn=collate, num_workers=2, drop_last=False)
    val_loader = DataLoader(val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    micro_per_epoch = len(train_loader)
    total_steps = math.ceil(micro_per_epoch / args.grad_accum) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(round(args.warmup_ratio * total_steps)))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in n.lower() else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = vars(args) | {"prompt_template": PROMPT_TEMPLATE, "n_train": len(train), "n_val": len(val),
                           "total_optimizer_steps": total_steps, "warmup_steps": warmup,
                           "effective_batch": args.batch_size * args.grad_accum, "precision": "fp16 autocast, fp32 weights",
                           "loss_reduction": "token mean over each effective batch", "optimizer": "AdamW"}
    write_json(os.path.join(args.out_dir, "config.json"), config)
    log = MetricLogger(args.out_dir)

    best_val, step, t0 = float("inf"), 0, time.time()
    val_loss = evaluate(model, val_loader, device)
    log.log(kind="eval", step=0, val_loss=val_loss, val_ppl=math.exp(min(val_loss, 50)))

    stop = False
    for epoch in range(args.epochs):
        it = iter(train_loader)
        remaining = micro_per_epoch
        while remaining > 0 and not stop:
            group = [next(it) for _ in range(min(args.grad_accum, remaining))]
            remaining -= len(group)
            n_tok = sum(int(b["labels"][:, 1:].ne(IGNORE_INDEX).sum()) for b in group)
            group_nll = 0.0
            for b in group:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logps, _ = sequence_logprobs(model, b)
                loss = -logps.sum() / max(n_tok, 1)
                scaler.scale(loss).backward()
                group_nll += loss.item()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                log.log(kind="train", step=step, epoch=epoch, train_loss=group_nll, grad_norm=grad_norm,
                        lr=sched.get_last_lr()[0], tokens=n_tok, scale=scaler.get_scale())
            hours = (time.time() - t0) / 3600
            last = step >= total_steps or (remaining == 0 and epoch == args.epochs - 1)
            out_of_time = hours > args.max_hours
            if step % args.eval_every == 0 or last or out_of_time:
                val_loss = evaluate(model, val_loader, device)
                is_best = val_loss < best_val
                log.log(kind="eval", step=step, val_loss=val_loss, val_ppl=math.exp(min(val_loss, 50)), best=is_best)
                if is_best:
                    best_val = val_loss
                    if not args.no_save:
                        save(model, tok, os.path.join(args.out_dir, "best"), {"step": step, "val_loss": val_loss})
            if step >= total_steps or out_of_time:
                stop = True

    if not args.no_save:
        save(model, tok, os.path.join(args.out_dir, "final"), {"step": step, "val_loss": val_loss})
    summary = {"best_val_loss": best_val, "final_val_loss": val_loss, "steps": step,
               "hours": (time.time() - t0) / 3600, "peak_gpu_mem_gb": peak_gpu_memory_gb()}
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
