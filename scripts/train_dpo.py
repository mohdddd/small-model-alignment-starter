"""From-scratch DPO (no TRL): trainable policy (fp32 weights, fp16 autocast,
GradScaler) on --policy_device, frozen reference on --ref_device under no_grad.
Tracks validation DPO loss, reward accuracy (strict r_w > r_l) and reward
margin; keeps the checkpoint with the best validation reward accuracy (ties
broken by higher margin) plus the final checkpoint.
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
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from pa2.core import _pad_id, build_pair_batch, dpo_logits, dpo_rewards, pair_logprobs  # noqa: E402
from pa2.prompting import PROMPT_TEMPLATE, format_prompt  # noqa: E402
from pa2.utils import MetricLogger, peak_gpu_memory_gb, read_jsonl, set_seed, write_json  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_path", required=True, help="SFT best checkpoint (policy init + reference)")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=0, help="0 = fixed LR (guide default)")
    ap.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1, help="pairs per micro-batch")
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_batch_size", type=int, default=8, help="pairs")
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_val_examples", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_hours", type=float, default=10.5)
    ap.add_argument("--policy_device", default="cuda:0")
    ap.add_argument("--ref_device", default="cuda:1")
    ap.add_argument("--ref_dtype", choices=["float32", "float16"], default="float32",
                    help="float32 weights + fp16 autocast matches the policy's initial numerics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_save", action="store_true")
    return ap.parse_args(argv)


def resolve_devices(args):
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    pol = torch.device(args.policy_device)
    ref = torch.device(args.ref_device) if torch.cuda.device_count() > 1 else pol
    return pol, ref


def logps_for(model, pair_batch, pad_id, amp):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        return pair_logprobs(model, pair_batch, pad_id)


@torch.no_grad()
def reference_val_logps(ref, loader, pad_id, amp):
    out_c, out_r = [], []
    for pb in loader:
        c, r = logps_for(ref, pb, pad_id, amp)
        out_c.append(c.float().cpu())
        out_r.append(r.float().cpu())
    return torch.cat(out_c), torch.cat(out_r)


@torch.no_grad()
def evaluate(policy, loader, ref_c_all, ref_r_all, beta, pad_id, amp):
    policy.eval()
    pc_all, pr_all = [], []
    for pb in loader:
        c, r = logps_for(policy, pb, pad_id, amp)
        pc_all.append(c.float().cpu())
        pr_all.append(r.float().cpu())
    policy.train()
    pc, pr = torch.cat(pc_all), torch.cat(pr_all)
    logits = dpo_logits(pc, pr, ref_c_all, ref_r_all, beta)
    r_w, r_l = dpo_rewards(pc, pr, ref_c_all, ref_r_all, beta)
    return {
        "val_loss": (-F.logsigmoid(logits)).mean().item(),
        "val_reward_acc": (r_w > r_l).float().mean().item(),
        "val_reward_margin": (r_w - r_l).mean().item(),
        "val_chosen_reward": r_w.mean().item(),
        "val_rejected_reward": r_l.mean().item(),
        "val_policy_chosen_logp": pc.mean().item(),
        "val_policy_rejected_logp": pr.mean().item(),
    }


def save(model, tok, path, meta):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    write_json(os.path.join(path, "checkpoint_meta.json"), meta)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    pol_dev, ref_dev = resolve_devices(args)
    amp = pol_dev.type == "cuda"

    tok = AutoTokenizer.from_pretrained(args.sft_path)
    pad_id = _pad_id(tok)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32).to(pol_dev)
    policy.config.use_cache = False
    policy.train()
    ref_dtype = torch.float16 if args.ref_dtype == "float16" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=ref_dtype).to(ref_dev)
    reference.eval()
    reference.requires_grad_(False)

    def prompt_ok(r):
        return len(tok.encode(format_prompt(r["instruction"]), add_special_tokens=False)) <= args.max_prompt_length

    train = [r for r in read_jsonl(args.train_file)[: args.max_train_examples] if prompt_ok(r)]
    val = [r for r in read_jsonl(args.val_file)[: args.max_val_examples] if prompt_ok(r)]

    def collate(rows):
        return build_pair_batch(tok, [format_prompt(r["instruction"]) for r in rows],
                                [r["chosen"] for r in rows], [r["rejected"] for r in rows], args.max_length)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=gen,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    micro_per_epoch = len(train_loader)
    total_steps = math.ceil(micro_per_epoch / args.grad_accum) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    params = [p for p in policy.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) if args.warmup_steps > 0 else 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    config = vars(args) | {"prompt_template": PROMPT_TEMPLATE, "n_train": len(train), "n_val": len(val),
                           "total_optimizer_steps": total_steps, "effective_batch": args.batch_size * args.grad_accum,
                           "policy_device": str(pol_dev), "ref_device": str(ref_dev),
                           "precision": "policy fp32 weights + fp16 autocast; reference no_grad",
                           "reference_checkpoint": os.path.abspath(args.sft_path)}
    write_json(os.path.join(args.out_dir, "config.json"), config)
    log = MetricLogger(args.out_dir)

    t0 = time.time()
    ref_c_val, ref_r_val = reference_val_logps(reference, val_loader, pad_id, amp)
    log.log(kind="info", msg="reference val logps cached", n_val=len(val), seconds=round(time.time() - t0, 1))
    metrics = evaluate(policy, val_loader, ref_c_val, ref_r_val, args.beta, pad_id, amp)
    log.log(kind="eval", step=0, **metrics)
    best = (-1.0, -float("inf"))
    step, stop = 0, False

    for epoch in range(args.epochs):
        it = iter(train_loader)
        remaining = micro_per_epoch
        while remaining > 0 and not stop:
            group = [next(it) for _ in range(min(args.grad_accum, remaining))]
            remaining -= len(group)
            n_pairs = sum(pb["chosen"]["input_ids"].shape[0] for pb in group)
            g_loss, g_acc, g_margin = 0.0, 0.0, 0.0
            for pb in group:
                with torch.no_grad():  # reference first so its GPU work overlaps the policy pass
                    ref_c, ref_r = logps_for(reference, pb, pad_id, amp)
                pol_c, pol_r = logps_for(policy, pb, pad_id, amp)
                ref_c, ref_r = ref_c.float().to(pol_dev), ref_r.float().to(pol_dev)
                logits = dpo_logits(pol_c, pol_r, ref_c, ref_r, args.beta)
                losses = -F.logsigmoid(logits)
                loss = losses.sum() / n_pairs  # exact mean over the effective batch
                scaler.scale(loss).backward()
                r_w, r_l = dpo_rewards(pol_c, pol_r, ref_c, ref_r, args.beta)
                g_loss += losses.detach().sum().item()
                g_acc += (r_w > r_l).float().sum().item()
                g_margin += (r_w - r_l).sum().item()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                log.log(kind="train", step=step, epoch=epoch, train_loss=g_loss / n_pairs,
                        train_reward_acc=g_acc / n_pairs, train_reward_margin=g_margin / n_pairs,
                        grad_norm=grad_norm, lr=sched.get_last_lr()[0], scale=scaler.get_scale())
            hours = (time.time() - t0) / 3600
            out_of_time = hours > args.max_hours
            last = step >= total_steps or (remaining == 0 and epoch == args.epochs - 1)
            if step % args.eval_every == 0 or last or out_of_time:
                metrics = evaluate(policy, val_loader, ref_c_val, ref_r_val, args.beta, pad_id, amp)
                key = (metrics["val_reward_acc"], metrics["val_reward_margin"])
                is_best = key > best
                log.log(kind="eval", step=step, best=is_best, **metrics)
                if is_best:
                    best = key
                    if not args.no_save:
                        save(policy, tok, os.path.join(args.out_dir, "best"), {"step": step, **metrics})
            if step >= total_steps or out_of_time:
                stop = True

    if not args.no_save:
        save(policy, tok, os.path.join(args.out_dir, "final"), {"step": step, **metrics})
    summary = {"best_val_reward_acc": best[0], "best_val_reward_margin": best[1], "final": metrics,
               "steps": step, "hours": (time.time() - t0) / 3600, "peak_gpu_mem_gb": peak_gpu_memory_gb()}
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
