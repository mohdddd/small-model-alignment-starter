"""Task 5: TRL DPOTrainer + PEFT/LoRA persona tuning of SmolLM2-360M-Instruct.

Rows {prompt, chosen, rejected} are converted to TRL's conversational format:
  prompt   = [{"role": "user", "content": ...}]
  chosen   = [{"role": "assistant", "content": ...}]
  rejected = [{"role": "assistant", "content": ...}]
With a peft_config and no ref_model, TRL uses the base model with the adapter
disabled as the frozen reference.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # single GPU: avoid HF Trainer DataParallel
os.environ.setdefault("WANDB_DISABLED", "true")

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

from pa2.utils import append_jsonl, read_json, set_seed, write_json  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def to_conversational(row):
    return {"prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}]}


def filtered_kwargs(cls, kwargs):
    """Keep only kwargs this TRL/transformers version accepts (API drift across versions)."""
    params = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in params)
    if dropped:
        print(f"[lora] {cls.__name__} ignores unsupported args: {dropped}")
    return {k: v for k, v in kwargs.items() if k in params}


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    cuda = torch.cuda.is_available()

    rows = read_json(args.data)
    random.Random(args.seed).shuffle(rows)
    eval_rows, train_rows = rows[: args.n_eval], rows[args.n_eval:]
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    eval_ds = Dataset.from_list([to_conversational(r) for r in eval_rows])

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=args.target_modules.split(","), bias="none", task_type="CAUSAL_LM")
    steps_per_epoch = -(-len(train_rows) // (args.batch_size * args.grad_accum))
    cfg_kwargs = dict(
        output_dir=args.out_dir, beta=args.beta, learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio,
        logging_steps=2, eval_strategy="steps", evaluation_strategy="steps", eval_steps=max(1, steps_per_epoch // 2),
        save_strategy="no", fp16=cuda, bf16=False, max_length=args.max_length, max_prompt_length=256,
        report_to="none", seed=args.seed, remove_unused_columns=False, gradient_checkpointing=False,
        use_cpu=not cuda,
    )
    cfg = DPOConfig(**filtered_kwargs(DPOConfig, cfg_kwargs))

    trainer_kwargs = dict(model=model, ref_model=None, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                          processing_class=tok, tokenizer=tok, peft_config=peft_config)
    trainer = DPOTrainer(**filtered_kwargs(DPOTrainer, trainer_kwargs))
    trainer.model.print_trainable_parameters()

    first_eval = trainer.evaluate()
    append_jsonl(os.path.join(args.out_dir, "metrics.jsonl"), {"step": 0, **first_eval})
    trainer.train()
    for rec in trainer.state.log_history:
        append_jsonl(os.path.join(args.out_dir, "metrics.jsonl"), rec)

    adapter_dir = os.path.join(args.out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    import trl
    write_json(os.path.join(args.out_dir, "config.json"), vars(args) | {
        "n_train": len(train_rows), "n_eval": len(eval_rows), "trainable_params": n_trainable,
        "total_params": n_total, "trl_version": trl.__version__,
        "effective_batch": args.batch_size * args.grad_accum, "precision": "fp16 AMP" if cuda else "fp32 cpu",
        "dpo_config_used": {k: v for k, v in cfg.to_dict().items() if k in cfg_kwargs}})
    final = trainer.evaluate()
    write_json(os.path.join(args.out_dir, "summary.json"), {"initial_eval": first_eval, "final_eval": final})
    print(json.dumps({"initial_eval": first_eval, "final_eval": final}, indent=1, default=str))


if __name__ == "__main__":
    main()
