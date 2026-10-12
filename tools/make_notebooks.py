"""Regenerate notebooks/*.ipynb (run: python tools/make_notebooks.py)."""

import json
import os

GH_USER, REPO = "mohdddd", "small-model-alignment-starter"

SETUP = f'''# ---- Setup: identical first cell in every notebook (a commit starts on a fresh machine) ----
GH_USER, REPO = "{GH_USER}", "{REPO}"
%cd /kaggle/working
!rm -rf repo && git clone -q https://github.com/{{GH_USER}}/{{REPO}}.git repo
%cd /kaggle/working/repo
!git log --oneline -1
!python -m pip uninstall -y -q torchao || true
!python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
!python -m pip install -q -e . --no-deps
import os, sys, json, glob, shutil, subprocess
sys.path.insert(0, "/kaggle/working/repo/src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("outputs", exist_ok=True)'''

ENV = '''!nvidia-smi --query-gpu=index,name,memory.total --format=csv
import torch, transformers, datasets
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| GPUs:", torch.cuda.device_count())'''

TESTS = "!PYTHONPATH=. python -m pytest -q tests/ 2>&1 | tail -3"

SHOW = '''from IPython.display import Image, display
def show(path):
    display(Image(path))'''


def disk():
    return "!du -sh /kaggle/working/repo/outputs/* 2>/dev/null; df -h /kaggle/working | tail -1"


_ids = iter(range(10**6))


def md(text):
    return {"cell_type": "markdown", "id": f"c{next(_ids):04d}", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "id": f"c{next(_ids):04d}", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


NOTEBOOKS = {}

NOTEBOOKS["00_dev_smoke"] = [
    md("# pa2-dev — setup checks + GPU smoke tests (INTERACTIVE, ~15 min)\n"
       "Settings: GPU T4 x2, Internet on. Run cells one by one. Nothing here is a submitted result.\n"
       "Purpose: catch CUDA / memory / path problems in minutes instead of after a 2-hour commit."),
    code(SETUP), code(ENV), code(TESTS),
    code("!python scripts/prepare_data.py --which sft hh alpaca --out_dir outputs/smoke_data "
         "--sft_train 64 --pref_train 64 --n_val 16"),
    md("SFT smoke: 3 optimizer steps at the real batch shape (2 x 512). Check `peak_gpu_mem_gb` < ~14."),
    code("!python scripts/train_sft.py --train_file outputs/smoke_data/sft_train.jsonl "
         "--val_file outputs/smoke_data/sft_val.jsonl --out_dir outputs/smoke_sft "
         "--max_steps 3 --grad_accum 4 --eval_every 3 --log_every 1 --no_save\n"
         "print(open('outputs/smoke_sft/summary.json').read())"),
    md("DPO smoke (base model stands in for the SFT checkpoint): policy cuda:0, reference cuda:1. "
       "Step-0 eval must show val_loss = 0.6931, reward_acc = 0.0, margin = 0.0."),
    code("!python scripts/train_dpo.py --sft_path Qwen/Qwen2.5-0.5B --train_file outputs/smoke_data/hh_train.jsonl "
         "--val_file outputs/smoke_data/hh_val.jsonl --out_dir outputs/smoke_dpo "
         "--max_steps 3 --grad_accum 4 --eval_every 3 --log_every 1 --no_save\n"
         "print(open('outputs/smoke_dpo/summary.json').read())"),
    md("Generation sanity check on 3 prompts (base model, greedy + repetition_penalty 1.1)."),
    code("!python scripts/generate.py --model Qwen/Qwen2.5-0.5B --eval_set outputs/smoke_data/alpaca_eval_300_seed42.json "
         "--out outputs/smoke_gen/base.json --generator qwen2.5-0.5b-base --limit 3 --max_new_tokens 200"),
    md("If all cells passed: **Quick Save**, stop this session (to free GPU quota), then commit `01_sft`."),
]

NOTEBOOKS["01_sft"] = [
    md("# pa2-sft — Task 2 baseline generation + Task 3 SFT\n"
       "Settings: GPU T4 x2, Internet on. Launch with **Save Version → Save & Run All (Commit)**. ~1.5–2 h.\n"
       "SFT trains on cuda:0 while base-model generations run on cuda:1 in the background."),
    code(SETUP), code(ENV), code(TESTS),
    code("!python scripts/prepare_data.py --which sft alpaca --out_dir outputs/data\n"
         "print(open('outputs/data/data_stats.json').read())"),
    code('''# Task 2: base-model generations on cuda:1 (background) while SFT uses cuda:0
os.makedirs("outputs/generations", exist_ok=True)
BASE_CMD = (
    "python scripts/generate.py --model Qwen/Qwen2.5-0.5B --eval_set outputs/data/alpaca_eval_300_seed42.json "
    "--out outputs/generations/alpaca_base.json --generator qwen2.5-0.5b-base --device cuda:1 && "
    "python scripts/generate.py --model Qwen/Qwen2.5-0.5B --eval_set eval_sets/hh_rlhf_eval_300_seed42.json "
    "--out outputs/generations/hh_base.json --generator qwen2.5-0.5b-base --device cuda:1")
base_log = open("outputs/base_generation.log", "w")
base_proc = subprocess.Popen(BASE_CMD, shell=True, stdout=base_log, stderr=subprocess.STDOUT)
print("base generation started, pid", base_proc.pid)'''),
    code("!python scripts/train_sft.py --train_file outputs/data/sft_train.jsonl --val_file outputs/data/sft_val.jsonl "
         "--out_dir outputs/sft --device cuda:0"),
    code('''rc = base_proc.wait(); base_log.close()
print("base generation exit code:", rc)
print(open("outputs/base_generation.log").read()[-2500:])'''),
    code(SHOW + "\n!python scripts/plot_metrics.py --run_dir outputs/sft --kind sft\nshow('outputs/sft/sft_curves.png')\n"
         "print(open('outputs/sft/summary.json').read())"),
    md("Sanity check: SFT best checkpoint on 3 AlpacaEval prompts (checks path, tokenizer, template, EOS, length)."),
    code("!python scripts/generate.py --model outputs/sft/best --eval_set outputs/data/alpaca_eval_300_seed42.json "
         "--out outputs/generations/alpaca_sft.json --generator qwen2.5-0.5b-sft --limit 3 --device cuda:1"),
    code("!cd outputs && zip -qr /kaggle/working/sft_run_outputs.zip sft/metrics.jsonl sft/config.json sft/summary.json "
         "sft/sft_curves.png sft/best/checkpoint_meta.json data/data_stats.json data/alpaca_eval_300_seed42.json "
         "generations base_generation.log\n" + disk()),
]


def dpo_nb(tag, title, which):
    return [
        md(f"# pa2-dpo-{tag} — Task 4: from-scratch DPO on {title}\n"
           "Settings: GPU T4 x2, Internet on. **Add Input → Notebook Output → pa2-sft** (latest version).\n"
           "Launch with **Save Version → Save & Run All (Commit)**. ~2–2.5 h. Policy on cuda:0, frozen reference on cuda:1."),
        code(SETUP), code(ENV),
        code('''from pa2.utils import find_one
SFT = os.path.dirname(find_one("sft/best/model.safetensors"))
print("SFT best checkpoint:", SFT)
print(open(os.path.join(SFT, "checkpoint_meta.json")).read())'''),
        code(f"!python scripts/prepare_data.py --which {which} --out_dir outputs/data\n"
             "print(open('outputs/data/data_stats.json').read())"),
        code(f"!python scripts/train_dpo.py --sft_path {{SFT}} --train_file outputs/data/{which}_train.jsonl "
             f"--val_file outputs/data/{which}_val.jsonl --out_dir outputs/dpo_{tag}"),
        code(SHOW + f"\n!python scripts/plot_metrics.py --run_dir outputs/dpo_{tag} --kind dpo --title \"{title} DPO\"\n"
             f"show('outputs/dpo_{tag}/dpo_curves.png')\nprint(open('outputs/dpo_{tag}/summary.json').read())"),
        code(f"!cd outputs && zip -qr /kaggle/working/dpo_{tag}_run_outputs.zip dpo_{tag}/metrics.jsonl dpo_{tag}/config.json "
             f"dpo_{tag}/summary.json dpo_{tag}/dpo_curves.png dpo_{tag}/best/checkpoint_meta.json data/data_stats.json\n" + disk()),
    ]


NOTEBOOKS["02_dpo_hh"] = dpo_nb("hh", "HH-RLHF", "hh")
NOTEBOOKS["03_dpo_uf"] = dpo_nb("uf", "UltraFeedback", "uf")

NOTEBOOKS["04_lora_persona"] = [
    md("# pa2-lora-persona — Task 5: TRL DPOTrainer + LoRA on SmolLM2-360M-Instruct\n"
       "Settings: GPU T4 x2 (uses one), Internet on. No inputs needed. "
       "Launch with **Save Version → Save & Run All (Commit)**. ~10–20 min."),
    code(SETUP),
    code('!python -m pip install -q "trl==0.24.0" "peft>=0.13" "transformers>=4.43.0,<5"\n'
         "import trl, peft; print('trl', trl.__version__, '| peft', peft.__version__)"),
    code(ENV),
    code("!python scripts/train_lora_dpo.py --out_dir outputs/lora_persona"),
    code(SHOW + "\n!python scripts/plot_metrics.py --run_dir outputs/lora_persona --kind lora\n"
         "show('outputs/lora_persona/lora_curves.png')"),
    code("!cd outputs && zip -qr /kaggle/working/lora_persona_outputs.zip lora_persona/metrics.jsonl lora_persona/config.json "
         "lora_persona/summary.json lora_persona/lora_curves.png lora_persona/adapter\n" + disk()),
]

NOTEBOOKS["05_generate"] = [
    md("# pa2-generate — Task 6: evaluation generations + blinded pairwise files\n"
       "Settings: GPU T4 x2, Internet on. **Add Input → Notebook Output**: pa2-sft, pa2-dpo-hh, pa2-dpo-uf, pa2-lora-persona.\n"
       "Run the sanity cell interactively first if you like, then **Save & Run All (Commit)**. ~30–60 min.\n"
       "Decoding (identical for every model in a comparison): greedy, repetition_penalty 1.1, max_new_tokens 512."),
    code(SETUP),
    code('!python -m pip install -q "peft>=0.13"'),
    code(ENV),
    code('''from pa2.utils import find_one
d = os.path.dirname
P = {
    "sft": d(find_one("sft/best/model.safetensors")),
    "dpo_hh": d(find_one("dpo_hh/best/model.safetensors")),
    "dpo_uf": d(find_one("dpo_uf/best/model.safetensors")),
    "lora": d(find_one("lora_persona/adapter/adapter_config.json")),
}
ALPACA = "outputs/data/alpaca_eval_300_seed42.json"
HH, PERSONA = "eval_sets/hh_rlhf_eval_300_seed42.json", "eval_sets/persona_style_eval_24.json"
SMOL = "HuggingFaceTB/SmolLM2-360M-Instruct"
os.makedirs("outputs/data", exist_ok=True); os.makedirs("outputs/generations", exist_ok=True)
shutil.copy(find_one("data/alpaca_eval_300_seed42.json"), ALPACA)
for f in ("alpaca_base.json", "hh_base.json"):
    shutil.copy(find_one("generations/" + f), "outputs/generations/" + f)
print(json.dumps(P, indent=1))'''),
    code('''def gen(model, eval_set, out, name, device, extra=""):
    return (f"python scripts/generate.py --model {model} --eval_set {eval_set} --out outputs/generations/{out}.json "
            f"--generator {name} --device {device} {extra}")

G0 = [gen(P["sft"], ALPACA, "alpaca_sft", "qwen2.5-0.5b-sft", "cuda:0"),
      gen(P["dpo_uf"], ALPACA, "alpaca_dpo_uf", "qwen2.5-0.5b-dpo-uf", "cuda:0"),
      gen(SMOL, PERSONA, "persona_smol_base", "smollm2-360m-instruct", "cuda:0", "--template chat")]
G1 = [gen(P["sft"], HH, "hh_sft", "qwen2.5-0.5b-sft", "cuda:1"),
      gen(P["dpo_hh"], HH, "hh_dpo_hh", "qwen2.5-0.5b-dpo-hh", "cuda:1"),
      gen(SMOL, PERSONA, "persona_lora", "smollm2-360m-lora-dpo", "cuda:1", f"--template chat --adapter {P['lora']}")]
RUN_CROSS = False  # optional alignment-tax check: HH-DPO on AlpacaEval, UF-DPO on the HH set (+~15 min)
if RUN_CROSS:
    G0.append(gen(P["dpo_hh"], ALPACA, "alpaca_dpo_hh", "qwen2.5-0.5b-dpo-hh", "cuda:0"))
    G1.append(gen(P["dpo_uf"], HH, "hh_dpo_uf", "qwen2.5-0.5b-dpo-uf", "cuda:1"))'''),
    md("Sanity check (2–5 prompts per model) — read the outputs before the full run."),
    code('''for c in [G0[0], G0[1], G1[1], G1[2]]:
    !{c} --limit 3 --max_new_tokens 256 2>&1 | grep -v Warning | tail -12'''),
    md("Full generation: two queues in parallel (cuda:0 and cuda:1). Each file is written as soon as its model finishes; "
       "existing files are skipped, so a rerun resumes."),
    code('''procs = []
for i, chain in enumerate([G0, G1]):
    log = open(f"outputs/generation_gpu{i}.log", "w")
    procs.append((subprocess.Popen(" && ".join(chain), shell=True, stdout=log, stderr=subprocess.STDOUT), log))
for p, log in procs:
    print("exit code", p.wait()); log.close()
for i in range(2):
    print(f"--- gpu{i} log tail ---"); print(open(f"outputs/generation_gpu{i}.log").read()[-1500:])'''),
    code('''# Blinded pairwise files (random A/B order, seed 42; identities only in hidden meta)
!python scripts/build_pairs.py --candidate outputs/generations/alpaca_dpo_uf.json --baseline outputs/generations/alpaca_sft.json --out outputs/pairs/uf_dpo_vs_sft_alpaca.jsonl --mode general
!python scripts/build_pairs.py --candidate outputs/generations/hh_dpo_hh.json --baseline outputs/generations/hh_sft.json --out outputs/pairs/hh_dpo_vs_sft_hh.jsonl --mode safety
!python scripts/build_pairs.py --candidate outputs/generations/persona_lora.json --baseline outputs/generations/persona_smol_base.json --out outputs/pairs/lora_vs_base_persona.jsonl --mode persona'''),
    code('''# Length / truncation stats per generation file (for the verbosity discussion in the report)
import statistics
for f in sorted(glob.glob("outputs/generations/*.json")):
    if f.endswith(".settings.json") or ".sanity" in f: continue
    rows = json.load(open(f))
    words = [len(r["output"].split()) for r in rows]
    print(f"{os.path.basename(f):28s} n={len(rows):3d}  mean_words={statistics.mean(words):6.1f}  "
          f"median={statistics.median(words):5.0f}  hit_max_tokens={sum(r['hit_max_new_tokens'] for r in rows):3d}  "
          f"empty={sum(not r['output'] for r in rows)}")'''),
    code("!cd outputs && zip -qr /kaggle/working/generation_outputs.zip generations pairs data *.log\n" + disk()),
]

NOTEBOOKS["06_judge"] = [
    md("# pa2-judge — Task 7: pairwise judging with Kaggle model proxy\n"
       "Create this notebook as a **Kaggle Benchmarks** notebook (so `kaggle_benchmarks` + the model proxy are available). "
       "Accelerator: None. Internet on. **Add Input → Notebook Output → pa2-generate.**\n"
       "Batches of 5, ≥5 s sleep before every request, 3 retries, judgments checkpointed after every batch, failures "
       "logged, `--reuse_judgments` resumes. Run the smoke cell interactively first, then **Save & Run All (Commit)**."),
    code(f'''GH_USER, REPO = "{GH_USER}", "{REPO}"
%cd /kaggle/working
!rm -rf repo && git clone -q https://github.com/{{GH_USER}}/{{REPO}}.git repo
%cd /kaggle/working/repo
!git log --oneline -1
import os, sys, json, shutil
sys.path.insert(0, "/kaggle/working/repo/src"); sys.path.insert(0, "/kaggle/working/repo/scripts")
from pa2.utils import find_one
import judge_pairwise_kaggle as J
os.makedirs("outputs/pairs", exist_ok=True); os.makedirs("outputs/eval", exist_ok=True)
PAIRS = ["uf_dpo_vs_sft_alpaca", "hh_dpo_vs_sft_hh", "lora_vs_base_persona"]
for name in PAIRS:
    shutil.copy(find_one(f"pairs/{{name}}.jsonl"), f"outputs/pairs/{{name}}.jsonl")
JUDGE = "google/gemini-3.6-flash"   # change if not available; record the one you use in the report'''),
    code('''import kaggle_benchmarks as kbench
try:
    print(sorted(kbench.llms.keys()))
except Exception as e:
    print("could not list models:", e, kbench.llms)'''),
    md("Smoke test: one batch of 5 pairs. Check that the output parses and A/B maps to model names."),
    code('''s = J.main(["--pairs_path", "outputs/pairs/uf_dpo_vs_sft_alpaca.jsonl", "--output_path", "outputs/eval/smoke_judged.jsonl",
            "--judge_model", JUDGE, "--limit", "5", "--batch_size", "5", "--sleep_seconds", "5", "--max_retries", "3"])
!head -n 2 outputs/eval/smoke_judged.jsonl'''),
    md("Full judging (~125 requests, ~30–45 min). Safe to re-run: completed pairs are skipped."),
    code('''summaries = {}
for name in PAIRS:
    summaries[name] = J.main(["--pairs_path", f"outputs/pairs/{name}.jsonl", "--output_path", f"outputs/eval/{name}_judged.jsonl",
                              "--judge_model", JUDGE, "--batch_size", "5", "--sleep_seconds", "5", "--max_retries", "3",
                              "--reuse_judgments"])'''),
    code('''for name, s in summaries.items():
    o = s["overall"]
    print(f"\\n== {name}: {s['candidate']} vs {s['baseline']} | judge {s['judge_model']}")
    print(f"   judged {s['n_judged']}/{s['n_pairs']}  failed {s['n_failed_unjudged']}  "
          f"W/L/T/U = {o['wins']}/{o['losses']}/{o['ties']}/{o['unknown']}")
    print(f"   win rate excl ties {o['win_rate_excl_ties']}, ties as half {o['win_rate_ties_half']}")
    print("   position counts (bias check):", s["position_counts"])
    if "persona" in s:
        print("   persona criteria:", {k: (v["wins"], v["losses"], v["ties"]) for k, v in s["persona"].items()})
    for cat, c in s["by_category"].items():
        print(f"     {cat:28s} W/L/T {c['wins']}/{c['losses']}/{c['ties']}")'''),
    code("!cd outputs && zip -qr /kaggle/working/judge_outputs.zip eval pairs && ls -la eval"),
]


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
    os.makedirs(out_dir, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        with open(os.path.join(out_dir, f"{name}.ipynb"), "w") as f:
            json.dump(notebook(cells), f, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main()
