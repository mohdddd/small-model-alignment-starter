"""Core SFT / DPO math shared by the training scripts and submission_adapter.py.

Conventions (documented in the report):
- Sequence = prompt tokens + response tokens + EOS (if the tokenizer has one).
  EOS is appended BEFORE truncation to ``max_length`` (right truncation), so a
  truncated response may lose its EOS.
- Labels are -100 on prompt and padding positions; response (+EOS) tokens keep
  their ids. Padding is on the right, to the longest row in the batch.
- Padding/prompt positions are identified by position, never by token id, so a
  tokenizer whose pad token equals its EOS token (Qwen2.5) is handled correctly.
- Response log-prob = sum over unmasked label positions of log p(token | prefix).
- SFT loss reduction = mean over all unmasked response tokens in the batch.
- DPO loss = mean_i -log sigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r))).
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Tokenization / batching
# --------------------------------------------------------------------------- #
def _encode(tokenizer, text: str):
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer) -> int:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    return 0


def tokenize_example(tokenizer, prompt: str, response: str, max_length: int):
    """Return (input_ids, labels) lists for one prompt/response pair."""
    prompt_ids = _encode(tokenizer, prompt)
    response_ids = _encode(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def pad_rows(rows, pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad a list of (input_ids, labels) to the longest row."""
    width = max(1, max(len(ids) for ids, _ in rows))
    input_ids, attention_mask, labels = [], [], []
    for ids, lab in rows:
        n_pad = width - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels.append(lab + [IGNORE_INDEX] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_lm_batch(tokenizer, prompts: Sequence[str], responses: Sequence[str], max_length: int):
    """Padded LM batch with response-only labels; row order is preserved."""
    prompts, responses = list(prompts), list(responses)
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    rows = [tokenize_example(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    return pad_rows(rows, _pad_id(tokenizer))


def build_pair_batch(tokenizer, prompts, chosen, rejected, max_length):
    prompts, chosen, rejected = list(prompts), list(chosen), list(rejected)
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen and rejected must have the same length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen, max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected, max_length),
    }


def concat_pair_batch(pair_batch, pad_id: int) -> Dict[str, torch.Tensor]:
    """Stack chosen rows then rejected rows into one batch (one forward pass)."""
    c, r = pair_batch["chosen"], pair_batch["rejected"]
    width = max(c["input_ids"].shape[1], r["input_ids"].shape[1])

    def _pad(t, value):
        return F.pad(t, (0, width - t.shape[1]), value=value)

    return {
        "input_ids": torch.cat([_pad(c["input_ids"], pad_id), _pad(r["input_ids"], pad_id)]),
        "attention_mask": torch.cat([_pad(c["attention_mask"], 0), _pad(r["attention_mask"], 0)]),
        "labels": torch.cat([_pad(c["labels"], IGNORE_INDEX), _pad(r["labels"], IGNORE_INDEX)]),
    }


# --------------------------------------------------------------------------- #
# Log-probabilities and losses
# --------------------------------------------------------------------------- #
def model_device(model) -> torch.device:
    return next(model.parameters()).device


def sequence_logprobs(model, batch: Dict[str, torch.Tensor]):
    """Return (summed response log-probs [B], response token counts [B]).

    Logits at position t predict token t+1, so labels are shifted left by one.
    Log-softmax is done in fp32 via gather - logsumexp (avoids materialising a
    full fp32 log-softmax tensor over the 150k-token vocabulary).
    """
    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe_targets = targets.masked_fill(~mask, 0)

    selected = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = selected - torch.logsumexp(logits, dim=-1)
    token_logps = token_logps * mask
    return token_logps.sum(-1), mask.sum(-1)


def response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    logps, _ = sequence_logprobs(model, batch)
    return logps


def sft_loss_from_batch(model, batch, token_normalizer=None):
    """Token-mean NLL over response tokens.

    ``token_normalizer`` lets the trainer divide by the token count of the whole
    gradient-accumulation group so the effective-batch loss is an exact token mean.
    """
    logps, counts = sequence_logprobs(model, batch)
    denom = counts.sum() if token_normalizer is None else token_normalizer
    denom = torch.as_tensor(denom, device=logps.device, dtype=logps.dtype).clamp_min(1)
    return -logps.sum() / denom


def sft_loss(model, tokenizer, prompts, responses, max_length):
    return sft_loss_from_batch(model, build_lm_batch(tokenizer, prompts, responses, max_length))


def dpo_logits(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    return beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))


def dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    logits = dpo_logits(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return -F.logsigmoid(logits).mean()


def dpo_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Implicit rewards r_w, r_l (detached)."""
    r_w = beta * (policy_chosen - ref_chosen).detach()
    r_l = beta * (policy_rejected - ref_rejected).detach()
    return r_w, r_l


def pair_logprobs(model, pair_batch, pad_id):
    """Chosen and rejected response log-probs from a single concatenated forward."""
    merged = concat_pair_batch(pair_batch, pad_id)
    logps, _ = sequence_logprobs(model, merged)
    n = pair_batch["chosen"]["input_ids"].shape[0]
    return logps[:n], logps[n:]


def dpo_loss_from_models(policy_model, reference_model, tokenizer, prompts, chosen, rejected,
                         beta, max_length, return_details=False):
    pair_batch = build_pair_batch(tokenizer, prompts, chosen, rejected, max_length)
    pad_id = _pad_id(tokenizer)
    policy_c, policy_r = pair_logprobs(policy_model, pair_batch, pad_id)
    with torch.no_grad():
        ref_c, ref_r = pair_logprobs(reference_model, pair_batch, pad_id)
    device = policy_c.device
    ref_c, ref_r = ref_c.detach().to(device), ref_r.detach().to(device)
    loss = dpo_loss_from_logps(policy_c, policy_r, ref_c, ref_r, beta)
    if return_details:
        return loss, (policy_c, policy_r, ref_c, ref_r)
    return loss
