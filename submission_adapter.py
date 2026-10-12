"""Stable grading interface for the SFT and DPO implementation.

Every function is a thin wrapper around src/pa2/core.py (the same code the
training scripts use). Signatures are unchanged from the starter scaffold.

Documented choices:
- Prompt/response are tokenized separately (no special tokens) and concatenated;
  EOS is appended to the response before right-truncation to ``max_length``.
- Rows are right-padded to the longest row; labels are -100 on prompt + padding.
- compute_sft_loss reduction: mean NLL over all unmasked response tokens in the batch.
- compute_dpo_loss_from_logps: batch mean of -log sigmoid(beta * (policy gap - reference gap)).
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pa2 import core  # noqa: E402


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return core.build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return core.response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return core.sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return core.build_pair_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return core.dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return core.dpo_loss_from_models(
        policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length
    )
