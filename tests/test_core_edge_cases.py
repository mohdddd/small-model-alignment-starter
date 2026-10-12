"""Extra checks beyond the public smoke tests (truncation, pad==eos, ordering,
reference math against a manual computation on a tiny real HF causal LM)."""

from __future__ import annotations

import torch

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class PadIsEosTokenizer(ToyTokenizer):
    """Mimics Qwen2.5: pad token id == eos token id."""

    def __init__(self):
        super().__init__()
        self.pad_token_id = self.eos_token_id  # build vocab first, then alias pad -> eos


class NoEosNoPadTokenizer(ToyTokenizer):
    eos_token_id = None
    pad_token_id = None


def test_truncation_drops_eos_and_keeps_shape():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: "], ["hello world"], max_length=8)
    assert batch["input_ids"].shape == (1, 8)
    # prompt 4 tokens + response 11 + eos -> truncated to 8: no eos left
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    assert batch["labels"][0, :4].eq(-100).all() and batch["labels"][0, 4:].ne(-100).all()


def test_eos_kept_exactly_at_boundary():
    tok = ToyTokenizer()
    # prompt 2 + response 3 + eos = 6 == max_length
    batch = adapter.build_sft_batch(tok, ["q:"], ["abc"], max_length=6)
    assert batch["labels"][0, -1].item() == tok.eos_token_id
    assert batch["attention_mask"][0].eq(1).all()


def test_pad_equal_to_eos_keeps_eos_label():
    tok = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["q: ", "longer prompt: "], ["a", "b"], max_length=32)
    # short row is padded with id 1 (== eos) but only the real eos is a label
    row0_labels = batch["labels"][0]
    assert (row0_labels == 1).sum().item() == 1
    assert batch["attention_mask"][0].sum().item() == len(tok.encode("q: ")) + 2


def test_no_eos_no_pad_tokenizer():
    tok = NoEosNoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["q: ", "q2: "], ["ab", "a"], max_length=16)
    assert batch["labels"].ne(-100).sum().item() == 3


def test_prompt_longer_than_max_length_gives_zero_logp_and_finite_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    logps = adapter.compute_response_logprobs(model, tok, ["x" * 30], ["abc"], max_length=10)
    assert logps.item() == 0.0
    loss = adapter.compute_sft_loss(model, tok, ["x" * 30, "q: "], ["abc", "d"], max_length=10)
    assert torch.isfinite(loss)


def test_row_order_preserved_with_mixed_lengths():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    prompts = ["a: ", "bbbbbbb: ", "c: "]
    responses = ["long response here", "x", "mid one"]
    together = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=40)
    single = torch.stack([
        adapter.compute_response_logprobs(model, tok, [p], [r], max_length=40)[0]
        for p, r in zip(prompts, responses)
    ])
    torch.testing.assert_close(together, single)
    dpo = adapter.build_dpo_batch(tok, prompts, responses, list(reversed(responses)), max_length=40)
    assert dpo["chosen"]["input_ids"].shape[0] == 3


def test_dpo_formula_and_gradient_direction():
    pc, pr = torch.tensor([-3.0, -1.0], requires_grad=True), torch.tensor([-2.0, -4.0], requires_grad=True)
    rc, rr = torch.tensor([-3.5, -1.5]), torch.tensor([-2.5, -3.0])
    beta = 0.1
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    logits = beta * ((pc - pr) - (rc - rr))
    expected = -torch.nn.functional.logsigmoid(logits).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (pc.grad < 0).all() and (pr.grad > 0).all()  # descent raises chosen, lowers rejected


def test_against_manual_logprobs_on_tiny_hf_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tok = PadIsEosTokenizer()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg).eval()
    prompts, responses = ["hi: ", "a much longer prompt: "], ["yo", "ok then"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=64)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        p_ids = tok.encode(p)
        r_ids = tok.encode(r) + [tok.eos_token_id]
        ids = torch.tensor([p_ids + r_ids])
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0].float(), -1)
        manual = sum(lp[len(p_ids) + j - 1, t].item() for j, t in enumerate(r_ids))
        assert abs(got[i].item() - manual) < 1e-4  # padding did not change the score

    # DPO: identical policy/reference -> log 2; one SGD step lowers the loss
    import copy
    ref = copy.deepcopy(model)
    policy = model.train()
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    args = (tok, ["q: ", "w: "], ["good", "yes"], ["bad", "no"], 0.5, 32)
    loss0 = adapter.compute_dpo_loss(policy, ref, *args)
    torch.testing.assert_close(loss0, torch.log(torch.tensor(2.0)), atol=1e-5, rtol=0)
    loss0.backward()
    assert all(p.grad is None for p in ref.parameters())
    opt.step()
    opt.zero_grad()
    assert adapter.compute_dpo_loss(policy, ref, *args).item() < loss0.item()
