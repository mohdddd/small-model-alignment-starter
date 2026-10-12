"""Offline checks of the dataset parsers on records shaped like the real ones."""

from datasets import Dataset

from pa2 import data as D
from pa2.prompting import clean_response, format_prompt
from tests.toy_lm import ToyTokenizer


class CharTok(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [2] * len(text)  # 1 token per char; content irrelevant here


def test_parse_hh_single_turn_and_rejects_multi_turn():
    c = "\n\nHuman: How do I bake bread?\n\nAssistant: Mix flour, water, yeast."
    r = "\n\nHuman: How do I bake bread?\n\nAssistant: No."
    assert D.parse_hh(c, r) == ("How do I bake bread?", "Mix flour, water, yeast.", "No.")
    multi = c + "\n\nHuman: thanks\n\nAssistant: welcome"
    assert D.parse_hh(multi, multi.replace("welcome", "bye")) is None
    assert D.parse_hh(c, c) is None  # identical responses


def test_hh_and_uf_builders_filter_and_cap():
    hh = Dataset.from_list([
        {"chosen": f"\n\nHuman: q{i}\n\nAssistant: good {i}", "rejected": f"\n\nHuman: q{i}\n\nAssistant: bad {i}"}
        for i in range(10)
    ] + [{"chosen": "\n\nHuman: " + "x" * 400 + "\n\nAssistant: a", "rejected": "\n\nHuman: " + "x" * 400 + "\n\nAssistant: b"}])
    rows, stats = D.build_split(hh, D.hh_rows, CharTok(), 5, 256, 42, "train")
    assert len(rows) == 5 and all(set(r) == {"id", "instruction", "chosen", "rejected"} for r in rows)

    msg = lambda u, a: [{"role": "user", "content": u}, {"role": "assistant", "content": a}]  # noqa: E731
    uf = Dataset.from_list([
        {"prompt": "p1", "prompt_id": "a", "chosen": msg("p1", "great"), "rejected": msg("p1", "meh"),
         "score_chosen": 8.0, "score_rejected": 3.0},
        {"prompt": "p2", "prompt_id": "b", "chosen": msg("p2", "x"), "rejected": msg("p2", "y"),
         "score_chosen": 5.0, "score_rejected": 5.0},  # tie -> dropped
    ])
    rows, stats = D.build_split(uf, D.uf_rows, CharTok(), 10, 256, 42, "train")
    assert [r["chosen"] for r in rows] == ["great"] and stats.get("score_tie") == 1


def test_alpaca_subset_is_deterministic():
    recs = [{"instruction": f"i{k}", "dataset": "x"} for k in range(805)]
    a, b = D.alpaca_subset(recs, 300, 42), D.alpaca_subset(recs, 300, 42)
    assert a == b and len(a) == 300 and len({r["id"] for r in a}) == 300


def test_template_roundtrip():
    assert format_prompt("  Hi ") == "### Instruction:\nHi\n\n### Response:\n"
    assert clean_response(" answer\n### Instruction:\nnext") == "answer"
