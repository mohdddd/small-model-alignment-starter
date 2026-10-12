"""Single prompt template used for SFT, both DPO runs and Qwen evaluation.

Plain-text (Alpaca-style) template rather than ChatML: the base model never saw
ChatML turns as a trained format, and a plain template keeps EOS handling simple
(we append the tokenizer's ``<|endoftext|>`` after every response).
"""

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
STOP_STRINGS = ("\n### Instruction:", "### Instruction:")


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def clean_response(text: str) -> str:
    """Cut a generation at the start of a hallucinated next turn, then strip."""
    for stop in STOP_STRINGS:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()
