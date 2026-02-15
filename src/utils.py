"""Shared utility helpers for Thinking States."""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Backbone detection
# ---------------------------------------------------------------------------

def is_gpt2(model_name: str) -> bool:
    """Check if model_name refers to a GPT-2 model."""
    return "gpt2" in model_name.lower()


def is_qwen(model_name: str) -> bool:
    """Check if model_name refers to a Qwen model."""
    return "qwen" in model_name.lower()


# ---------------------------------------------------------------------------
# Answer extraction (used by eval scripts)
# ---------------------------------------------------------------------------

def extract_number(text: str) -> Optional[str]:
    """Extract the final answer number from generated text.

    Strategy (in priority order):
    1. "the answer is <number>"
    2. "#### <number>" (GSM8K gold format)
    3. Standalone number on its own line
    4. Number at the very start of the output (answer-first pattern)
    5. Last "= <number>" (computation result)
    6. First number if output starts with a digit
    7. Last number as fallback
    """
    text = text.strip()

    # 1: explicit "answer is" phrasing
    m = re.search(r'(?:the answer is|answer is|answer:)\s*\$?(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(',', '')

    # 2: "#### <number>"
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m:
        return m.group(1).replace(',', '')

    # 3: standalone number on its own line
    for line in text.split('\n'):
        line = line.strip()
        if re.fullmatch(r'\$?-?[\d,]+\.?\d*', line):
            return line.lstrip('$').replace(',', '')

    # 4: answer-first pattern ("42\nExplanation: ...")
    m = re.match(r'\$?(-?[\d,]+\.?\d*)\s*\n', text)
    if m:
        return m.group(1).replace(',', '')

    # 5: last "= <number>"
    equals_matches = re.findall(r'=\s*\$?(-?[\d,]+\.?\d*)', text)
    if equals_matches:
        return equals_matches[-1].replace(',', '')

    # 6: starts with a digit
    m = re.match(r'\$?(-?[\d,]+\.?\d*)', text)
    if m:
        return m.group(1).replace(',', '')

    # 7: last number
    matches = re.findall(r'-?[\d,]+\.?\d*', text)
    if matches:
        return matches[-1].replace(',', '')
    return None


def answers_match(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold numerically."""
    pred_num = extract_number(predicted)
    gold_num = extract_number(gold)
    if pred_num is None or gold_num is None:
        return False
    try:
        return float(pred_num) == float(gold_num)
    except ValueError:
        return pred_num == gold_num
