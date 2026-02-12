"""
Standalone evaluation of the vanilla (pre-trained) backbone on GSM8K test set.

This is a lightweight script that only loads the base model -- no ThinkingStates
wrapper -- so it can run alongside training without competing for memory.

Usage:
    python eval_vanilla.py                  # full test set
    python eval_vanilla.py --limit 50       # first 50 examples
    python eval_vanilla.py --limit 20 --verbose
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, MODEL_NAME, GSM8K_MAX_ANSWER_TOKENS, GSM8K_CACHE_DIR


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_number(text: str) -> str | None:
    """Extract the final answer number from text.

    Strategy (in priority order):
    1. "the answer is <number>" / "answer: <number>"
    2. "#### <number>" (GSM8K gold format)
    3. "= <number>" at the end of a line (computation result)
    4. Number on its own line (model outputs just the number)
    5. First number if output starts with a digit (answer-first pattern)
    6. Last number as final fallback
    """
    text = text.strip()

    # Pattern 1: explicit "answer is" phrasing
    m = re.search(r'(?:the answer is|answer is|answer:)\s*\$?(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(',', '')

    # Pattern 2: "#### <number>"
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m:
        return m.group(1).replace(',', '')

    # Pattern 3: standalone number on its own line
    for line in text.split('\n'):
        line = line.strip()
        if re.fullmatch(r'\$?-?[\d,]+\.?\d*', line):
            return line.lstrip('$').replace(',', '')

    # Pattern 4: output starts with a number (answer-first, then explanation)
    # This is common for base models: "42\nExplanation: ..."
    m = re.match(r'\$?(-?[\d,]+\.?\d*)\s*\n', text)
    if m:
        return m.group(1).replace(',', '')

    # Pattern 5: "= <number>" as the final computation result
    equals_matches = re.findall(r'=\s*\$?(-?[\d,]+\.?\d*)', text)
    if equals_matches:
        return equals_matches[-1].replace(',', '')

    # Pattern 6: first number if output starts with a digit
    m = re.match(r'\$?(-?[\d,]+\.?\d*)', text)
    if m:
        return m.group(1).replace(',', '')

    # Pattern 7: last number as fallback
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


# ---------------------------------------------------------------------------
# Few-shot prompt template (standard GSM8K evaluation format)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
    },
]


def build_fewshot_prompt(question: str) -> str:
    """Build a few-shot prompt that teaches the model the #### answer format."""
    parts = []
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

FEWSHOT_MAX_NEW_TOKENS = 256  # enough for CoT reasoning + #### answer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = FEWSHOT_MAX_NEW_TOKENS, device: str = DEVICE) -> str:
    """Greedy generation with stop at #### delimiter."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Stop at the next "Question:" if the model tries to continue with another example
    if "Question:" in text:
        text = text[:text.index("Question:")].strip()

    return text


# ---------------------------------------------------------------------------
# Load test data directly (no dependency on data.py / ThinkingStatesModel)
# ---------------------------------------------------------------------------

def load_gsm8k_test(cache_dir: str = GSM8K_CACHE_DIR, limit: int | None = None):
    """Load GSM8K test set from aligned JSONL or raw dataset."""
    path = Path(cache_dir) / "test_aligned.jsonl"
    if not path.exists():
        # Fall back: try loading from HuggingFace datasets
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        examples = []
        for row in ds:
            # GSM8K gold answer is after "####"
            answer_text = row["answer"].split("####")[-1].strip()
            examples.append({
                "question": row["question"],
                "final_answer": answer_text,
            })
            if limit and len(examples) >= limit:
                break
        return examples

    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append({
                "question": ex["question"],
                "final_answer": ex["final_answer"],
            })
            if limit and len(examples) >= limit:
                break
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate vanilla backbone on GSM8K")
    parser.add_argument("--limit", type=int, default=None, help="Max examples")
    parser.add_argument("--verbose", action="store_true", help="Per-example output")
    parser.add_argument("--show-samples", type=int, default=10, help="Samples to display")
    args = parser.parse_args()

    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print()

    # Load model
    print("Loading model...")
    load_kwargs = {}
    if "qwen" in MODEL_NAME.lower():
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("  Model loaded.\n")

    # Load data
    print("Loading GSM8K test set...")
    examples = load_gsm8k_test(limit=args.limit)
    print(f"  {len(examples)} examples\n")

    # Evaluate
    correct = 0
    numeric_correct = 0
    total = 0
    total_time = 0.0
    samples = []

    for ex in tqdm(examples, desc="Evaluating"):
        question = ex["question"].strip()
        gold = ex["final_answer"].strip()
        prompt = build_fewshot_prompt(question)

        t0 = time.time()
        pred = generate(model, tokenizer, prompt, device=DEVICE)
        elapsed = time.time() - t0
        total_time += elapsed

        exact = extract_number(pred) == gold
        numeric = answers_match(pred, gold)

        if exact:
            correct += 1
        if numeric:
            numeric_correct += 1
        total += 1

        if len(samples) < args.show_samples:
            samples.append({
                "question": question[:80] + ("..." if len(question) > 80 else ""),
                "gold": gold,
                "pred": pred[:80],
                "correct": numeric,
            })

        if args.verbose:
            mark = "OK" if numeric else "X"
            print(f"\n  Q: {question[:80]}...")
            print(f"  Gold: {gold}")
            print(f"  Pred: {pred[:60]}  [{mark}]")

    # Results
    print()
    print("=" * 60)
    print(f"VANILLA {MODEL_NAME.split('/')[-1]} RESULTS")
    print("=" * 60)
    print(f"  Exact match accuracy:   {correct}/{total}  ({correct/total:.2%})")
    print(f"  Numeric match accuracy: {numeric_correct}/{total}  ({numeric_correct/total:.2%})")
    print(f"  Total time:             {total_time:.1f}s")
    print(f"  Avg per example:        {total_time/total:.3f}s")
    print()

    # Samples
    print("=" * 60)
    print(f"SAMPLE OUTPUTS (first {len(samples)})")
    print("=" * 60)
    for i, s in enumerate(samples):
        mark = "OK" if s["correct"] else "X"
        print(f"\n--- Example {i+1} ---")
        print(f"  Q:    {s['question']}")
        print(f"  Gold: {s['gold']}")
        print(f"  Pred: {s['pred']}  [{mark}]")

    print(f"\nFinal: {numeric_correct}/{total} ({numeric_correct/total:.2%})")


if __name__ == "__main__":
    main()
