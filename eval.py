"""
Comparative evaluation: vanilla GPT-2 vs Thinking States on GSM8K test set.

Both models receive the same prompts ("question\nAnswer:") and generate up to
GSM8K_MAX_ANSWER_TOKENS tokens greedily. We compare:
  - Exact-match accuracy (predicted number == gold number)
  - Numeric accuracy (parsed number matches, tolerant of formatting)
  - Example-level output for qualitative inspection

Usage:
    python eval.py                        # full test set
    python eval.py --limit 100            # first 100 examples
    python eval.py --limit 50 --verbose   # 50 examples with per-example output
"""

import argparse
import re
import time

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import DEVICE, MODEL_NAME, GSM8K_MAX_ANSWER_TOKENS, BOS_TOKEN, EOS_TOKEN
from data import GSM8KDataset
from model import ThinkingStatesModel
from inference import ThinkingStatesInference


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_number(text: str) -> str | None:
    """
    Extract the first number from generated text.
    Handles integers, decimals, negatives, and comma-separated numbers.
    Returns a cleaned string or None.
    """
    text = text.strip()
    # Try to find a number pattern
    matches = re.findall(r'-?[\d,]+\.?\d*', text)
    if matches:
        return matches[0].replace(',', '')
    return None


def answers_match(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold, with numeric tolerance."""
    pred_num = extract_number(predicted)
    gold_num = extract_number(gold)
    if pred_num is None or gold_num is None:
        return False
    try:
        return float(pred_num) == float(gold_num)
    except ValueError:
        return pred_num == gold_num


# ---------------------------------------------------------------------------
# Vanilla GPT-2 generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_vanilla_gpt2(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = GSM8K_MAX_ANSWER_TOKENS,
    device: str = DEVICE,
) -> str:
    """Greedy generation from vanilla GPT-2."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the new tokens
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(limit: int | None = None, verbose: bool = False, show_samples: int = 10):
    print(f"Device: {DEVICE}")
    print()

    # --- Load test data ---
    # We need the tokenizer from the Thinking States model (has special tokens)
    print("Loading Thinking States model...")
    ts_model = ThinkingStatesModel()
    try:
        state_dict = torch.load("thinking_states_model.pt", map_location="cpu")
        ts_model.load_state_dict(state_dict)
        print("  Loaded trained weights.")
    except FileNotFoundError:
        print("  WARNING: No trained weights found! Using untrained model.")
    ts_model.to(DEVICE)
    ts_model.eval()
    ts_tokenizer = ts_model.tokenizer

    print("Loading vanilla GPT-2...")
    vanilla_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    vanilla_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    vanilla_tokenizer.pad_token = vanilla_tokenizer.eos_token
    vanilla_model.to(DEVICE)
    vanilla_model.eval()
    print()

    # --- Load dataset ---
    print("Loading GSM8K test set...")
    dataset = GSM8KDataset("test", tokenizer=ts_tokenizer, limit=limit)
    print(f"  {len(dataset)} examples")
    print()

    # --- Inference setup ---
    ts_inference = ThinkingStatesInference(ts_model)

    # --- Run evaluation ---
    ts_correct = 0
    vanilla_correct = 0
    ts_numeric_correct = 0
    vanilla_numeric_correct = 0
    total = 0

    ts_time = 0.0
    vanilla_time = 0.0

    samples = []  # Store examples for display

    for ex in tqdm(dataset, desc="Evaluating"):
        question = ex["question"].strip()
        gold_answer = ex["final_answer"].strip()
        prompt = question + "\nAnswer:"

        # --- Thinking States ---
        t0 = time.time()
        ts_result = ts_inference.infer(prompt, verbose=False)
        ts_time += time.time() - t0
        ts_pred = ts_result["predicted_answer"]

        # --- Vanilla GPT-2 ---
        t0 = time.time()
        vanilla_pred = generate_vanilla_gpt2(
            vanilla_model, vanilla_tokenizer, prompt, device=DEVICE
        )
        vanilla_time += time.time() - t0

        # --- Score ---
        ts_exact = extract_number(ts_pred) == gold_answer
        vanilla_exact = extract_number(vanilla_pred) == gold_answer
        ts_numeric = answers_match(ts_pred, gold_answer)
        vanilla_numeric = answers_match(vanilla_pred, gold_answer)

        if ts_exact:
            ts_correct += 1
        if vanilla_exact:
            vanilla_correct += 1
        if ts_numeric:
            ts_numeric_correct += 1
        if vanilla_numeric:
            vanilla_numeric_correct += 1
        total += 1

        # Store sample for display
        if len(samples) < show_samples:
            samples.append({
                "question": question[:80] + ("..." if len(question) > 80 else ""),
                "gold": gold_answer,
                "ts_pred": ts_pred[:60],
                "vanilla_pred": vanilla_pred[:60],
                "ts_thoughts": ts_result.get("thoughts", []),
                "ts_correct": ts_numeric,
                "vanilla_correct": vanilla_numeric,
            })

        if verbose:
            mark_ts = "OK" if ts_numeric else "X"
            mark_v = "OK" if vanilla_numeric else "X"
            print(f"\n  Q: {question[:80]}...")
            print(f"  Gold: {gold_answer}")
            print(f"  TS:   {ts_pred[:60]}  [{mark_ts}]")
            print(f"  GPT2: {vanilla_pred[:60]}  [{mark_v}]")

    # --- Results ---
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'':30s} {'Thinking States':>18s} {'Vanilla GPT-2':>18s}")
    print("-" * 70)
    print(f"{'Exact match accuracy':30s} {ts_correct/total:>17.2%} {vanilla_correct/total:>17.2%}")
    print(f"{'Numeric match accuracy':30s} {ts_numeric_correct/total:>17.2%} {vanilla_numeric_correct/total:>17.2%}")
    print(f"{'Correct count':30s} {ts_correct:>13d}/{total:<4d} {vanilla_correct:>13d}/{total:<4d}")
    print(f"{'Total time (s)':30s} {ts_time:>17.1f} {vanilla_time:>17.1f}")
    print(f"{'Avg time per example (s)':30s} {ts_time/total:>17.3f} {vanilla_time/total:>17.3f}")
    print()

    # --- Sample outputs ---
    print("=" * 70)
    print(f"SAMPLE OUTPUTS (first {len(samples)})")
    print("=" * 70)
    for i, s in enumerate(samples):
        ts_mark = "OK" if s["ts_correct"] else "X"
        v_mark = "OK" if s["vanilla_correct"] else "X"
        print(f"\n--- Example {i+1} ---")
        print(f"  Q:    {s['question']}")
        print(f"  Gold: {s['gold']}")
        print(f"  TS:   {s['ts_pred']}  [{ts_mark}]")
        print(f"  GPT2: {s['vanilla_pred']}  [{v_mark}]")
        if s["ts_thoughts"]:
            thoughts_str = " | ".join(t for t in s["ts_thoughts"] if t)
            if thoughts_str:
                print(f"  Thoughts: {thoughts_str[:100]}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if ts_numeric_correct > vanilla_numeric_correct:
        print(f"Thinking States wins by {ts_numeric_correct - vanilla_numeric_correct} examples.")
    elif vanilla_numeric_correct > ts_numeric_correct:
        print(f"Vanilla GPT-2 wins by {vanilla_numeric_correct - ts_numeric_correct} examples.")
    else:
        print("Both models tied.")

    delta = ts_numeric_correct - vanilla_numeric_correct
    print(f"Thinking States: {ts_numeric_correct}/{total} ({ts_numeric_correct/total:.2%})")
    print(f"Vanilla GPT-2:   {vanilla_numeric_correct}/{total} ({vanilla_numeric_correct/total:.2%})")
    print(f"Delta:           {'+' if delta >= 0 else ''}{delta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Thinking States vs vanilla GPT-2 on GSM8K")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples to evaluate (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example results")
    parser.add_argument("--show-samples", type=int, default=10,
                        help="Number of sample outputs to display (default: 10)")
    args = parser.parse_args()

    evaluate(limit=args.limit, verbose=args.verbose, show_samples=args.show_samples)
