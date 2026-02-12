"""
Proper evaluation: Thinking States vs Vanilla Backbone on GSM8K.

The comparison that matters (per the paper) is:
  - Vanilla (direct answer, no CoT) -- what the backbone can do alone
  - Thinking States (direct answer, internal latent reasoning) -- what
    the T/C mechanism adds on top

Both receive the SAME prompt ("question\nAnswer:") and generate a short
direct answer.  The Thinking States model processes chunks internally
with its T and C blocks, but the *output* is still a short answer.

We additionally report a few-shot CoT reference (4-shot chain-of-thought
with #### formatting) as a ceiling showing what the backbone can do with
explicit in-context reasoning.

Usage:
    python eval.py                        # full test set
    python eval.py --limit 50             # first 50 examples
    python eval.py --limit 50 --verbose   # with per-example output
    python eval.py --skip-fewshot         # skip the slow CoT reference
"""

import argparse
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, MODEL_NAME, GSM8K_MAX_ANSWER_TOKENS, BOS_TOKEN, EOS_TOKEN
from data import GSM8KDataset
from model import ThinkingStatesModel
from inference import ThinkingStatesInference
from utils import extract_number, answers_match


# ---------------------------------------------------------------------------
# Vanilla direct-answer generation (same prompt as Thinking States)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_direct(model, tokenizer, prompt: str,
                    max_new_tokens: int = GSM8K_MAX_ANSWER_TOKENS,
                    device: str = DEVICE) -> str:
    """Greedy direct-answer generation (short, no CoT)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Few-shot CoT generation (reference ceiling)
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES = [
    {
        "q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
    },
    {
        "q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
    {
        "q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "a": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
    },
    {
        "q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "a": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
    },
]

FEWSHOT_MAX_NEW_TOKENS = 256


def build_fewshot_prompt(question: str) -> str:
    parts = [f"Question: {ex['q']}\nAnswer: {ex['a']}" for ex in FEWSHOT_EXAMPLES]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


@torch.no_grad()
def generate_fewshot(model, tokenizer, question: str,
                     device: str = DEVICE) -> str:
    """4-shot CoT generation with #### stop."""
    prompt = build_fewshot_prompt(question)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids,
        max_new_tokens=FEWSHOT_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Stop at the next "Question:" (model continuing to next example)
    if "Question:" in text:
        text = text[:text.index("Question:")].strip()
    return text


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(limit=None, verbose=False, show_samples=10, skip_fewshot=False):
    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print()

    # --- Load Thinking States model ---
    print("Loading Thinking States model...")
    ts_model = ThinkingStatesModel()
    try:
        state_dict = torch.load("thinking_states_model.pt", map_location="cpu")
        missing, unexpected = ts_model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  (ignored unexpected keys: {unexpected})")
        if missing:
            print(f"  (missing keys: {missing})")
        print("  Loaded trained weights.")
    except FileNotFoundError:
        print("  WARNING: No trained weights found! Using untrained model.")
    ts_model.to(DEVICE)
    ts_model.eval()
    ts_tokenizer = ts_model.tokenizer

    # --- Load vanilla backbone ---
    print(f"Loading vanilla {MODEL_NAME}...")
    load_kwargs = {}
    if "qwen" in MODEL_NAME.lower():
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["trust_remote_code"] = True
    vanilla_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    vanilla_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if vanilla_tokenizer.pad_token is None:
        vanilla_tokenizer.pad_token = vanilla_tokenizer.eos_token
    vanilla_model.to(DEVICE)
    vanilla_model.eval()
    print()

    # --- Load test set ---
    print("Loading GSM8K test set...")
    dataset = GSM8KDataset("test", tokenizer=ts_tokenizer, limit=limit)
    print(f"  {len(dataset)} examples\n")

    # --- Inference engine ---
    ts_inference = ThinkingStatesInference(ts_model)

    # --- Counters ---
    ts_correct = 0
    vanilla_correct = 0
    fewshot_correct = 0
    total = 0
    ts_time = 0.0
    vanilla_time = 0.0
    fewshot_time = 0.0
    samples = []

    for ex in tqdm(dataset, desc="Evaluating"):
        question = ex["question"].strip()
        gold = ex["final_answer"].strip()
        direct_prompt = question + "\nAnswer:"

        # 1) Thinking States (direct answer with internal reasoning)
        t0 = time.time()
        ts_result = ts_inference.infer(direct_prompt, verbose=False)
        ts_time += time.time() - t0
        ts_pred = ts_result["predicted_answer"]

        # 2) Vanilla direct answer (same prompt, same max tokens)
        t0 = time.time()
        vanilla_pred = generate_direct(vanilla_model, vanilla_tokenizer,
                                       direct_prompt, device=DEVICE)
        vanilla_time += time.time() - t0

        # 3) Few-shot CoT reference (optional)
        fewshot_pred = ""
        if not skip_fewshot:
            t0 = time.time()
            fewshot_pred = generate_fewshot(vanilla_model, vanilla_tokenizer,
                                           question, device=DEVICE)
            fewshot_time += time.time() - t0

        # --- Score ---
        ts_match = answers_match(ts_pred, gold)
        vanilla_match = answers_match(vanilla_pred, gold)
        fewshot_match = answers_match(fewshot_pred, gold) if not skip_fewshot else False

        if ts_match:
            ts_correct += 1
        if vanilla_match:
            vanilla_correct += 1
        if fewshot_match:
            fewshot_correct += 1
        total += 1

        if len(samples) < show_samples:
            samples.append({
                "question": question[:80] + ("..." if len(question) > 80 else ""),
                "gold": gold,
                "ts_pred": ts_pred[:80],
                "vanilla_pred": vanilla_pred[:80],
                "fewshot_pred": fewshot_pred[:80] if fewshot_pred else "",
                "thoughts": ts_result.get("thoughts", []),
                "ts_ok": ts_match,
                "vanilla_ok": vanilla_match,
                "fewshot_ok": fewshot_match,
            })

        if verbose:
            m_ts = "OK" if ts_match else "X"
            m_v = "OK" if vanilla_match else "X"
            m_f = ("OK" if fewshot_match else "X") if not skip_fewshot else "-"
            print(f"\n  Q: {question[:80]}...")
            print(f"  Gold: {gold}")
            print(f"  TS:      {ts_pred[:60]}  [{m_ts}]")
            print(f"  Vanilla: {vanilla_pred[:60]}  [{m_v}]")
            if not skip_fewshot:
                print(f"  CoT:     {fewshot_pred[:60]}  [{m_f}]")

    # --- Results table ---
    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    cols = ["Thinking States", "Vanilla (direct)", "4-shot CoT ref"]
    accs = [ts_correct, vanilla_correct, fewshot_correct if not skip_fewshot else None]
    times = [ts_time, vanilla_time, fewshot_time if not skip_fewshot else None]

    header = f"{'':30s}"
    for c in cols:
        if skip_fewshot and c == "4-shot CoT ref":
            continue
        header += f" {c:>18s}"
    print(header)
    print("-" * 78)

    # Accuracy row
    row = f"{'Numeric accuracy':30s}"
    for a in accs:
        if a is None:
            continue
        row += f" {a/total:>17.2%}"
    print(row)

    # Count row
    row = f"{'Correct / Total':30s}"
    for a in accs:
        if a is None:
            continue
        row += f" {a:>13d}/{total:<4d}"
    print(row)

    # Time row
    row = f"{'Total time (s)':30s}"
    for t in times:
        if t is None:
            continue
        row += f" {t:>17.1f}"
    print(row)

    row = f"{'Avg per example (s)':30s}"
    for t in times:
        if t is None:
            continue
        row += f" {t/total:>17.3f}"
    print(row)

    # --- Sample outputs ---
    print()
    print("=" * 78)
    print(f"SAMPLE OUTPUTS (first {len(samples)})")
    print("=" * 78)
    for i, s in enumerate(samples):
        m_ts = "OK" if s["ts_ok"] else "X"
        m_v = "OK" if s["vanilla_ok"] else "X"
        m_f = ("OK" if s["fewshot_ok"] else "X") if not skip_fewshot else ""
        print(f"\n--- Example {i+1} ---")
        print(f"  Q:       {s['question']}")
        print(f"  Gold:    {s['gold']}")
        print(f"  TS:      {s['ts_pred']}  [{m_ts}]")
        print(f"  Vanilla: {s['vanilla_pred']}  [{m_v}]")
        if not skip_fewshot and s["fewshot_pred"]:
            print(f"  CoT:     {s['fewshot_pred']}  [{m_f}]")
        if s["thoughts"]:
            non_empty = [t for t in s["thoughts"] if t]
            if non_empty:
                print(f"  Thoughts: {' | '.join(non_empty)[:120]}")

    # --- Summary ---
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Thinking States:       {ts_correct:>4d}/{total}  ({ts_correct/total:.2%})")
    print(f"  Vanilla (direct):      {vanilla_correct:>4d}/{total}  ({vanilla_correct/total:.2%})")
    if not skip_fewshot:
        print(f"  4-shot CoT reference:  {fewshot_correct:>4d}/{total}  ({fewshot_correct/total:.2%})")
    delta = ts_correct - vanilla_correct
    print(f"\n  TS vs Vanilla delta:   {'+' if delta >= 0 else ''}{delta}")
    if delta > 0:
        print(f"  ==> Thinking States improves over vanilla by {delta} examples.")
    elif delta < 0:
        print(f"  ==> Vanilla outperforms (TS may need more training).")
    else:
        print(f"  ==> Tied.")
    print()
    print("Note: The fair comparison is TS vs Vanilla (direct). Both receive the")
    print("same prompt and generate a short answer. TS reasons internally via the")
    print("T/C blocks; Vanilla relies purely on the pre-trained backbone.")
    print("The 4-shot CoT column is a reference ceiling (explicit reasoning).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Thinking States vs Vanilla on GSM8K")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-example output")
    parser.add_argument("--show-samples", type=int, default=10,
                        help="Number of samples to display (default: 10)")
    parser.add_argument("--skip-fewshot", action="store_true",
                        help="Skip the slow 4-shot CoT reference column")
    args = parser.parse_args()

    evaluate(limit=args.limit, verbose=args.verbose,
             show_samples=args.show_samples, skip_fewshot=args.skip_fewshot)
