"""Unified evaluation: Thinking States vs Vanilla on GSM8K or Parity.

Branches on TASK from config.py:
  - gsm8k:  loads both models, evaluates on GSM8K test set, prints table
  - parity: loads both models, evaluates across flip ranges, prints table + chart

Usage:
    python eval.py                        # evaluate current TASK
    python eval.py --limit 50             # first 50 examples (gsm8k)
    python eval.py --num-examples 200     # examples per range (parity)
    python eval.py --verbose              # per-example output
    python eval.py --skip-fewshot         # skip slow CoT reference (gsm8k)
"""

import argparse
import os
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DEVICE, MODEL_NAME, TASK, CHUNK_SIZE,
    GSM8K_MAX_ANSWER_TOKENS, BOS_TOKEN, EOS_TOKEN,
    MIN_FLIPS, MAX_FLIPS,
)
from src.model import ThinkingStatesModel
from src.inference import ThinkingStatesInference
from src.data import GSM8KDataset, generate_parity_example
from src.utils import extract_number, answers_match


CHECKPOINT_DIR = "checkpoints"


def _checkpoint_path(task: str, vanilla: bool) -> str:
    prefix = "vanilla" if vanilla else "thinking_states"
    return os.path.join(CHECKPOINT_DIR, f"{prefix}_{task}.pt")


# ===========================================================================
# GSM8K evaluation
# ===========================================================================

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
    if "Question:" in text:
        text = text[:text.index("Question:")].strip()
    return text


def evaluate_gsm8k(limit=None, verbose=False, show_samples=10, skip_fewshot=False):
    """Evaluate Thinking States vs Vanilla on GSM8K."""
    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print()

    # --- Load Thinking States model ---
    print("Loading Thinking States model...")
    ts_model = ThinkingStatesModel()
    ts_ckpt = _checkpoint_path("gsm8k", vanilla=False)
    try:
        state_dict = torch.load(ts_ckpt, map_location="cpu")
        missing, unexpected = ts_model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  (ignored unexpected keys: {unexpected})")
        if missing:
            print(f"  (missing keys: {missing})")
        print(f"  Loaded {ts_ckpt}")
    except FileNotFoundError:
        print(f"  WARNING: {ts_ckpt} not found! Using untrained model.")
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

    ts_inference = ThinkingStatesInference(ts_model)

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

        t0 = time.time()
        ts_result = ts_inference.infer(direct_prompt, verbose=False)
        ts_time += time.time() - t0
        ts_pred = ts_result["predicted_answer"]

        t0 = time.time()
        vanilla_pred = generate_direct(vanilla_model, vanilla_tokenizer,
                                       direct_prompt, device=DEVICE)
        vanilla_time += time.time() - t0

        fewshot_pred = ""
        if not skip_fewshot:
            t0 = time.time()
            fewshot_pred = generate_fewshot(vanilla_model, vanilla_tokenizer,
                                           question, device=DEVICE)
            fewshot_time += time.time() - t0

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

    # --- Results ---
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

    row = f"{'Numeric accuracy':30s}"
    for a in accs:
        if a is None:
            continue
        row += f" {a/total:>17.2%}"
    print(row)

    row = f"{'Correct / Total':30s}"
    for a in accs:
        if a is None:
            continue
        row += f" {a:>13d}/{total:<4d}"
    print(row)

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

    # --- Samples ---
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


# ===========================================================================
# Parity evaluation
# ===========================================================================

@torch.no_grad()
def vanilla_predict_parity(model, tokenizer, input_text, device, max_new_tokens=3):
    """Generate answer from vanilla model on parity task."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def ts_predict_parity(inference_engine, input_text):
    """Generate answer from Thinking States model on parity task."""
    result = inference_engine.infer(input_text, verbose=False, max_answer_tokens=3)
    return result["predicted_answer"]


def evaluate_parity_range(
    vanilla_model, vanilla_tokenizer,
    ts_inference,
    min_flips, max_flips, num_examples, device, tokenizer,
    label: str = "",
):
    """Return (vanilla_acc, ts_acc) over num_examples parity problems."""
    vanilla_correct = 0
    ts_correct = 0
    total = 0

    desc = f"  {label}" if label else "Evaluating"
    for i in tqdm(range(num_examples), desc=desc, leave=False):
        num_flips = min_flips + (i % (max_flips - min_flips + 1))
        example = generate_parity_example(num_flips, CHUNK_SIZE, tokenizer)
        input_text = example["input_text"]
        true_answer = example["answer"].lower().strip()

        if vanilla_model is not None:
            pred = vanilla_predict_parity(vanilla_model, vanilla_tokenizer, input_text, device)
            if pred.lower().strip().startswith(true_answer):
                vanilla_correct += 1

        if ts_inference is not None:
            pred = ts_predict_parity(ts_inference, input_text)
            if pred.lower().strip().startswith(true_answer):
                ts_correct += 1

        total += 1

    v_acc = vanilla_correct / total if total > 0 else 0.0
    t_acc = ts_correct / total if total > 0 else 0.0
    return v_acc, t_acc


def make_parity_chart(labels, vanilla_accs, ts_accs, path="eval_parity_results.png"):
    """Save a grouped bar chart comparing the two models."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_v = ax.bar(x - width / 2, [a * 100 for a in vanilla_accs],
                    width, label="Vanilla GPT-2", color="#5B9BD5")
    bars_t = ax.bar(x + width / 2, [a * 100 for a in ts_accs],
                    width, label="Thinking States", color="#ED7D31")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Parity Task: Length Generalization", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(len(labels) - 0.5, 51.5, "random baseline", fontsize=8,
            color="gray", ha="right")

    for bar in bars_v:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars_t:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nChart saved to {path}")
    plt.close(fig)


def evaluate_parity(num_examples=200):
    """Evaluate Thinking States vs Vanilla on parity task."""
    device = DEVICE
    print(f"Device: {device}")
    print(f"Backbone: {MODEL_NAME}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Training flips: {MIN_FLIPS}-{MAX_FLIPS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load models ---
    print("\nLoading models...")

    vanilla_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Add special tokens so vanilla model matches the trained tokenizer
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [BOS_TOKEN, EOS_TOKEN]}
    )
    if added > 0:
        vanilla_model.resize_token_embeddings(len(tokenizer))
    vanilla_ckpt = _checkpoint_path("parity", vanilla=True)
    try:
        vanilla_model.load_state_dict(
            torch.load(vanilla_ckpt, map_location="cpu")
        )
        print(f"  Loaded {vanilla_ckpt}")
    except FileNotFoundError:
        print(f"  WARNING: {vanilla_ckpt} not found -- using untrained {MODEL_NAME}")
    vanilla_model.to(device)
    vanilla_model.eval()

    ts_model = ThinkingStatesModel(task="parity")
    ts_ckpt = _checkpoint_path("parity", vanilla=False)
    try:
        ts_model.load_state_dict(
            torch.load(ts_ckpt, map_location="cpu"),
            strict=False,
        )
        print(f"  Loaded {ts_ckpt}")
    except FileNotFoundError:
        print(f"  WARNING: {ts_ckpt} not found -- using untrained model")
    ts_model.to(device)
    ts_model.eval()
    ts_inference = ThinkingStatesInference(ts_model)

    # --- Evaluate across flip ranges ---
    eval_ranges = [
        (MIN_FLIPS, MAX_FLIPS, f"In-dist\n({MIN_FLIPS}-{MAX_FLIPS})"),
        (10, 16,               "OOD\n(10-16)"),
        (20, 30,               "OOD\n(20-30)"),
        (30, 40,               "OOD\n(30-40)"),
    ]

    labels, vanilla_accs, ts_accs = [], [], []

    print(f"\nEvaluating ({num_examples} examples per range)...\n")
    print(f"{'Range':<22s} {'Vanilla':>10s} {'ThinkStates':>13s} {'Delta':>8s}")
    print("-" * 56)

    for min_f, max_f, label in eval_ranges:
        tag = label.replace("\n", " ")
        v_acc, t_acc = evaluate_parity_range(
            vanilla_model, tokenizer,
            ts_inference,
            min_f, max_f, num_examples, device, tokenizer,
            label=tag,
        )
        delta = t_acc - v_acc
        sign = "+" if delta >= 0 else ""
        print(f"{tag:<22s} {v_acc:>9.1%} {t_acc:>12.1%} {sign}{delta:>7.1%}")

        labels.append(label)
        vanilla_accs.append(v_acc)
        ts_accs.append(t_acc)

    print("-" * 56)

    # --- Sample predictions ---
    print(f"\nSample predictions (20 flips):")
    for i in range(5):
        example = generate_parity_example(20, CHUNK_SIZE, tokenizer)
        true = example["answer"]
        v_pred = vanilla_predict_parity(vanilla_model, tokenizer, example["input_text"], device)
        t_pred = ts_predict_parity(ts_inference, example["input_text"])
        v_ok = "OK" if v_pred.lower().strip().startswith(true.lower()) else "X "
        t_ok = "OK" if t_pred.lower().strip().startswith(true.lower()) else "X "
        print(f"  True: {true:6s}  Vanilla: {v_pred:8s} [{v_ok}]  "
              f"TS: {t_pred:8s} [{t_ok}]")

    # --- Chart ---
    make_parity_chart(labels, vanilla_accs, ts_accs)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Thinking States vs Vanilla")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples (gsm8k, default: all)")
    parser.add_argument("--num-examples", type=int, default=200,
                        help="Examples per flip range (parity, default: 200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-example output (gsm8k)")
    parser.add_argument("--show-samples", type=int, default=10,
                        help="Number of samples to display (gsm8k, default: 10)")
    parser.add_argument("--skip-fewshot", action="store_true",
                        help="Skip the slow 4-shot CoT reference (gsm8k)")
    args = parser.parse_args()

    if TASK == "gsm8k":
        evaluate_gsm8k(limit=args.limit, verbose=args.verbose,
                        show_samples=args.show_samples, skip_fewshot=args.skip_fewshot)
    elif TASK == "parity":
        evaluate_parity(num_examples=args.num_examples)
    else:
        print(f"Unknown TASK: {TASK}")
