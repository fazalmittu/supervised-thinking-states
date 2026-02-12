"""Teacher-LLM alignment pipeline for GSM8K dataset.

Uses Gemini as the teacher LLM to insert <THINK> markers into questions
at positions where each reasoning step becomes inferable, producing
chunk-aligned training data for Supervised Thinking States.
"""

import asyncio
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer

from config import GSM8K_CACHE_DIR, GSM8K_CHUNK_SIZE, TEACHER_MODEL, MODEL_NAME

load_dotenv()

DEFAULT_CONCURRENCY = 50
DEFAULT_MAX_RETRIES = 3
RATE_LIMIT_BACKOFF = 5.0


# ---------------------------------------------------------------------------
# GSM8K answer parsing
# ---------------------------------------------------------------------------

def parse_gsm8k_answer(answer_text: str) -> dict:
    """
    Parse GSM8K answer into steps, final answer, and calculation trace.
    Calc trace entries are bare equations (<<>> stripped).
    """
    parts = answer_text.split("####")
    if len(parts) == 2:
        steps_text = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        steps_text = answer_text.strip()
        numbers = re.findall(r'-?[\d,]+\.?\d*', answer_text)
        final_answer = numbers[-1] if numbers else ""

    # Extract bare equations from <<...>> annotations (capture group strips <<>>)
    calc_trace = re.findall(r'<<([^<>]+)>>', steps_text)

    # Clean step text
    clean_text = re.sub(r'<<.*?>>', '', steps_text)
    steps = [s.strip() for s in clean_text.split('\n') if s.strip()]

    final_answer = final_answer.replace(',', '')
    return {
        "steps": steps,
        "final_answer": final_answer,
        "calc_trace": calc_trace,
    }


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize for comparison: unicode NFKC, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    return ' '.join(text.split()).strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_alignment_prompt(question: str, calc_trace: list) -> str:
    """Build prompt asking teacher to insert <THINK> markers."""
    trace_str = ", ".join(f"'{eq}'" for eq in calc_trace)

    return f"""You are an expert in computational linguistics. Your task is to augment a given query with thinking markers (<THINK>) based on a provided reasoning trace.

Task Instructions:
1. You will receive a Query and a Reasoning Trace (ordered list of calculations).
2. Insert <THINK> tokens into the Query at the earliest position where each calculation becomes inferable.
3. Rules for <THINK> Placement:
- The number of <THINK> tokens must exactly equal the number of calculations.
- The order of <THINK> tokens corresponds one-to-one with the calculations.
- Each <THINK> goes immediately after the word or phrase that provides the final piece of information needed for that calculation.
4. Return ONLY a Python dictionary with key "query" containing the modified text.

Example 1:
Input Query: 'Hannah has three dogs. The first dog eats 1.5 cups of dog food a day. The second dog eats twice as much while the third dog eats 2.5 cups more than the second dog. How many cups of dog food should Hannah prepare in a day for her three dogs?'
Input Reasoning Trace: ['1.5*2=3', '3+2.5=5.5', '1.5+3+5.5=10']
Output:
{{"query": "Hannah has three dogs. The first dog eats 1.5 cups of dog food a day. The second dog eats twice as much<THINK> while the third dog eats 2.5 cups more than the second dog.<THINK> How many cups of dog food should Hannah prepare in a day for her three dogs?<THINK>"}}

Example 2:
Input Query: 'Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple DMV combined. How many flowers does Mark have in his garden?'
Input Reasoning Trace: ['10*80/100=8', '10+8=18', '18*25/100=4.5', '10+8+4.5=22.5']
Output:
{{"query": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple.<THINK><THINK> There are only 25% as many green flowers as there are yellow and purple combined.<THINK> How many flowers does Mark have in his garden?<THINK>"}}

Task:
Input Query: '{question}'
Input Reasoning Trace: [{trace_str}]
Output:"""


# ---------------------------------------------------------------------------
# Response parsing (lenient, multi-strategy)
# ---------------------------------------------------------------------------

def parse_alignment_response(response: str, num_steps: int, question: str) -> str | None:
    """
    Extract query with <THINK> markers from teacher response.
    Tries multiple extraction strategies. Returns query string or None.
    """
    text = response.strip()
    candidates = []

    # Strategy 1: JSON double-quoted "query" value
    for m in re.finditer(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL):
        val = m.group(1).replace('\\"', '"').replace('\\n', '\n')
        if '<THINK>' in val:
            candidates.append(val)

    # Strategy 2: <<...>> delimited value
    for m in re.finditer(r'"query"\s*:\s*<<(.+?)>>', text, re.DOTALL):
        val = m.group(1).strip().strip("'\"")
        if '<THINK>' in val:
            candidates.append(val)

    # Strategy 3: single-quoted value
    for m in re.finditer(r"""['"]query['"]\s*:\s*'((?:[^'\\]|\\.)*)'""", text, re.DOTALL):
        val = m.group(1)
        if '<THINK>' in val:
            candidates.append(val)

    # Strategy 4: any line with the right number of THINK markers
    for line in text.split('\n'):
        if '<THINK>' in line and line.count('<THINK>') == num_steps:
            # Strip dict key prefix if present
            cleaned = re.sub(r'^[^:]*:\s*', '', line)
            cleaned = cleaned.strip().strip('{}"\',<> ')
            if len(cleaned) > len(question) * 0.5:
                candidates.append(cleaned)

    # Evaluate candidates
    norm_q = _normalize(question)
    for candidate in candidates:
        if candidate.count('<THINK>') != num_steps:
            continue

        # Strip potential leading/trailing quote junk
        c = candidate.strip().strip("'\"")
        cleaned = c.replace('<THINK>', '')
        if _normalize(cleaned) == norm_q:
            return c

    return None


# ---------------------------------------------------------------------------
# Token/chunk alignment
# ---------------------------------------------------------------------------

def char_pos_to_token_pos(question: str, char_positions: list, tokenizer) -> list:
    """Convert character positions to token positions."""
    tokens = tokenizer.encode(question, add_special_tokens=False)

    char_offset = 0
    token_char_starts = []
    for tok_id in tokens:
        token_text = tokenizer.decode([tok_id])
        token_char_starts.append(char_offset)
        char_offset += len(token_text)

    token_positions = []
    for char_pos in char_positions:
        tok_pos = len(tokens) - 1
        for t_idx, t_start in enumerate(token_char_starts):
            if t_start > char_pos:
                tok_pos = max(0, t_idx - 1)
                break
        token_positions.append(tok_pos)

    return token_positions


def build_aligned_example(question, answer, query_with_markers, tokenizer):
    """Build final aligned example dict from a successfully parsed response."""
    parsed = parse_gsm8k_answer(answer)
    calc_trace = parsed["calc_trace"]
    final_answer = parsed["final_answer"]
    steps = parsed["steps"]

    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    chunk_size = GSM8K_CHUNK_SIZE
    num_chunks = (len(question_tokens) + chunk_size - 1) // chunk_size

    # Extract character positions of <THINK> markers
    char_positions = []
    clean_idx = 0
    i = 0
    while i < len(query_with_markers):
        if query_with_markers.startswith("<THINK>", i):
            char_positions.append(clean_idx)
            i += 7
            continue
        clean_idx += 1
        i += 1

    token_positions = char_pos_to_token_pos(question, char_positions, tokenizer)

    step_chunk_assignments = [
        min(tok_pos // chunk_size, num_chunks - 1)
        for tok_pos in token_positions
    ]

    chunk_thoughts = []
    for c in range(num_chunks):
        assigned = [calc_trace[j] for j, ca in enumerate(step_chunk_assignments) if ca == c]
        chunk_thoughts.append(" ".join(assigned) if assigned else "")

    return {
        "question": question,
        "final_answer": final_answer,
        "num_chunks": num_chunks,
        "chunk_thoughts": chunk_thoughts,
        "steps": steps,
        "step_chunk_assignments": step_chunk_assignments,
    }


# ---------------------------------------------------------------------------
# Async alignment pipeline
# ---------------------------------------------------------------------------

async def align_dataset(
    split="train",
    tokenizer=None,
    max_examples=None,
    output_suffix="",
    concurrency=DEFAULT_CONCURRENCY,
    max_retries=DEFAULT_MAX_RETRIES,
):
    """Align a GSM8K split using async concurrent Gemini calls."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    cache_dir = Path(GSM8K_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{split}_aligned{output_suffix}.jsonl"

    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    print(f"  {len(dataset)} total examples")

    # Resume: skip already-processed questions
    existing_questions = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        existing_questions.add(json.loads(line)["question"])
                    except Exception:
                        pass
        if existing_questions:
            print(f"  {len(existing_questions)} already aligned (resuming)")

    remaining = [ex for ex in dataset if ex["question"] not in existing_questions]
    if max_examples is not None:
        remaining = remaining[:max_examples]

    if not remaining:
        print("  Nothing to do!")
        return

    print(f"  {len(remaining)} to align (concurrency={concurrency}, retries={max_retries})")

    # Setup Gemini client
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Export it or add to .env")
    client = genai.Client(api_key=api_key)

    use_native_async = hasattr(client, 'aio')
    sem = asyncio.Semaphore(concurrency)
    total = len(remaining)
    written = 0
    failed = 0
    start_time = time.time()

    # Global rate-limit cooldown: when any task hits a 429, all tasks
    # respect this timestamp before sending new requests.
    rate_limit_until = 0.0
    rate_limit_backoff = RATE_LIMIT_BACKOFF

    async def call_gemini(prompt):
        """Single Gemini call with rate-limit awareness and retries."""
        nonlocal rate_limit_until, rate_limit_backoff

        for attempt in range(6):
            # Respect global cooldown from any task's rate-limit hit
            now = time.time()
            if now < rate_limit_until:
                await asyncio.sleep(rate_limit_until - now + random.uniform(0, 2))

            try:
                if use_native_async:
                    resp = await client.aio.models.generate_content(
                        model=TEACHER_MODEL, contents=prompt
                    )
                else:
                    resp = await asyncio.to_thread(
                        client.models.generate_content,
                        model=TEACHER_MODEL, contents=prompt
                    )
                # Success — decay the backoff
                rate_limit_backoff = max(RATE_LIMIT_BACKOFF, rate_limit_backoff * 0.8)
                return resp.text or ""
            except Exception as e:
                err = str(e).lower()
                if "rate" in err and "limit" in err:
                    # Set global cooldown so all tasks back off
                    rate_limit_until = time.time() + rate_limit_backoff
                    jitter = random.uniform(0, rate_limit_backoff * 0.5)
                    print(f"    rate limit hit, backing off {rate_limit_backoff:.0f}s")
                    await asyncio.sleep(rate_limit_backoff + jitter)
                    rate_limit_backoff = min(rate_limit_backoff * 1.5, 120)
                elif attempt < 5:
                    await asyncio.sleep(min(2 ** attempt, 30))
                else:
                    return None
        return None

    async def process_one(example):
        """Align a single example with retries on parse failure."""
        question = example["question"]
        answer = example["answer"]

        parsed = parse_gsm8k_answer(answer)
        calc_trace = parsed["calc_trace"]
        if not calc_trace:
            return None

        tokens = tokenizer.encode(question, add_special_tokens=False)
        if not tokens:
            return None

        prompt = build_alignment_prompt(question, calc_trace)

        async with sem:
            for attempt in range(max_retries):
                resp_text = await call_gemini(prompt)
                if resp_text is None:
                    continue

                qwm = parse_alignment_response(resp_text, len(calc_trace), question)
                if qwm is not None:
                    return build_aligned_example(question, answer, qwm, tokenizer)

                # Parse failed — retry with a fresh API call
            return None

    # Launch all tasks, collect as they complete
    tasks = [asyncio.create_task(process_one(ex)) for ex in remaining]

    with open(output_path, "a") as f:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
            except Exception:
                result = None

            if result is not None:
                f.write(json.dumps(result) + "\n")
                f.flush()
                written += 1
            else:
                failed += 1

            done = written + failed
            if done % 50 == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / max(elapsed, 0.01)
                eta = (total - done) / max(rate, 0.01)
                success_pct = written / max(done, 1) * 100
                print(
                    f"  {done}/{total} | "
                    f"written={written} ({success_pct:.0f}%) failed={failed} | "
                    f"{rate:.1f} ex/s | ETA {eta/60:.1f}m"
                )

    elapsed = time.time() - start_time
    print(f"\n  {split} done: {written} written, {failed} failed, {elapsed/60:.1f}m")
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def test_single_example(tokenizer):
    """Quick offline test of parsing and prompt construction."""
    question = (
        "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every "
        "morning and bakes muffins for her friends every day with four. She sells "
        "every duck egg at the farmers' market daily for $2. How much in dollars "
        "does she make every day at the farmers' market?"
    )
    answer = (
        "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
        "She makes 9 * 2 = <<9*2=18>>$18 every day at the farmer\u2019s market.\n"
        "#### 18"
    )
    parsed = parse_gsm8k_answer(answer)
    print(f"Calc trace: {parsed['calc_trace']}")
    print(f"Steps: {parsed['steps']}")
    print(f"Final answer: {parsed['final_answer']}")

    prompt = build_alignment_prompt(question, parsed["calc_trace"])
    print(f"\nPrompt:\n{prompt}\n")

    # Test parser on a synthetic response
    mock = (
        '{"query": "Janet\u2019s ducks lay 16 eggs per day. She eats three for '
        'breakfast every morning and bakes muffins for her friends every day with '
        'four.<THINK> She sells every duck egg at the farmers\' market daily for $2. '
        'How much in dollars does she make every day at the farmers\' market?<THINK>"}'
    )
    qwm = parse_alignment_response(mock, len(parsed["calc_trace"]), question)
    if qwm:
        print(f"Parsed markers: {qwm}\n")
        result = build_aligned_example(question, answer, qwm, tokenizer)
        print(json.dumps(result, indent=2))
    else:
        print("Parser failed on mock response")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align GSM8K with teacher LLM.")
    parser.add_argument("--test-only", action="store_true",
                        help="Offline test of parsing (no API calls)")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "test", "both"])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if args.test_only:
        test_single_example(tokenizer)
    else:
        splits = ["train", "test"] if args.split == "both" else [args.split]
        for split in splits:
            asyncio.run(align_dataset(
                split, tokenizer,
                max_examples=args.max_examples,
                output_suffix=args.output_suffix,
                concurrency=args.concurrency,
                max_retries=args.max_retries,
            ))
