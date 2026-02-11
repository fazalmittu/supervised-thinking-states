"""Teacher-LLM alignment pipeline for GSM8K dataset.

Preprocesses GSM8K into chunk-aligned training data using Anthropic's Claude
as the teacher LLM to determine which reasoning steps are inferable from
which parts of the question.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from transformers import GPT2Tokenizer

from config import (
    GSM8K_CACHE_DIR, GSM8K_CHUNK_SIZE, TEACHER_MODEL, MODEL_NAME
)

# Load environment variables from .env if present
load_dotenv()

GEMINI_MAX_CONCURRENCY = int(os.getenv("GEMINI_MAX_CONCURRENCY", "2"))
GEMINI_MIN_DELAY = float(os.getenv("GEMINI_MIN_DELAY", "0.2"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
GEMINI_RATE_LIMIT_BACKOFF = float(os.getenv("GEMINI_RATE_LIMIT_BACKOFF", "10.0"))


def parse_gsm8k_answer(answer_text: str) -> dict:
    """
    Split GSM8K answer on '####' to get reasoning steps + final numeric answer.
    Extract calculation trace from <<...>> annotations.

    Args:
        answer_text: Raw answer string from GSM8K dataset

    Returns:
        dict with 'steps' (list of str) and 'final_answer' (str)
    """
    # Split on ####
    parts = answer_text.split("####")
    if len(parts) == 2:
        steps_text = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        # No #### delimiter - treat entire text as steps, try to find number
        steps_text = answer_text.strip()
        numbers = re.findall(r'-?[\d,]+\.?\d*', answer_text)
        final_answer = numbers[-1] if numbers else ""

    # Extract calculation trace from <<...>>
    calc_trace = re.findall(r'<<[^<>]+>>', steps_text)
    # Strip <<calc>> annotations from steps
    steps_text = re.sub(r'<<.*?>>', '', steps_text)

    # Split into individual steps
    steps = [s.strip() for s in steps_text.split('\n') if s.strip()]

    # Clean final answer (remove commas from numbers)
    final_answer = final_answer.replace(',', '')

    return {
        "steps": steps,
        "final_answer": final_answer,
        "calc_trace": calc_trace,
    }


def build_alignment_prompt(question: str, steps: list) -> str:
    """
    Build prompt asking teacher to insert <THINK> markers at the earliest
    position where each calculation becomes inferable.

    Args:
        question: The GSM8K question text
        steps: List of reasoning step strings

    Returns:
        Prompt string for teacher LLM
    """
    steps_formatted = ", ".join(steps)

    prompt = f"""You are an expert in computational linguistics. Your task is to augment a given query with thinking markers (<THINK>) based on a provided reasoning trace. This involves identifying the precise locations in the text where specific calculations can be performed.
Task Instructions:
1. Analyze Inputs: You will receive two inputs:
• A Query: A question or word problem.
• A Reasoning Trace: An ordered list of strings, where each string is a distinct mathematical calculation required to solve the query.
2. Produce Two Outputs:
• query: This is the original Query text, but with the special token <THINK> inserted at specific locations.
• thinking: This is the ordered list of calculations.
3. Rules for <THINK> Placement:
• The number of <THINK> tokens inserted into the query must be exactly equal to the number of calculations in the Reasoning Trace.
• The order of the <THINK> tokens in the query must correspond one-to-one with the order of the calculations in the Reasoning Trace.
• For each calculation, you must insert its corresponding <THINK> token at the earliest possible location in the query. This location is defined as the point immediately after the word or phrase that provides the final piece of information needed to perform that specific calculation.
4. Rule for the thinking List:
• The thinking output list is simply a direct copy of the ordered list of calculations provided in the Input Reasoning Trace.
5. Output Format:
• The output should be provided as a python dictionary with two keys: query and thinking.
• The values should be wrapped in <<>> to support easy parsing.
• Example:
{{
"query": "<<Your modified query here>>",
"thinking": <<['<<calculation1>>', '<<calculation2>>', ...]>>
}}
Examples
Example 1:
• Input Query: 'Hannah has three dogs. The first dog eats 1.5 cups of dog food a day. The second dog eats twice as much while the third dog eats 2.5 cups more than the second dog. How many cups of dog food should Hannah prepare in a day for her three dogs?'
• Input Reasoning Trace: ['<<1.5*2=3>>', '<<3+2.5=5.5>>', '<<1.5+3+5.5=10>>']
Required Output:
{{
"query": <<'Hannah has three dogs. The first dog eats 1.5 cups of dog food a day. The second dog eats twice as much<THINK> while the third dog eats 2.5 cups more than the second dog.<THINK> How many cups of dog food should Hannah prepare in a day for her three dogs?<THINK> '>>,
"thinking": <<['<<1.5*2=3>>', '<<3+2.5=5.5>>', '<<1.5+3+5.5=10>>']>>
}}
Task
Now, perform this transformation for the following input and return your output as a python dictionary.
Query: <<QUERY>>
Reasoning Trace: <<REASONING TRACE>>
Produce your output as a python dictionary in the specified format below:
{{
"query": <<ADD YOUR OUTPUT HERE>>,
"thinking": <<ADD YOUR OUTPUT HERE>>
}}

Query: {question}
Reasoning Trace: [{steps_formatted}]"""

    return prompt


def call_teacher_llm(prompt: str) -> str:
    """
    Call Anthropic API to get alignment response.

    Args:
        prompt: The alignment prompt

    Returns:
        Response text from Claude
    """
    if TEACHER_MODEL.startswith("gemini"):
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set for Gemini.")

        client = genai.Client(api_key=api_key)
        backoff = 1.0
        last_err = None
        for _ in range(GEMINI_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=TEACHER_MODEL,
                    contents=prompt,
                )
                return response.text or ""
            except Exception as e:
                last_err = e
                err_text = str(e).lower()
                if "rate" in err_text and "limit" in err_text:
                    time.sleep(GEMINI_RATE_LIMIT_BACKOFF)
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
        raise last_err

    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


async def call_teacher_llm_async(client, prompt: str) -> str:
    """Async version of call_teacher_llm."""
    if TEACHER_MODEL.startswith("gemini"):
        return await asyncio.to_thread(call_teacher_llm, prompt)

    message = await client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def parse_alignment_response(response: str, num_steps: int, question: str) -> tuple[str, list] | None:
    """
    Parse <THINK> marker positions and thinking list from LLM output.

    Args:
        response: Raw LLM response text
        num_steps: Expected number of steps

    Returns:
        List of character positions (one per step)
    """
    # find marker positions while aligning to the original question
    text = response.strip()
    query_match = re.search(r'"query"\s*:\s*<<(.+?)>>', text, re.DOTALL)
    thinking_match = re.search(r'"thinking"\s*:\s*<<(\[.+?\])>>', text, re.DOTALL)
    if not query_match or not thinking_match:
        return None

    query_with_markers = query_match.group(1).strip()
    thinking_raw = thinking_match.group(1).strip()

    try:
        thinking_list = json.loads(thinking_raw.replace("'", '"'))
    except Exception:
        return None

    if len(thinking_list) != num_steps:
        return None

    # validate that removing markers recovers the original question
    if query_with_markers.replace("<THINK>", "") != question:
        return None

    return query_with_markers, thinking_list


def char_pos_to_token_pos(question: str, char_positions: list, tokenizer) -> list:
    """
    Convert character positions to token positions.
    Build char->token map by iterating through decoded tokens.

    Args:
        question: Original question text
        char_positions: List of character positions
        tokenizer: GPT2 tokenizer

    Returns:
        List of token positions
    """
    tokens = tokenizer.encode(question, add_special_tokens=False)

    # Build character offset for each token
    char_offset = 0
    token_char_starts = []
    for tok_id in tokens:
        token_text = tokenizer.decode([tok_id])
        token_char_starts.append(char_offset)
        char_offset += len(token_text)

    # Convert each char position to token position
    token_positions = []
    for char_pos in char_positions:
        # Find the token that contains this character position
        tok_pos = len(tokens) - 1  # default to last token
        for t_idx, t_start in enumerate(token_char_starts):
            if t_start > char_pos:
                tok_pos = max(0, t_idx - 1)
                break
        token_positions.append(tok_pos)

    return token_positions


def align_single_example(question: str, answer: str, tokenizer) -> dict:
    """
    Full alignment pipeline for one GSM8K example.

    Args:
        question: GSM8K question text
        answer: GSM8K answer text (with #### delimiter)
        tokenizer: GPT2 tokenizer

    Returns:
        dict with alignment data, or None on failure
    """
    parsed = parse_gsm8k_answer(answer)
    steps = parsed["steps"]
    final_answer = parsed["final_answer"]
    calc_trace = parsed["calc_trace"]

    if not calc_trace:
        return None

    # Get token-level info
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    num_question_tokens = len(question_tokens)
    chunk_size = GSM8K_CHUNK_SIZE
    num_chunks = (num_question_tokens + chunk_size - 1) // chunk_size

    if num_chunks == 0:
        return None

    # Call teacher LLM for alignment
    prompt = build_alignment_prompt(question, calc_trace)
    try:
        response = call_teacher_llm(prompt)
        parsed_out = parse_alignment_response(response, len(calc_trace), question)
    except Exception as e:
        print(f"Teacher LLM call failed: {e}. Using uniform fallback.")
        parsed_out = None

    if parsed_out is None:
        return None

    query_with_markers, thinking_list = parsed_out
    if thinking_list != calc_trace:
        return None

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

    # Convert to token positions
    token_positions = char_pos_to_token_pos(question, char_positions, tokenizer)

    # Assign steps to chunks based on token positions
    step_chunk_assignments = []
    for tok_pos in token_positions:
        chunk_idx = min(tok_pos // chunk_size, num_chunks - 1)
        step_chunk_assignments.append(chunk_idx)

    # Build chunk_thoughts: concatenate steps assigned to each chunk
    chunk_thoughts = []
    for c in range(num_chunks):
        assigned_steps = [calc_trace[i] for i, ca in enumerate(step_chunk_assignments) if ca == c]
        if assigned_steps:
            chunk_thoughts.append(" ".join(assigned_steps))
        else:
            chunk_thoughts.append("")

    return {
        "question": question,
        "final_answer": final_answer,
        "num_chunks": num_chunks,
        "chunk_thoughts": chunk_thoughts,
        "steps": steps,
        "step_chunk_assignments": step_chunk_assignments,
    }


async def align_single_example_async(
    client,
    question: str,
    answer: str,
    tokenizer
) -> dict:
    """Async version of align_single_example."""
    parsed = parse_gsm8k_answer(answer)
    steps = parsed["steps"]
    final_answer = parsed["final_answer"]
    calc_trace = parsed["calc_trace"]

    if not calc_trace:
        return None

    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    num_question_tokens = len(question_tokens)
    chunk_size = GSM8K_CHUNK_SIZE
    num_chunks = (num_question_tokens + chunk_size - 1) // chunk_size

    if num_chunks == 0:
        return None

    prompt = build_alignment_prompt(question, calc_trace)
    try:
        response = await call_teacher_llm_async(client, prompt)
        parsed_out = parse_alignment_response(response, len(calc_trace), question)
    except Exception as e:
        print(f"Teacher LLM call failed: {e}. Using uniform fallback.")
        parsed_out = None

    if parsed_out is None:
        return None

    query_with_markers, thinking_list = parsed_out
    if thinking_list != calc_trace:
        return None

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

    step_chunk_assignments = []
    for tok_pos in token_positions:
        chunk_idx = min(tok_pos // chunk_size, num_chunks - 1)
        step_chunk_assignments.append(chunk_idx)

    chunk_thoughts = []
    for c in range(num_chunks):
        assigned_steps = [calc_trace[i] for i, ca in enumerate(step_chunk_assignments) if ca == c]
        if assigned_steps:
            chunk_thoughts.append(" ".join(assigned_steps))
        else:
            chunk_thoughts.append("")

    return {
        "question": question,
        "final_answer": final_answer,
        "num_chunks": num_chunks,
        "chunk_thoughts": chunk_thoughts,
        "steps": steps,
        "step_chunk_assignments": step_chunk_assignments,
    }


async def _align_batch_async(examples: list, tokenizer, concurrency: int = 10):
    """Process a batch of examples concurrently."""
    client = None
    if not TEACHER_MODEL.startswith("gemini"):
        import anthropic
        client = anthropic.AsyncAnthropic()
    else:
        concurrency = min(concurrency, GEMINI_MAX_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(q, a):
        async with semaphore:
            result = await align_single_example_async(client, q, a, tokenizer)
            if TEACHER_MODEL.startswith("gemini") and GEMINI_MIN_DELAY > 0:
                await asyncio.sleep(GEMINI_MIN_DELAY)
            return result

    tasks = [
        process_one(ex["question"], ex["answer"])
        for ex in examples
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


def align_dataset(split: str = "train", tokenizer=None, max_examples: int = None, output_suffix: str = ""):
    """
    Process full GSM8K split with teacher LLM alignment.

    Saves as JSONL to GSM8K_CACHE_DIR/{split}_aligned.jsonl.
    Supports resume on interruption.

    Args:
        split: "train" or "test"
        tokenizer: GPT2 tokenizer (created if None)
    """
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # Ensure cache dir exists
    cache_dir = Path(GSM8K_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / f"{split}_aligned{output_suffix}.jsonl"

    # Load dataset
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    print(f"Loaded {len(dataset)} examples")

    # Check for resume
    existing_count = 0
    if output_path.exists():
        with open(output_path) as f:
            existing_count = sum(1 for _ in f)
        print(f"Found {existing_count} existing aligned examples, resuming...")

    if existing_count >= len(dataset):
        print("All examples already aligned!")
        return

    # Process remaining examples
    remaining = list(dataset)[existing_count:]
    if max_examples is not None:
        remaining = remaining[:max_examples]
    batch_size = 10

    print(f"Aligning {len(remaining)} remaining examples...")
    with open(output_path, "a") as f:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]

            results = asyncio.run(_align_batch_async(batch, tokenizer))

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Error on example {existing_count + batch_start + i}: {result}")
                    continue
                if result is None:
                    continue

                f.write(json.dumps(result) + "\n")
                f.flush()

            done = existing_count + batch_start + len(batch)
            total = existing_count + len(remaining)
            print(f"Progress: {done}/{total} ({done/total:.1%})")

    print(f"Alignment complete! Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align GSM8K with a teacher LLM.")
    parser.add_argument("--test-only", action="store_true", help="Run a single-example test.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples per split.")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output JSONL files.")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent requests (Gemini only).")
    parser.add_argument("--min-delay", type=float, default=None, help="Min delay between Gemini requests.")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    if args.test_only:
        # Quick test with one example
        print("Testing alignment on a single example...")
        result = align_single_example(
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
            "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = <<9*2=18>>$18 every day at the farmer's market.\n#### 18",
            tokenizer
        )
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Alignment failed!")
    else:
        # Full dataset alignment (optionally limited)
        if args.concurrency is not None:
            GEMINI_MAX_CONCURRENCY = args.concurrency
        if args.min_delay is not None:
            GEMINI_MIN_DELAY = args.min_delay

        align_dataset("train", tokenizer, max_examples=args.max_examples, output_suffix=args.output_suffix)
        align_dataset("test", tokenizer, max_examples=args.max_examples, output_suffix=args.output_suffix)
