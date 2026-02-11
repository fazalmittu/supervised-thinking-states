"""Parity and GSM8K datasets for Thinking States."""

import random
import json
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from config import (
    CHUNK_SIZE, MIN_FLIPS, MAX_FLIPS,
    GSM8K_CACHE_DIR, GSM8K_MAX_THOUGHT_LEN, BOS_TOKEN, EOS_TOKEN
)


def generate_parity_example(
    num_flips: int,
    chunk_size: int = CHUNK_SIZE,
    tokenizer: GPT2Tokenizer = None
) -> Dict:
    """
    Generate a single parity task example.

    The task: Start with 'heads', apply num_flips flips, predict final state.
    Ground truth thoughts are the state after each chunk of tokens.

    Returns:
        Dict with keys: input_text, answer, chunk_thoughts
    """
    # Build input text
    input_text = "Start: heads."
    for _ in range(num_flips):
        input_text += " Flip."
    input_text += " Answer:"

    # Compute states after each flip
    state = "heads"
    states_after_flip = []
    for _ in range(num_flips):
        state = "tails" if state == "heads" else "heads"
        states_after_flip.append(state)

    answer = state

    # Now we need to assign thoughts to chunks based on token positions
    # Tokenize to get token positions
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    num_tokens = len(tokens)
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size

    # Find which tokens correspond to "Flip." occurrences
    # We'll assign the thought (state after flip) to the chunk containing that flip
    flip_token_positions = []
    text_so_far = "Start: heads."
    current_pos = len(tokenizer.encode(text_so_far, add_special_tokens=False))

    for i in range(num_flips):
        text_so_far += " Flip."
        new_pos = len(tokenizer.encode(text_so_far, add_special_tokens=False))
        # The flip ends at new_pos - 1
        flip_token_positions.append(new_pos - 1)

    # Assign thoughts to chunks
    # For each chunk, the thought is the state after the last flip in that chunk
    # If no flip in chunk, thought is empty or previous state
    chunk_thoughts = []
    last_state = "heads"

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, num_tokens)

        # Find flips that end in this chunk
        chunk_state = last_state
        for flip_idx, flip_pos in enumerate(flip_token_positions):
            if chunk_start <= flip_pos < chunk_end:
                chunk_state = states_after_flip[flip_idx]

        chunk_thoughts.append(chunk_state)
        last_state = chunk_state

    return {
        "input_text": input_text,
        "answer": answer,
        "chunk_thoughts": chunk_thoughts,
        "num_chunks": num_chunks,
    }


class ParityDataset(Dataset):
    """Dataset for the parity task."""

    def __init__(
        self,
        size: int,
        min_flips: int = MIN_FLIPS,
        max_flips: int = MAX_FLIPS,
        tokenizer: GPT2Tokenizer = None
    ):
        self.size = size
        self.min_flips = min_flips
        self.max_flips = max_flips
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")

        # Pre-generate examples
        self.examples = []
        for _ in range(size):
            num_flips = random.randint(min_flips, max_flips)
            example = generate_parity_example(num_flips, CHUNK_SIZE, self.tokenizer)
            self.examples.append(example)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_fn(batch: List[Dict], tokenizer: GPT2Tokenizer) -> Dict:
    """
    Collate function for DataLoader.

    Returns:
        Dict with:
        - input_ids: (batch, max_seq_len)
        - attention_mask: (batch, max_seq_len)
        - labels: (batch, max_seq_len) with -100 for non-answer tokens
        - chunk_thought_ids: List of (batch, num_chunks) lists of token ids
        - num_chunks: max number of chunks
    """
    # Tokenize inputs with answers
    full_texts = [ex["input_text"] + " " + ex["answer"] for ex in batch]

    encoded = tokenizer(
        full_texts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Create labels: -100 for input, actual ids for answer
    labels = input_ids.clone()
    for i, ex in enumerate(batch):
        input_len = len(tokenizer.encode(ex["input_text"], add_special_tokens=False))
        labels[i, :input_len] = -100

    # Tokenize chunk thoughts
    max_num_chunks = max(ex["num_chunks"] for ex in batch)

    # Pad chunk_thoughts to same length
    chunk_thought_texts = []
    for ex in batch:
        thoughts = ex["chunk_thoughts"]
        # Pad with empty thought if needed
        while len(thoughts) < max_num_chunks:
            thoughts = thoughts + [thoughts[-1]]  # Repeat last state
        chunk_thought_texts.append(thoughts)

    # Tokenize each chunk's thought
    chunk_thought_ids = []
    for chunk_idx in range(max_num_chunks):
        thoughts_for_chunk = [chunk_thought_texts[b][chunk_idx] for b in range(len(batch))]
        encoded_thoughts = tokenizer(
            thoughts_for_chunk,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        chunk_thought_ids.append(encoded_thoughts["input_ids"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "chunk_thought_ids": chunk_thought_ids,  # List of (batch, thought_len) tensors
        "num_chunks": max_num_chunks,
    }


def get_collate_fn(tokenizer: GPT2Tokenizer):
    """Returns a collate function with the tokenizer bound."""
    def _collate(batch):
        return collate_fn(batch, tokenizer)
    return _collate


class GSM8KDataset(Dataset):
    """Dataset for GSM8K with teacher-aligned chunk thoughts."""

    def __init__(
        self,
        split: str = "train",
        cache_dir: str = GSM8K_CACHE_DIR,
        tokenizer: GPT2Tokenizer = None,
        limit: int = None
    ):
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")

        path = self.cache_dir / f"{split}_aligned.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing GSM8K cache at {path}. Run align_gsm8k.py first."
            )

        self.examples = []
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                self.examples.append(json.loads(line))
                if limit is not None and len(self.examples) >= limit:
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def gsm8k_collate_fn(batch: List[Dict], tokenizer: GPT2Tokenizer) -> Dict:
    """
    Collate function for GSM8K DataLoader.

    Returns:
        Dict with:
        - input_ids: (batch, max_seq_len)
        - attention_mask: (batch, max_seq_len)
        - labels: (batch, max_seq_len) with -100 for non-answer tokens
        - chunk_thought_ids: List of (batch, thought_len) tensors
        - chunk_thought_masks: List of (batch, thought_len) tensors
        - num_chunks: max number of chunks
    """
    bos_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError("BOS/EOS tokens not found in tokenizer. Ensure model adds them.")

    # Build prompt + answer
    input_texts = [ex["question"].strip() + "\nAnswer:" for ex in batch]
    full_texts = [f"{input_texts[i]} {batch[i]['final_answer'].strip()}" for i in range(len(batch))]

    encoded = tokenizer(
        full_texts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Labels: -100 for prompt, real ids for answer
    labels = input_ids.clone()
    for i, prompt in enumerate(input_texts):
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        labels[i, :prompt_len] = -100

    # Pad chunk thoughts to same number of chunks
    max_num_chunks = max(ex["num_chunks"] for ex in batch)
    chunk_thought_texts = []
    for ex in batch:
        thoughts = list(ex["chunk_thoughts"])
        while len(thoughts) < max_num_chunks:
            thoughts.append("")
        chunk_thought_texts.append(thoughts)

    chunk_thought_ids = []
    chunk_thought_masks = []

    for chunk_idx in range(max_num_chunks):
        thought_ids_list = []
        for b in range(len(batch)):
            thought_text = chunk_thought_texts[b][chunk_idx].strip()
            thought_ids = [bos_id]
            if thought_text:
                thought_ids += tokenizer.encode(thought_text, add_special_tokens=False)
            thought_ids.append(eos_id)

            if len(thought_ids) > GSM8K_MAX_THOUGHT_LEN:
                thought_ids = thought_ids[:GSM8K_MAX_THOUGHT_LEN]
                thought_ids[-1] = eos_id

            thought_ids_list.append(thought_ids)

        max_len = min(
            GSM8K_MAX_THOUGHT_LEN,
            max(len(ids) for ids in thought_ids_list)
        )
        padded = []
        masks = []
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for ids in thought_ids_list:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        chunk_thought_ids.append(torch.tensor(padded, dtype=torch.long))
        chunk_thought_masks.append(torch.tensor(masks, dtype=torch.long))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "chunk_thought_ids": chunk_thought_ids,
        "chunk_thought_masks": chunk_thought_masks,
        "num_chunks": max_num_chunks,
    }


def get_gsm8k_collate_fn(tokenizer: GPT2Tokenizer):
    """Returns a GSM8K collate function with the tokenizer bound."""
    def _collate(batch):
        return gsm8k_collate_fn(batch, tokenizer)
    return _collate


if __name__ == "__main__":
    # Test the data generation
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Testing single example generation:")
    example = generate_parity_example(5, CHUNK_SIZE, tokenizer)
    print(f"Input: {example['input_text']}")
    print(f"Answer: {example['answer']}")
    print(f"Chunk thoughts: {example['chunk_thoughts']}")
    print(f"Num chunks: {example['num_chunks']}")

    print("\nTesting dataset:")
    dataset = ParityDataset(10, tokenizer=tokenizer)
    print(f"Dataset size: {len(dataset)}")
    print(f"First example: {dataset[0]}")
