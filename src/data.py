"""Parity and GSM8K datasets for Thinking States."""

import random
import json
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from config import (
    CHUNK_SIZE, MIN_FLIPS, MAX_FLIPS, MAX_THOUGHT_LEN,
    GSM8K_CACHE_DIR, GSM8K_MAX_THOUGHT_LEN, BOS_TOKEN, EOS_TOKEN, GSM8K_ALIGNED_SUFFIX
)


def generate_parity_example(
    num_ops: int,
    chunk_size: int = CHUNK_SIZE,
    tokenizer: PreTrainedTokenizer = None
) -> Dict:
    """
    Generate a single parity task example matching the paper (Appendix A.4).

    Paper format:
        "The coin starts at state heads. Alice doesn't flip the coin.
         Bob flips the coin. Alice flips the coin.
         The final state of the coin is heads."

    Each operation gets a thought = current state after that operation.
    Thoughts are assigned to chunks based on the token position of each op.

    Args:
        num_ops: number of operations (each is a flip or no-flip)
        chunk_size: tokens per chunk
        tokenizer: tokenizer for token-position alignment

    Returns:
        Dict with keys: input_text, answer, chunk_thoughts, num_chunks
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    entities = ["Alice", "Bob"]

    # Generate random flip/no-flip operations
    operations = []
    for i in range(num_ops):
        entity = entities[i % 2]
        does_flip = random.choice([True, False])
        operations.append((entity, does_flip))

    # Build input text (paper format)
    input_text = "The coin starts at state heads."
    for entity, does_flip in operations:
        if does_flip:
            input_text += f" {entity} flips the coin."
        else:
            input_text += f" {entity} doesn't flip the coin."

    # Compute state after each operation
    state = "heads"
    states_after_op = []
    for _, does_flip in operations:
        if does_flip:
            state = "tails" if state == "heads" else "heads"
        states_after_op.append(state)

    answer = state
    input_text += f" The final state of the coin is"

    # Tokenize to find token positions of each operation's end
    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    num_tokens = len(tokens)
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size

    # Find token position where each operation ends (last token of that sentence)
    op_token_positions = []
    text_so_far = "The coin starts at state heads."
    for entity, does_flip in operations:
        if does_flip:
            text_so_far += f" {entity} flips the coin."
        else:
            text_so_far += f" {entity} doesn't flip the coin."
        new_pos = len(tokenizer.encode(text_so_far, add_special_tokens=False))
        op_token_positions.append(new_pos - 1)  # last token of this operation

    # Assign thoughts to chunks: concatenate all thoughts whose operation
    # falls within that chunk (same logic as GSM8K token-to-chunk alignment)
    chunk_thoughts = []
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, num_tokens)

        thoughts_in_chunk = []
        for op_idx, op_pos in enumerate(op_token_positions):
            if chunk_start <= op_pos < chunk_end:
                thoughts_in_chunk.append(states_after_op[op_idx])

        # Concatenate thoughts in this chunk (like GSM8K), or empty string
        chunk_thoughts.append(" ".join(thoughts_in_chunk) if thoughts_in_chunk else "")

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
        tokenizer: PreTrainedTokenizer = None
    ):
        self.size = size
        self.min_flips = min_flips
        self.max_flips = max_flips
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer = tokenizer

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


def collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict:
    """Collate function for parity DataLoader.

    Produces the same output format as gsm8k_collate_fn: chunk_thought_ids
    and chunk_thought_masks with BOS/EOS wrapping so the unified Transformer
    T/C blocks can be used for all tasks.
    """
    bos_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError("BOS/EOS tokens not found in tokenizer. Ensure model adds them.")

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

    # Build chunk_thought_ids and chunk_thought_masks (same format as GSM8K)
    max_num_chunks = max(ex["num_chunks"] for ex in batch)

    chunk_thought_texts = []
    for ex in batch:
        thoughts = list(ex["chunk_thoughts"])
        while len(thoughts) < max_num_chunks:
            thoughts.append("")
        chunk_thought_texts.append(thoughts)

    chunk_thought_ids = []
    chunk_thought_masks = []

    max_thought_len = MAX_THOUGHT_LEN

    for chunk_idx in range(max_num_chunks):
        thought_ids_list = []
        for b in range(len(batch)):
            thought_text = chunk_thought_texts[b][chunk_idx].strip()
            thought_ids = [bos_id]
            if thought_text:
                thought_ids += tokenizer.encode(thought_text, add_special_tokens=False)
            thought_ids.append(eos_id)

            if len(thought_ids) > max_thought_len:
                thought_ids = thought_ids[:max_thought_len]
                thought_ids[-1] = eos_id

            thought_ids_list.append(thought_ids)

        max_len = min(max_thought_len, max(len(ids) for ids in thought_ids_list))
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


def get_collate_fn(tokenizer: PreTrainedTokenizer):
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
        tokenizer: PreTrainedTokenizer = None,
        limit: int = None,
        aligned_suffix: str = GSM8K_ALIGNED_SUFFIX
    ):
        self.split = split
        self.cache_dir = Path(cache_dir)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer = tokenizer
        self.aligned_suffix = aligned_suffix

        path = self.cache_dir / f"{split}_aligned{self.aligned_suffix}.jsonl"
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


def gsm8k_collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict:
    """Collate function for GSM8K DataLoader."""
    bos_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError("BOS/EOS tokens not found in tokenizer. Ensure model adds them.")

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

    labels = input_ids.clone()
    for i, prompt in enumerate(input_texts):
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        labels[i, :prompt_len] = -100

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


def get_gsm8k_collate_fn(tokenizer: PreTrainedTokenizer):
    """Returns a GSM8K collate function with the tokenizer bound."""
    def _collate(batch):
        return gsm8k_collate_fn(batch, tokenizer)
    return _collate
