"""Sequential chunk-by-chunk inference engine for Thinking States."""

import torch
from tqdm import tqdm

from config import (
    DEVICE, CHUNK_SIZE, HIDDEN_DIM,
    TEST_MIN_FLIPS, TEST_MAX_FLIPS,
    TASK, GSM8K_MAX_ANSWER_TOKENS, BOS_TOKEN, EOS_TOKEN
)
from src.data import generate_parity_example, GSM8KDataset
from src.model import ThinkingStatesModel


class ThinkingStatesInference:
    """
    Sequential chunk-by-chunk inference for Thinking States.

    During inference:
    1. Process input in chunks
    2. For each chunk: inject state, forward, generate thought, compress to new state
    3. Generate final answer

    All tasks use the same unified T.generate() / C.forward() interface.
    """

    def __init__(self, model: ThinkingStatesModel):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = next(model.parameters()).device
        self.chunk_size = model.chunk_size
        self.hidden_dim = model.hidden_dim
        self.task = model.task
        # All tasks now use BOS/EOS for the autoregressive T block
        self.bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    @torch.no_grad()
    def infer(self, input_text: str, verbose: bool = False, max_answer_tokens: int = None) -> dict:
        """Run inference on a single example.

        Processes input chunks sequentially. Each chunk sees all previous tokens
        via causal attention (paper Fig. 2a: "Chunk representations can access
        the history via attention layers and the KV-cache"). States from previous
        chunks are injected at the correct positions matching the training setup.
        """
        self.model.eval()

        tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        num_tokens = len(tokens)
        num_chunks = (num_tokens + self.chunk_size - 1) // self.chunk_size

        thoughts = []
        # Maps chunk_idx -> compressed state tensor from that chunk's thought.
        # Used to build the injection tensor for subsequent chunks.
        chunk_state_map = {}

        for chunk_idx in range(num_chunks):
            end_pos = min((chunk_idx + 1) * self.chunk_size, num_tokens)
            # Feed all tokens up to and including the current chunk so that
            # causal attention can see the full history (matching training).
            prefix_tokens = tokens[:end_pos]

            input_ids = torch.tensor([prefix_tokens], device=self.device)
            seq_len = len(prefix_tokens)

            # Build state tensor for all positions up to current chunk.
            # State from chunk i's thought is injected into chunk i+1.
            state_tensor = torch.zeros(1, seq_len, self.hidden_dim, device=self.device)
            for prev_idx, prev_state in chunk_state_map.items():
                target_chunk = prev_idx + 1
                start = target_chunk * self.chunk_size
                end = min(start + self.chunk_size, seq_len)
                if start >= seq_len:
                    break
                actual_len = end - start
                state_tensor[:, start:end, :] = prev_state[:, :actual_len, :]

            self.model._state_to_inject = state_tensor

            _ = self.model.backbone(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
            )

            extracted_hidden = self.model._extracted_hidden

            if extracted_hidden is not None:
                # Extract only the current chunk's hidden states for T block
                chunk_start = chunk_idx * self.chunk_size
                chunk_hidden = extracted_hidden[:, chunk_start:end_pos, :]

                if chunk_hidden.shape[1] < self.chunk_size:
                    pad_len = self.chunk_size - chunk_hidden.shape[1]
                    padding = torch.zeros(1, pad_len, self.hidden_dim, device=self.device)
                    chunk_hidden = torch.cat([chunk_hidden, padding], dim=1)

                thought_ids = self.model.thinking_block.generate(
                    chunk_hidden[:, :self.chunk_size, :],
                    self.bos_id,
                    self.eos_id,
                    max_len=self.model.thinking_block.max_thought_len,
                )
                thought_text = self.tokenizer.decode(
                    thought_ids[0], skip_special_tokens=True
                ).strip()
                thoughts.append(thought_text)

                if verbose:
                    print(f"Chunk {chunk_idx}: thought = '{thought_text}'")

                thought_mask = torch.ones_like(thought_ids)
                state = self.model.compression_block(thought_ids, attention_mask=thought_mask)
                chunk_state_map[chunk_idx] = state

        self.model._state_to_inject = None
        self.model._extracted_hidden = None

        if max_answer_tokens is None:
            max_answer_tokens = GSM8K_MAX_ANSWER_TOKENS if self.task == "gsm8k" else 3

        generated = self._generate_answer(tokens, chunk_state_map, max_answer_tokens)
        predicted_answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return {
            "predicted_answer": predicted_answer,
            "thoughts": thoughts,
        }

    def _generate_answer(self, prompt_tokens, chunk_state_map, max_new_tokens):
        """Autoregressively generate answer tokens with states injected."""
        generated = []
        for _ in range(max_new_tokens):
            full_tokens = prompt_tokens + generated
            seq_len = len(full_tokens)

            # Build state tensor matching training alignment:
            # state from chunk i's thought -> injected at chunk i+1
            state_tensor = torch.zeros(1, seq_len, self.hidden_dim, device=self.device)
            for chunk_idx, chunk_state in chunk_state_map.items():
                target_chunk = chunk_idx + 1
                start_pos = target_chunk * self.chunk_size
                end_pos = min(start_pos + self.chunk_size, seq_len)
                if start_pos >= seq_len:
                    continue
                state_tensor[:, start_pos:end_pos, :] = chunk_state[:, :end_pos - start_pos, :]

            self.model._state_to_inject = state_tensor
            input_ids = torch.tensor([full_tokens], device=self.device)
            outputs = self.model.backbone(input_ids=input_ids)
            logits = outputs.logits
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        self.model._state_to_inject = None
        return generated


def evaluate_length_generalization(
    model: ThinkingStatesModel,
    min_flips: int = TEST_MIN_FLIPS,
    max_flips: int = TEST_MAX_FLIPS,
    num_examples: int = 100
):
    """Evaluate model on longer sequences than training."""
    print(f"\nEvaluating length generalization ({min_flips}-{max_flips} flips)...")

    inference = ThinkingStatesInference(model)
    tokenizer = model.tokenizer

    correct = 0
    total = 0

    for i in tqdm(range(num_examples)):
        num_flips = min_flips + (i % (max_flips - min_flips + 1))
        example = generate_parity_example(num_flips, CHUNK_SIZE, tokenizer)

        result = inference.infer(example["input_text"], verbose=False)

        pred = result["predicted_answer"].lower().strip()
        true = example["answer"].lower().strip()

        if pred.startswith(true) or true.startswith(pred):
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Length generalization accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy
