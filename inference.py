"""Sequential inference and evaluation for Thinking States."""

import torch
from tqdm import tqdm

from config import (
    DEVICE, CHUNK_SIZE, HIDDEN_DIM,
    TEST_MIN_FLIPS, TEST_MAX_FLIPS,
    TASK, GSM8K_MAX_ANSWER_TOKENS, BOS_TOKEN, EOS_TOKEN
)
from data import generate_parity_example, GSM8KDataset
from model import ThinkingStatesModel


class ThinkingStatesInference:
    """
    Sequential chunk-by-chunk inference for Thinking States.

    During inference:
    1. Process input in chunks
    2. For each chunk: inject state, forward, generate thought, compress to new state
    3. Generate final answer
    """

    def __init__(self, model: ThinkingStatesModel):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = next(model.parameters()).device
        self.chunk_size = model.chunk_size
        self.hidden_dim = model.hidden_dim
        self.task = model.task
        if self.task == "gsm8k":
            self.bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
            self.eos_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        else:
            self.bos_id = None
            self.eos_id = None

    @torch.no_grad()
    def infer(self, input_text: str, verbose: bool = False, max_answer_tokens: int = None) -> dict:
        """
        Run inference on a single example.

        Args:
            input_text: The input prompt (e.g., "Start: heads. Flip. Flip. Answer:")
            verbose: Print intermediate thoughts

        Returns:
            dict with predicted_answer, thoughts
        """
        self.model.eval()

        # Tokenize input
        tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        num_tokens = len(tokens)
        num_chunks = (num_tokens + self.chunk_size - 1) // self.chunk_size

        # Initialize state
        state = torch.zeros(1, self.chunk_size, self.hidden_dim, device=self.device)

        thoughts = []
        chunk_states = []

        # Process chunk by chunk
        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * self.chunk_size
            end_pos = min(start_pos + self.chunk_size, num_tokens)
            chunk_tokens = tokens[start_pos:end_pos]

            chunk_ids = torch.tensor([chunk_tokens], device=self.device)
            attention_mask = torch.ones_like(chunk_ids)

            # Build state tensor for this chunk
            state_tensor = torch.zeros(1, len(chunk_tokens), self.hidden_dim, device=self.device)
            state_tensor[:, :len(chunk_tokens), :] = state[:, :len(chunk_tokens), :]

            # Set state for injection
            self.model._state_to_inject = state_tensor

            # Forward through backbone
            _ = self.model.backbone(
                input_ids=chunk_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Get extracted hidden states
            extracted_hidden = self.model._extracted_hidden

            # Generate thought
            if extracted_hidden is not None:
                # Pad to chunk_size if needed
                if extracted_hidden.shape[1] < self.chunk_size:
                    pad_len = self.chunk_size - extracted_hidden.shape[1]
                    padding = torch.zeros(1, pad_len, self.hidden_dim, device=self.device)
                    extracted_hidden = torch.cat([extracted_hidden, padding], dim=1)

                if self.task == "gsm8k":
                    thought_ids = self.model.thinking_block.generate(
                        extracted_hidden[:, :self.chunk_size, :],
                        self.bos_id,
                        self.eos_id,
                        max_len=self.model.thinking_block.max_thought_len
                    )
                    decoded = self.tokenizer.decode(
                        thought_ids[0],
                        skip_special_tokens=True
                    )
                    thought_text = decoded.strip()
                else:
                    thought_ids = self.model.thinking_block.generate(
                        extracted_hidden[:, :self.chunk_size, :],
                        self.tokenizer,
                        max_len=4
                    )
                    thought_text = self.tokenizer.decode(thought_ids[0], skip_special_tokens=True)
                thoughts.append(thought_text)

                if verbose:
                    print(f"Chunk {chunk_idx}: thought = '{thought_text}'")

                # Compress to new state
                if self.task == "gsm8k":
                    thought_mask = torch.ones_like(thought_ids)
                    state = self.model.compression_block(thought_ids, attention_mask=thought_mask)
                else:
                    state = self.model.compression_block(thought_ids)
                chunk_states.append(state)

        # Clear state
        self.model._state_to_inject = None
        self.model._extracted_hidden = None

        # Generate answer with manual loop so we can inject prompt states
        if max_answer_tokens is None:
            max_answer_tokens = GSM8K_MAX_ANSWER_TOKENS if self.task == "gsm8k" else 3

        generated = self._generate_answer(tokens, chunk_states, max_answer_tokens)
        predicted_answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return {
            "predicted_answer": predicted_answer,
            "thoughts": thoughts
        }

    def _generate_answer(self, prompt_tokens, chunk_states, max_new_tokens):
        generated = []
        for _ in range(max_new_tokens):
            full_tokens = prompt_tokens + generated
            seq_len = len(full_tokens)

            state_tensor = torch.zeros(1, seq_len, self.hidden_dim, device=self.device)
            for chunk_idx, chunk_state in enumerate(chunk_states):
                start_pos = chunk_idx * self.chunk_size
                end_pos = min(start_pos + self.chunk_size, seq_len)
                if start_pos >= seq_len:
                    break
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
    """
    Evaluate model on longer sequences than training.

    Args:
        model: Trained ThinkingStatesModel
        min_flips: Minimum number of flips (should be > training max)
        max_flips: Maximum number of flips
        num_examples: Number of examples to test
    """
    print(f"\nEvaluating length generalization ({min_flips}-{max_flips} flips)...")

    inference = ThinkingStatesInference(model)
    tokenizer = model.tokenizer

    correct = 0
    total = 0

    for i in tqdm(range(num_examples)):
        num_flips = min_flips + (i % (max_flips - min_flips + 1))
        example = generate_parity_example(num_flips, CHUNK_SIZE, tokenizer)

        result = inference.infer(example["input_text"], verbose=False)

        # Check if predicted answer matches
        pred = result["predicted_answer"].lower().strip()
        true = example["answer"].lower().strip()

        if pred.startswith(true) or true.startswith(pred):
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Length generalization accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy


def demo_inference(model: ThinkingStatesModel):
    """Run a few demo inferences with verbose output."""
    print("\n" + "="*50)
    print("Demo Inference")
    print("="*50)

    inference = ThinkingStatesInference(model)
    tokenizer = model.tokenizer

    test_cases = [
        ("Start: heads. Flip. Answer:", "tails"),
        ("Start: heads. Flip. Flip. Answer:", "heads"),
        ("Start: heads. Flip. Flip. Flip. Answer:", "tails"),
        ("Start: heads. Flip. Flip. Flip. Flip. Flip. Answer:", "tails"),
    ]

    for input_text, expected in test_cases:
        print(f"\nInput: {input_text}")
        print(f"Expected: {expected}")

        result = inference.infer(input_text, verbose=True)

        print(f"Predicted: {result['predicted_answer']}")
        print(f"Thoughts: {result['thoughts']}")

        is_correct = result['predicted_answer'].lower().strip().startswith(expected)
        print(f"Correct: {is_correct}")


def main():
    # Load trained model
    print("Loading model...")
    model = ThinkingStatesModel()

    try:
        # Load to CPU first to avoid MPS unaligned blit issues
        state_dict = torch.load("thinking_states_model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        print("Loaded trained weights.")
    except FileNotFoundError:
        print("No trained weights found. Using untrained model (for testing).")

    model.to(DEVICE)
    model.eval()

    if TASK == "gsm8k":
        print("Running GSM8K demo inference...")
        dataset = GSM8KDataset("test", tokenizer=model.tokenizer, limit=3)
        inference = ThinkingStatesInference(model)
        for ex in dataset:
            question = ex["question"].strip()
            prompt = question + "\nAnswer:"
            print("\nQuestion:")
            print(question)
            result = inference.infer(prompt, verbose=True)
            print(f"Predicted answer: {result['predicted_answer']}")
    else:
        # Demo inference
        demo_inference(model)

        # Length generalization test
        evaluate_length_generalization(model)


if __name__ == "__main__":
    main()
