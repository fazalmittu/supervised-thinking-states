"""Sequential inference and evaluation for Thinking States."""

import torch
from tqdm import tqdm

from config import (
    DEVICE, CHUNK_SIZE, HIDDEN_DIM,
    TEST_MIN_FLIPS, TEST_MAX_FLIPS
)
from data import ParityDataset, generate_parity_example
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

    @torch.no_grad()
    def infer(self, input_text: str, verbose: bool = False) -> dict:
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
        num_chunks = (num_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Initialize state
        state = torch.zeros(1, CHUNK_SIZE, HIDDEN_DIM, device=self.device)

        thoughts = []

        # Process chunk by chunk
        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * CHUNK_SIZE
            end_pos = min(start_pos + CHUNK_SIZE, num_tokens)
            chunk_tokens = tokens[start_pos:end_pos]

            # Pad chunk if needed
            if len(chunk_tokens) < CHUNK_SIZE:
                chunk_tokens = chunk_tokens + [self.tokenizer.pad_token_id] * (CHUNK_SIZE - len(chunk_tokens))

            chunk_ids = torch.tensor([chunk_tokens], device=self.device)

            # Build state tensor for this chunk
            state_tensor = torch.zeros(1, len(chunk_tokens), HIDDEN_DIM, device=self.device)
            state_tensor[:, :min(CHUNK_SIZE, len(chunk_tokens)), :] = state[:, :min(CHUNK_SIZE, len(chunk_tokens)), :]

            # Set state for injection
            self.model._state_to_inject = state_tensor

            # Forward through backbone
            _ = self.model.backbone(input_ids=chunk_ids, output_hidden_states=True)

            # Get extracted hidden states
            extracted_hidden = self.model._extracted_hidden

            # Generate thought
            if extracted_hidden is not None:
                # Pad to chunk_size if needed
                if extracted_hidden.shape[1] < CHUNK_SIZE:
                    pad_len = CHUNK_SIZE - extracted_hidden.shape[1]
                    padding = torch.zeros(1, pad_len, HIDDEN_DIM, device=self.device)
                    extracted_hidden = torch.cat([extracted_hidden, padding], dim=1)

                thought_ids = self.model.thinking_block.generate(
                    extracted_hidden[:, :CHUNK_SIZE, :],
                    self.tokenizer,
                    max_len=4
                )

                thought_text = self.tokenizer.decode(thought_ids[0], skip_special_tokens=True)
                thoughts.append(thought_text)

                if verbose:
                    print(f"Chunk {chunk_idx}: thought = '{thought_text}'")

                # Compress to new state
                state = self.model.compression_block(thought_ids)

        # Clear state
        self.model._state_to_inject = None
        self.model._extracted_hidden = None

        # Generate answer
        input_ids = torch.tensor([tokens], device=self.device)
        outputs = self.model.backbone.generate(
            input_ids,
            max_new_tokens=3,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )

        generated = outputs[0, len(tokens):]
        predicted_answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return {
            "predicted_answer": predicted_answer,
            "thoughts": thoughts
        }


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
        model.load_state_dict(torch.load("thinking_states_model.pt", map_location=DEVICE))
        print("Loaded trained weights.")
    except FileNotFoundError:
        print("No trained weights found. Using untrained model (for testing).")

    model.to(DEVICE)
    model.eval()

    # Demo inference
    demo_inference(model)

    # Length generalization test
    evaluate_length_generalization(model)


if __name__ == "__main__":
    main()
