"""ThinkingStatesModel wrapper that handles state injection."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import HIDDEN_DIM, CHUNK_SIZE, L_IN, L_OUT, DEVICE, MODEL_NAME
from modules import ThinkingBlock, CompressionBlock


class ThinkingStatesModel(nn.Module):
    """
    Wrapper around GPT-2 that adds Thinking States functionality.

    Key features:
    - State injection at layer L_IN via forward hook
    - Hidden state extraction at layer L_OUT
    - Thinking and Compression blocks for state generation
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        chunk_size: int = CHUNK_SIZE,
        l_in: int = L_IN,
        l_out: int = L_OUT,
    ):
        super().__init__()

        # Load backbone
        self.backbone = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        vocab_size = self.tokenizer.vocab_size
        hidden_dim = self.backbone.config.n_embd

        self.chunk_size = chunk_size
        self.l_in = l_in
        self.l_out = l_out
        self.hidden_dim = hidden_dim

        # Thinking States modules
        self.thinking_block = ThinkingBlock(vocab_size, hidden_dim)
        self.compression_block = CompressionBlock(vocab_size, hidden_dim, chunk_size)

        # Learnable gate for injection (init to 0 for stability)
        self.injection_gate = nn.Parameter(torch.zeros(1))

        # State to inject (set during forward)
        self._state_to_inject = None
        self._extracted_hidden = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for state injection and hidden extraction."""

        def injection_hook(module, input, output):
            """Inject state at layer L_IN."""
            if self._state_to_inject is not None:
                # output is a tuple: (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Inject state
                    hidden = hidden + self.injection_gate * self._state_to_inject
                    return (hidden,) + output[1:]
                else:
                    return output + self.injection_gate * self._state_to_inject
            return output

        def extraction_hook(module, input, output):
            """Extract hidden states at layer L_OUT."""
            if isinstance(output, tuple):
                self._extracted_hidden = output[0].clone()
            else:
                self._extracted_hidden = output.clone()
            return output

        # Get the transformer blocks
        blocks = self.backbone.transformer.h

        # Register hooks
        blocks[self.l_in].register_forward_hook(injection_hook)
        blocks[self.l_out].register_forward_hook(extraction_hook)

    def build_state_tensor(
        self,
        chunk_thought_ids: list,
        seq_len: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Build the full state tensor from per-chunk thought ids.

        Args:
            chunk_thought_ids: List of (batch, thought_len) tensors, one per chunk
            seq_len: Total sequence length
            batch_size: Batch size

        Returns:
            state_tensor: (batch, seq_len, hidden_dim)
        """
        device = next(self.parameters()).device
        state_tensor = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        num_chunks = len(chunk_thought_ids)

        for chunk_idx, thought_ids in enumerate(chunk_thought_ids):
            # Compress thoughts to state
            thought_ids = thought_ids.to(device)
            chunk_state = self.compression_block(thought_ids)  # (batch, chunk_size, hidden_dim)

            # Place in state tensor at correct positions
            start_pos = chunk_idx * self.chunk_size
            end_pos = min(start_pos + self.chunk_size, seq_len)
            actual_len = end_pos - start_pos

            state_tensor[:, start_pos:end_pos, :] = chunk_state[:, :actual_len, :]

        return state_tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        chunk_thought_ids: list = None,
    ):
        """
        Forward pass with teacher-forced state injection.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) for LM loss
            chunk_thought_ids: List of (batch, thought_len) tensors for teacher forcing

        Returns:
            dict with lm_loss, thinking_loss, extracted_hidden
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Build state tensor from teacher-forced thoughts
        if chunk_thought_ids is not None:
            self._state_to_inject = self.build_state_tensor(
                chunk_thought_ids, seq_len, batch_size
            )
        else:
            self._state_to_inject = None

        # Forward through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        lm_loss = outputs.loss if labels is not None else None

        # Get extracted hidden states (set by hook)
        extracted_hidden = self._extracted_hidden

        # Compute thinking loss
        thinking_loss = None
        if chunk_thought_ids is not None and extracted_hidden is not None:
            thinking_loss = self._compute_thinking_loss(
                extracted_hidden, chunk_thought_ids
            )

        # Clear state
        self._state_to_inject = None
        self._extracted_hidden = None

        return {
            "lm_loss": lm_loss,
            "thinking_loss": thinking_loss,
            "extracted_hidden": extracted_hidden,
        }

    def _compute_thinking_loss(
        self,
        extracted_hidden: torch.Tensor,
        chunk_thought_ids: list
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for thinking block predictions.

        Args:
            extracted_hidden: (batch, seq_len, hidden_dim)
            chunk_thought_ids: List of (batch, thought_len) tensors

        Returns:
            thinking_loss: scalar tensor
        """
        batch_size, seq_len, _ = extracted_hidden.shape
        num_chunks = len(chunk_thought_ids)
        device = extracted_hidden.device

        total_loss = 0.0
        num_valid = 0

        for chunk_idx, thought_ids in enumerate(chunk_thought_ids):
            # Get hidden states for this chunk
            start_pos = chunk_idx * self.chunk_size
            end_pos = min(start_pos + self.chunk_size, seq_len)

            if start_pos >= seq_len:
                continue

            chunk_hidden = extracted_hidden[:, start_pos:end_pos, :]

            # Pad if chunk is smaller than chunk_size
            if chunk_hidden.shape[1] < self.chunk_size:
                pad_len = self.chunk_size - chunk_hidden.shape[1]
                padding = torch.zeros(batch_size, pad_len, self.hidden_dim, device=device)
                chunk_hidden = torch.cat([chunk_hidden, padding], dim=1)

            # Get logits from thinking block (direct prediction, no shifting)
            thought_ids = thought_ids.to(device)
            logits = self.thinking_block(chunk_hidden, thought_ids)

            # Compute cross-entropy loss (direct prediction)
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            chunk_loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                thought_ids.view(-1)
            )

            if not torch.isnan(chunk_loss):
                total_loss += chunk_loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        return torch.tensor(0.0, device=device)


if __name__ == "__main__":
    # Test the model
    print("Loading model...")
    model = ThinkingStatesModel()
    model.to(DEVICE)

    print(f"Model loaded on {DEVICE}")
    print(f"Injection gate value: {model.injection_gate.item()}")

    # Test forward pass
    tokenizer = model.tokenizer
    text = "Start: heads. Flip. Flip. Answer:"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(DEVICE)

    # Create dummy chunk thoughts
    thought_texts = ["tails", "heads"]
    chunk_thought_ids = [
        tokenizer(t, return_tensors="pt")["input_ids"].to(DEVICE)
        for t in thought_texts
    ]

    print("Running forward pass...")
    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        chunk_thought_ids=chunk_thought_ids
    )

    print(f"LM loss: {outputs['lm_loss']}")
    print(f"Thinking loss: {outputs['thinking_loss']}")
    print(f"Extracted hidden shape: {outputs['extracted_hidden'].shape}")
