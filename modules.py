"""Thinking Block and Compression Block modules."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import HIDDEN_DIM, CHUNK_SIZE, MAX_THOUGHT_LEN, DEVICE


class CompressionBlock(nn.Module):
    """
    Compresses thought tokens into a state tensor.

    Input: thought token ids (batch, thought_len)
    Output: state tensor (batch, chunk_size, hidden_dim)

    Simplest approach: embed -> mean pool -> linear -> expand
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        chunk_size: int = CHUNK_SIZE
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size

        # Embedding layer (can share with backbone later)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Project pooled embedding to state
        self.proj = nn.Linear(hidden_dim, hidden_dim * chunk_size)

    def forward(self, thought_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            thought_ids: (batch, thought_len) token ids

        Returns:
            state: (batch, chunk_size, hidden_dim)
        """
        # Embed tokens
        embedded = self.embedding(thought_ids)  # (batch, thought_len, hidden_dim)

        # Mean pool over sequence
        pooled = embedded.mean(dim=1)  # (batch, hidden_dim)

        # Project to state size
        state = self.proj(pooled)  # (batch, hidden_dim * chunk_size)

        # Reshape to (batch, chunk_size, hidden_dim)
        state = state.view(-1, self.chunk_size, self.hidden_dim)

        return state


class ThinkingBlock(nn.Module):
    """
    Generates thought tokens from hidden states.

    Input: hidden states (batch, chunk_size, hidden_dim)
    Output: thought token logits (batch, max_thought_len, vocab_size)

    Approach: mean pool hidden -> MLP -> predict fixed number of output positions
    This is simpler than autoregressive - we directly predict thought tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        max_thought_len: int = MAX_THOUGHT_LEN,
        num_heads: int = 8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_thought_len = max_thought_len
        self.vocab_size = vocab_size

        # Project pooled hidden states
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * max_thought_len)
        )

        # LM head for each position
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, chunk_size, hidden_dim)
            target_ids: (batch, thought_len) - only used for determining output length

        Returns:
            logits: (batch, thought_len, vocab_size)
        """
        batch_size = hidden_states.shape[0]

        # Mean pool hidden states
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        # Project to multiple positions
        projected = self.hidden_proj(pooled)  # (batch, hidden_dim * max_thought_len)
        projected = projected.view(batch_size, self.max_thought_len, self.hidden_dim)

        # LM head
        logits = self.lm_head(projected)  # (batch, max_thought_len, vocab_size)

        # If target_ids provided, truncate to match target length
        if target_ids is not None:
            target_len = target_ids.shape[1]
            logits = logits[:, :target_len, :]

        return logits

    @torch.no_grad()
    def generate(
        self,
        hidden_states: torch.Tensor,
        tokenizer: GPT2Tokenizer,
        max_len: int = None
    ) -> torch.Tensor:
        """
        Generate thought tokens (non-autoregressive).

        Args:
            hidden_states: (batch, chunk_size, hidden_dim)
            tokenizer: for getting special tokens
            max_len: max generation length

        Returns:
            generated_ids: (batch, gen_len)
        """
        if max_len is None:
            max_len = self.max_thought_len

        # Get logits for all positions
        logits = self.forward(hidden_states, None)  # (batch, max_thought_len, vocab_size)

        # Take argmax
        generated = logits[:, :max_len, :].argmax(dim=-1)  # (batch, max_len)

        return generated


if __name__ == "__main__":
    # Test the modules
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    print("Testing CompressionBlock:")
    comp_block = CompressionBlock(vocab_size)
    thought_ids = torch.randint(0, vocab_size, (2, 3))  # batch=2, len=3
    state = comp_block(thought_ids)
    print(f"Input shape: {thought_ids.shape}")
    print(f"Output shape: {state.shape}")  # Should be (2, chunk_size, hidden_dim)

    print("\nTesting ThinkingBlock:")
    think_block = ThinkingBlock(vocab_size)
    hidden = torch.randn(2, CHUNK_SIZE, HIDDEN_DIM)
    target = torch.randint(0, vocab_size, (2, 4))
    logits = think_block(hidden, target)
    print(f"Hidden shape: {hidden.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Logits shape: {logits.shape}")  # Should be (2, 4, vocab_size)
