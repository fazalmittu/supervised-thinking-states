"""Thinking Block and Compression Block modules."""

import math

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import HIDDEN_DIM, CHUNK_SIZE, MAX_THOUGHT_LEN, GSM8K_CHUNK_SIZE, GSM8K_MAX_THOUGHT_LEN, DEVICE


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


class AutoregressiveThinkingBlock(nn.Module):
    """
    Autoregressive Transformer decoder for generating thought tokens.

    Uses a single causal Transformer block. The chunk hidden states are
    treated as a prefix context, and thought tokens are generated
    autoregressively. Designed for GSM8K where thoughts are full natural
    language reasoning steps.

    Input: chunk hidden states (batch, chunk_size, hidden_dim)
    Output (training): logits (batch, thought_len-1, vocab_size) predicting positions 1..T
    Output (inference): generated token ids
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        max_thought_len: int = GSM8K_MAX_THOUGHT_LEN,
        num_heads: int = 8,
        chunk_size: int = GSM8K_CHUNK_SIZE,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_thought_len = max_thought_len
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_thought_len, hidden_dim)

        # same deal as compression block, encoder layer for causal self-attn without cross attention
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        # mps needs us to disable nested tensor path
        self.block = nn.TransformerEncoder(block, num_layers=1, enable_nested_tensor=False)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # mask for self attn
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.float32),
            diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        chunk_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.

        Args:
            chunk_hidden: (batch, chunk_size, hidden_dim) - hidden states from backbone
            target_ids: (batch, thought_len) - full thought token ids including BOS
            target_mask: (batch, thought_len) - attention mask (1=attend, 0=pad)

        Returns:
            logits: (batch, thought_len-1, vocab_size) predicting positions 1..T
        """
        # Decoder input: shift right (drop last token)
        decoder_input_ids = target_ids[:, :-1]  # (batch, thought_len-1)
        batch_size, dec_len = decoder_input_ids.shape

        # embed decoder input
        positions = torch.arange(dec_len, device=decoder_input_ids.device) + self.chunk_size
        tok_emb = self.token_embedding(decoder_input_ids)
        pos_emb = self.position_embedding(positions)
        decoder_input = tok_emb + pos_emb  # (batch, dec_len, hidden_dim)

        # full context: prefix hidden states + thought tokens
        prefix = chunk_hidden  # (batch, chunk_size, hidden_dim)
        full_input = torch.cat([prefix, decoder_input], dim=1)  # (batch, chunk_size + dec_len, hidden_dim)
        total_len = full_input.shape[1]

        # mask for self attn
        causal_mask = self._make_causal_mask(total_len, full_input.device)

        # decode
        decoded = self.block(
            full_input,
            mask=causal_mask,
        )  # (batch, total_len, hidden_dim)

        logits = self.lm_head(decoded[:, -dec_len:, :])  # (batch, dec_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        chunk_hidden: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation of thought tokens.

        Args:
            chunk_hidden: (batch, chunk_size, hidden_dim)
            bos_id: BOS token id
            eos_id: EOS token id
            max_len: Maximum generation length

        Returns:
            generated_ids: (batch, gen_len) - includes BOS, excludes EOS
        """
        if max_len is None:
            max_len = self.max_thought_len

        batch_size = chunk_hidden.shape[0]
        device = chunk_hidden.device

        # start with BOS
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            seq_len = generated.shape[1]

            # embed
            positions = torch.arange(seq_len, device=device) + self.chunk_size
            tok_emb = self.token_embedding(generated)
            pos_emb = self.position_embedding(positions)
            decoder_input = tok_emb + pos_emb

            # full context: prefix + generated tokens
            full_input = torch.cat([chunk_hidden, decoder_input], dim=1)
            total_len = full_input.shape[1]
            causal_mask = self._make_causal_mask(total_len, device)

            # decode
            decoded = self.block(full_input, mask=causal_mask)

            # next token from last position
            next_logits = self.lm_head(decoded[:, -1, :])  # (batch, vocab_size)
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

            # check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_id)

            # pad finished sequences with EOS
            next_token = next_token.masked_fill(
                finished.unsqueeze(-1) & (next_token != eos_id),
                eos_id
            )

            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated


class TransformerCompressionBlock(nn.Module):
    """
    Transformer-based compression of thought tokens into state tensor.

    Uses a causal Transformer block to process thought tokens, then takes the
    last chunk_size hidden states as the compressed state.

    Input: thought token ids (batch, thought_len)
    Output: state tensor (batch, chunk_size, hidden_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        chunk_size: int = GSM8K_CHUNK_SIZE,
        max_thought_len: int = GSM8K_MAX_THOUGHT_LEN,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_thought_len, hidden_dim)

        # this was tricky, we use the encoder layer bc we want causal self-attention
        # in the paper this is described as a decoder block but we don't want cross attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        # mps needs us to disable nested tensor path
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False
        )

    def forward(
        self,
        thought_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            thought_ids: (batch, thought_len) token ids
            attention_mask: (batch, thought_len) attention mask (1=attend, 0=pad)

        Returns:
            state: (batch, chunk_size, hidden_dim)
        """
        batch_size, seq_len = thought_ids.shape

        # embeddingssss
        positions = torch.arange(seq_len, device=thought_ids.device)
        tok_emb = self.embedding(thought_ids)
        pos_emb = self.position_embedding(positions)
        embedded = tok_emb + pos_emb  # (batch, thought_len, hidden_dim)

        # mask for self attn
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=thought_ids.device, dtype=torch.float32),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))

        # mask for padding
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True where padded

        # encode
        encoded = self.encoder(
            embedded,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # (batch, thought_len, hidden_dim)

        # take last chunk_size states (pad on left if needed)
        if seq_len >= self.chunk_size:
            state = encoded[:, -self.chunk_size:, :]
        else:
            pad_len = self.chunk_size - seq_len
            padding = torch.zeros(batch_size, pad_len, self.hidden_dim, device=encoded.device)
            state = torch.cat([padding, encoded], dim=1)

        return state


if __name__ == "__main__":
    # Test the modules
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    print("Testing CompressionBlock (parity):")
    comp_block = CompressionBlock(vocab_size)
    thought_ids = torch.randint(0, vocab_size, (2, 3))  # batch=2, len=3
    state = comp_block(thought_ids)
    print(f"Input shape: {thought_ids.shape}")
    print(f"Output shape: {state.shape}")  # Should be (2, chunk_size, hidden_dim)

    print("\nTesting ThinkingBlock (parity):")
    think_block = ThinkingBlock(vocab_size)
    hidden = torch.randn(2, CHUNK_SIZE, HIDDEN_DIM)
    target = torch.randint(0, vocab_size, (2, 4))
    logits = think_block(hidden, target)
    print(f"Hidden shape: {hidden.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Logits shape: {logits.shape}")  # Should be (2, 4, vocab_size)

    print("\n" + "="*50)
    print("Testing AutoregressiveThinkingBlock (GSM8K):")
    ar_think = AutoregressiveThinkingBlock(vocab_size)
    hidden_gsm = torch.randn(2, GSM8K_CHUNK_SIZE, HIDDEN_DIM)
    target_gsm = torch.randint(0, vocab_size, (2, 10))
    logits_gsm = ar_think(hidden_gsm, target_gsm)
    print(f"Hidden shape: {hidden_gsm.shape}")
    print(f"Target shape: {target_gsm.shape}")
    print(f"Logits shape: {logits_gsm.shape}")  # Should be (2, 9, vocab_size)

    print("\nTesting generation:")
    bos_id = 0
    eos_id = 1
    gen_ids = ar_think.generate(hidden_gsm, bos_id, eos_id, max_len=15)
    print(f"Generated shape: {gen_ids.shape}")  # Should be (2, <=15)

    print("\nTesting TransformerCompressionBlock (GSM8K):")
    tf_comp = TransformerCompressionBlock(vocab_size)
    thought_ids_gsm = torch.randint(0, vocab_size, (2, 20))
    mask = torch.ones(2, 20)
    mask[1, 15:] = 0  # Simulate padding
    state_gsm = tf_comp(thought_ids_gsm, attention_mask=mask)
    print(f"Input shape: {thought_ids_gsm.shape}")
    print(f"Output shape: {state_gsm.shape}")  # Should be (2, GSM8K_CHUNK_SIZE, HIDDEN_DIM)
