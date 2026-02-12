"""Thinking Block and Compression Block modules."""

import math

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, create_causal_mask

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
        gpt2_config=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_thought_len = max_thought_len
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

        # Token embedding is shared with backbone (set in model.py _init_gsm8k_modules_from_backbone).
        # We declare it here so the attribute exists; it gets overwritten with the backbone's wte.
        self.token_embedding = None

        # No position embeddings: per the paper, T is a single Transformer block
        # matching a backbone layer. The prefix H_out already encodes positional
        # information from the backbone's forward pass.

        if gpt2_config is None:
            raise ValueError("gpt2_config is required for AutoregressiveThinkingBlock")
        self.gpt2_config = gpt2_config
        self.block = GPT2Block(gpt2_config)

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

        # Embed decoder input (no position embeddings per paper)
        decoder_input = self.token_embedding(decoder_input_ids)  # (batch, dec_len, hidden_dim)

        # full context: prefix hidden states + thought tokens
        prefix = chunk_hidden  # (batch, chunk_size, hidden_dim)
        full_input = torch.cat([prefix, decoder_input], dim=1)  # (batch, chunk_size + dec_len, hidden_dim)
        # mask for self attn
        if target_mask is not None:
            dec_mask = target_mask[:, :-1].to(full_input.device)
        else:
            dec_mask = torch.ones(batch_size, dec_len, device=full_input.device)
        prefix_mask = torch.ones(batch_size, self.chunk_size, device=full_input.device)
        attn_mask = torch.cat([prefix_mask, dec_mask], dim=1)
        cache_position = torch.arange(full_input.shape[1], device=full_input.device)
        causal_mask = create_causal_mask(
            self.gpt2_config,
            full_input,
            attn_mask,
            cache_position,
            past_key_values=None,
        )

        # decode
        decoded = self.block(
            full_input,
            attention_mask=causal_mask,
        )[0]  # (batch, total_len, hidden_dim)

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

            # Embed (no position embeddings per paper)
            decoder_input = self.token_embedding(generated)

            # full context: prefix + generated tokens
            full_input = torch.cat([chunk_hidden, decoder_input], dim=1)
            attn_mask = torch.ones(batch_size, full_input.shape[1], device=device)
            cache_position = torch.arange(full_input.shape[1], device=device)
            causal_mask = create_causal_mask(
                self.gpt2_config,
                full_input,
                attn_mask,
                cache_position,
                past_key_values=None,
            )

            # decode
            decoded = self.block(full_input, attention_mask=causal_mask)[0]

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
        gpt2_config=None,
        pad_token_id: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id

        # Token embedding shared with backbone (set in model.py _init_gsm8k_modules_from_backbone).
        self.embedding = None

        # No position embeddings: per the paper, C is a single Transformer block
        # matching the first backbone layer. It contextualizes token embeddings
        # directly without adding its own positional information.

        if gpt2_config is None:
            raise ValueError("gpt2_config is required for TransformerCompressionBlock")
        self.gpt2_config = gpt2_config
        self.encoder = GPT2Block(gpt2_config)

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
        device = thought_ids.device

        # Per paper (Appendix A.1): if input is shorter than chunk_size, pad the
        # input token sequence with padding tokens *before* encoding, so the
        # Transformer block processes them and produces meaningful representations.
        if seq_len < self.chunk_size:
            pad_len = self.chunk_size - seq_len
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            pad_tokens = torch.full((batch_size, pad_len), pad_id, dtype=torch.long, device=device)
            thought_ids = torch.cat([thought_ids, pad_tokens], dim=1)
            # Extend attention mask: 0 for padding positions
            if attention_mask is not None:
                pad_mask = torch.zeros(batch_size, pad_len, device=device)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
            else:
                attention_mask = torch.cat([
                    torch.ones(batch_size, seq_len, device=device),
                    torch.zeros(batch_size, pad_len, device=device),
                ], dim=1)
            seq_len = thought_ids.shape[1]

        # Embed tokens (no position embeddings per paper)
        embedded = self.embedding(thought_ids)  # (batch, seq_len, hidden_dim)

        # Build attention mask
        if attention_mask is not None:
            attn_mask = attention_mask.to(embedded.device)
        else:
            attn_mask = torch.ones(batch_size, seq_len, device=embedded.device)

        # Encode with causal attention
        cache_position = torch.arange(seq_len, device=embedded.device)
        causal_mask = create_causal_mask(
            self.gpt2_config,
            embedded,
            attn_mask,
            cache_position,
            past_key_values=None,
        )
        encoded = self.encoder(embedded, attention_mask=causal_mask)[0]  # (batch, seq_len, hidden_dim)

        # Take last chunk_size states as the compressed state
        state = encoded[:, -self.chunk_size:, :]

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
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    ar_think = AutoregressiveThinkingBlock(vocab_size, gpt2_config=gpt2.config)
    # Must set token_embedding before use (normally done by model.py)
    ar_think.token_embedding = gpt2.transformer.wte
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
    tf_comp = TransformerCompressionBlock(
        vocab_size, gpt2_config=gpt2.config, pad_token_id=tokenizer.eos_token_id
    )
    # Must set embedding before use (normally done by model.py)
    tf_comp.embedding = gpt2.transformer.wte
    thought_ids_gsm = torch.randint(0, vocab_size, (2, 20))
    mask = torch.ones(2, 20)
    mask[1, 15:] = 0  # Simulate padding
    state_gsm = tf_comp(thought_ids_gsm, attention_mask=mask)
    print(f"Input shape: {thought_ids_gsm.shape}")
    print(f"Output shape: {state_gsm.shape}")  # Should be (2, GSM8K_CHUNK_SIZE, HIDDEN_DIM)

    print("\nTesting TransformerCompressionBlock short input padding:")
    short_ids = torch.randint(0, vocab_size, (2, 4))  # shorter than chunk_size
    short_mask = torch.ones(2, 4)
    state_short = tf_comp(short_ids, attention_mask=short_mask)
    print(f"Short input shape: {short_ids.shape}")
    print(f"Output shape: {state_short.shape}")  # Should be (2, GSM8K_CHUNK_SIZE, HIDDEN_DIM)
