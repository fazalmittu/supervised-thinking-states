"""Thinking Block and Compression Block modules.

Supports both GPT-2 and Qwen2.5 backbones. The correct decoder layer type and
causal mask function are selected based on the backbone model name.
"""

import torch
import torch.nn as nn

from config import HIDDEN_DIM, CHUNK_SIZE, MAX_THOUGHT_LEN, GSM8K_CHUNK_SIZE, GSM8K_MAX_THOUGHT_LEN, DEVICE
from utils import is_gpt2


# ---------------------------------------------------------------------------
# Helpers for backbone-agnostic layer creation
# ---------------------------------------------------------------------------


def _make_decoder_layer(backbone_config, model_name: str, layer_idx: int = 0):
    """Create one decoder layer matching the backbone architecture."""
    if is_gpt2(model_name):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        return GPT2Block(backbone_config)
    else:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        return Qwen2DecoderLayer(backbone_config, layer_idx=layer_idx)


# Cache for Qwen2RotaryEmbedding instances to avoid re-creating per call.
_rotary_cache: dict = {}


def _get_rotary_embedding(backbone_config, device):
    """Return a cached Qwen2RotaryEmbedding for the given config and device."""
    key = (id(backbone_config), str(device))
    if key not in _rotary_cache:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        _rotary_cache[key] = Qwen2RotaryEmbedding(config=backbone_config, device=device)
    return _rotary_cache[key]


def _run_decoder_layer(layer, hidden_states, attention_mask, model_name: str,
                       backbone_config=None):
    """Run a single decoder layer with architecture-appropriate arguments.

    For GPT-2 layers, we build a 4-D causal mask via create_causal_mask.
    For Qwen2 layers, we pass position_ids and position_embeddings (RoPE).
    """
    device = hidden_states.device
    batch_size, seq_len, _ = hidden_states.shape

    if is_gpt2(model_name):
        from transformers.models.gpt2.modeling_gpt2 import create_causal_mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        cache_position = torch.arange(seq_len, device=device)
        causal_mask = create_causal_mask(
            backbone_config, hidden_states, attention_mask,
            cache_position, past_key_values=None,
        )
        out = layer(hidden_states, attention_mask=causal_mask)
        return out[0] if isinstance(out, tuple) else out
    else:
        # Qwen2DecoderLayer expects: hidden_states, attention_mask, position_ids,
        #   position_embeddings, ...
        from transformers.masking_utils import create_causal_mask

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Build RoPE embeddings (cached to avoid re-instantiation every call)
        rotary = _get_rotary_embedding(backbone_config, device)
        position_embeddings = rotary(hidden_states, position_ids)

        # Build causal mask  (Qwen2 uses the generic create_causal_mask)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=hidden_states.dtype)
        cache_position = torch.arange(seq_len, device=device)
        causal_mask = create_causal_mask(
            config=backbone_config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
        )

        out = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Parity-task modules (unchanged, GPT-2 only)
# ---------------------------------------------------------------------------

class CompressionBlock(nn.Module):
    """Compresses thought tokens into a state tensor (parity task)."""

    def __init__(self, vocab_size: int, hidden_dim: int = HIDDEN_DIM,
                 chunk_size: int = CHUNK_SIZE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim * chunk_size)

    def forward(self, thought_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(thought_ids)
        pooled = embedded.mean(dim=1)
        state = self.proj(pooled)
        return state.view(-1, self.chunk_size, self.hidden_dim)


class ThinkingBlock(nn.Module):
    """Generates thought tokens from hidden states (parity task)."""

    def __init__(self, vocab_size: int, hidden_dim: int = HIDDEN_DIM,
                 max_thought_len: int = MAX_THOUGHT_LEN, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_thought_len = max_thought_len
        self.vocab_size = vocab_size

        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * max_thought_len),
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states, target_ids=None):
        batch_size = hidden_states.shape[0]
        pooled = hidden_states.mean(dim=1)
        projected = self.hidden_proj(pooled).view(batch_size, self.max_thought_len, self.hidden_dim)
        logits = self.lm_head(projected)
        if target_ids is not None:
            logits = logits[:, :target_ids.shape[1], :]
        return logits

    @torch.no_grad()
    def generate(self, hidden_states, tokenizer, max_len=None):
        if max_len is None:
            max_len = self.max_thought_len
        logits = self.forward(hidden_states, None)
        return logits[:, :max_len, :].argmax(dim=-1)


# ---------------------------------------------------------------------------
# GSM8K modules (supports GPT-2 and Qwen2.5)
# ---------------------------------------------------------------------------

class AutoregressiveThinkingBlock(nn.Module):
    """
    Autoregressive Transformer decoder for generating thought tokens (T block).

    Uses a single decoder layer matching the backbone architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        max_thought_len: int = GSM8K_MAX_THOUGHT_LEN,
        chunk_size: int = GSM8K_CHUNK_SIZE,
        backbone_config=None,
        model_name: str = "gpt2",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_thought_len = max_thought_len
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.model_name = model_name

        # Token embedding -- shared with backbone (set in model.py)
        self.token_embedding = None

        if backbone_config is None:
            raise ValueError("backbone_config is required")
        self.backbone_config = backbone_config

        # Single decoder layer (initialized from backbone's last layer in model.py)
        self.block = _make_decoder_layer(backbone_config, model_name, layer_idx=0)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, chunk_hidden, target_ids, target_mask=None):
        """Training forward pass with teacher forcing."""
        decoder_input_ids = target_ids[:, :-1]
        batch_size, dec_len = decoder_input_ids.shape

        decoder_input = self.token_embedding(decoder_input_ids)

        # Full context: prefix hidden states + thought token embeddings
        prefix = chunk_hidden
        full_input = torch.cat([prefix, decoder_input], dim=1)

        # Build attention mask
        if target_mask is not None:
            dec_mask = target_mask[:, :-1].to(full_input.device)
        else:
            dec_mask = torch.ones(batch_size, dec_len, device=full_input.device)
        prefix_mask = torch.ones(batch_size, self.chunk_size, device=full_input.device)
        attn_mask = torch.cat([prefix_mask, dec_mask], dim=1)

        # Run through decoder layer
        decoded = _run_decoder_layer(
            self.block, full_input, attn_mask, self.model_name, self.backbone_config
        )

        logits = self.lm_head(decoded[:, -dec_len:, :])
        return logits

    @torch.no_grad()
    def generate(self, chunk_hidden, bos_id, eos_id, max_len=None):
        """Autoregressive generation of thought tokens."""
        if max_len is None:
            max_len = self.max_thought_len

        batch_size = chunk_hidden.shape[0]
        device = chunk_hidden.device

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            decoder_input = self.token_embedding(generated)
            full_input = torch.cat([chunk_hidden, decoder_input], dim=1)
            attn_mask = torch.ones(batch_size, full_input.shape[1], device=device)

            decoded = _run_decoder_layer(
                self.block, full_input, attn_mask, self.model_name, self.backbone_config
            )

            next_logits = self.lm_head(decoded[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            finished = finished | (next_token.squeeze(-1) == eos_id)
            next_token = next_token.masked_fill(
                finished.unsqueeze(-1) & (next_token != eos_id), eos_id
            )

            generated = torch.cat([generated, next_token], dim=1)
            if finished.all():
                break

        return generated


class TransformerCompressionBlock(nn.Module):
    """
    Transformer-based compression of thought tokens into state tensor (C block).

    Uses a single decoder layer matching the backbone architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        chunk_size: int = GSM8K_CHUNK_SIZE,
        max_thought_len: int = GSM8K_MAX_THOUGHT_LEN,
        backbone_config=None,
        model_name: str = "gpt2",
        pad_token_id: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.model_name = model_name

        # Token embedding -- shared with backbone (set in model.py)
        self.embedding = None

        if backbone_config is None:
            raise ValueError("backbone_config is required")
        self.backbone_config = backbone_config

        # Single decoder layer (initialized from backbone's first layer in model.py)
        self.encoder = _make_decoder_layer(backbone_config, model_name, layer_idx=0)

    def forward(self, thought_ids, attention_mask=None):
        batch_size, seq_len = thought_ids.shape
        device = thought_ids.device

        # Pad input if shorter than chunk_size (paper Appendix A.1)
        if seq_len < self.chunk_size:
            pad_len = self.chunk_size - seq_len
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            pad_tokens = torch.full((batch_size, pad_len), pad_id, dtype=torch.long, device=device)
            thought_ids = torch.cat([thought_ids, pad_tokens], dim=1)
            if attention_mask is not None:
                pad_mask = torch.zeros(batch_size, pad_len, device=device)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
            else:
                attention_mask = torch.cat([
                    torch.ones(batch_size, seq_len, device=device),
                    torch.zeros(batch_size, pad_len, device=device),
                ], dim=1)
            seq_len = thought_ids.shape[1]

        embedded = self.embedding(thought_ids)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)

        encoded = _run_decoder_layer(
            self.encoder, embedded, attention_mask, self.model_name, self.backbone_config
        )

        state = encoded[:, -self.chunk_size:, :]
        return state


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    print("Testing CompressionBlock (parity):")
    comp_block = CompressionBlock(vocab_size)
    thought_ids = torch.randint(0, vocab_size, (2, 3))
    state = comp_block(thought_ids)
    print(f"  Input: {thought_ids.shape} -> Output: {state.shape}")

    print("\nTesting ThinkingBlock (parity):")
    think_block = ThinkingBlock(vocab_size)
    hidden = torch.randn(2, CHUNK_SIZE, HIDDEN_DIM)
    target = torch.randint(0, vocab_size, (2, 4))
    logits = think_block(hidden, target)
    print(f"  Hidden: {hidden.shape}, Target: {target.shape} -> Logits: {logits.shape}")

    print("\nTesting AutoregressiveThinkingBlock (GPT-2):")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    ar_think = AutoregressiveThinkingBlock(
        vocab_size, backbone_config=gpt2.config, model_name="gpt2"
    )
    ar_think.token_embedding = gpt2.transformer.wte
    hidden_gsm = torch.randn(2, GSM8K_CHUNK_SIZE, HIDDEN_DIM)
    target_gsm = torch.randint(0, vocab_size, (2, 10))
    logits_gsm = ar_think(hidden_gsm, target_gsm)
    print(f"  Hidden: {hidden_gsm.shape}, Target: {target_gsm.shape} -> Logits: {logits_gsm.shape}")

    print("\nTesting TransformerCompressionBlock (GPT-2):")
    tf_comp = TransformerCompressionBlock(
        vocab_size, backbone_config=gpt2.config, model_name="gpt2",
        pad_token_id=tokenizer.eos_token_id,
    )
    tf_comp.embedding = gpt2.transformer.wte
    thought_ids_gsm = torch.randint(0, vocab_size, (2, 20))
    mask = torch.ones(2, 20)
    mask[1, 15:] = 0
    state_gsm = tf_comp(thought_ids_gsm, attention_mask=mask)
    print(f"  Input: {thought_ids_gsm.shape} -> Output: {state_gsm.shape}")

    print("\nAll tests passed!")
