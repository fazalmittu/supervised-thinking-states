"""ThinkingStatesModel wrapper that handles state injection.

Supports both GPT-2 and Qwen2.5 backbones. The backbone type is auto-detected
from the model config.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    HIDDEN_DIM, CHUNK_SIZE, GSM8K_CHUNK_SIZE, GSM8K_MAX_THOUGHT_LEN,
    L_IN, L_OUT, DEVICE, MODEL_NAME, TASK, BOS_TOKEN, EOS_TOKEN,
    GSM8K_FREEZE_BACKBONE,
)
from modules import (
    ThinkingBlock, CompressionBlock,
    AutoregressiveThinkingBlock, TransformerCompressionBlock
)
from utils import is_gpt2, is_qwen


class ThinkingStatesModel(nn.Module):
    """
    Wrapper around a causal LM backbone that adds Thinking States functionality.

    Supports GPT-2 and Qwen2.5 backbones. Architecture-specific access patterns
    (layer names, embedding locations, etc.) are abstracted behind helper methods.

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
        task: str = TASK,
    ):
        super().__init__()
        self.model_name = model_name
        self.task = task

        # ------------------------------------------------------------------
        # Load backbone and tokenizer
        # ------------------------------------------------------------------
        load_kwargs = {}
        if is_qwen(model_name):
            load_kwargs["dtype"] = torch.bfloat16
            load_kwargs["trust_remote_code"] = True

        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------------------------------------------------------------
        # Detect architecture and extract config values
        # ------------------------------------------------------------------
        backbone_config = self.backbone.config
        self.hidden_dim = getattr(backbone_config, "hidden_size",
                                  getattr(backbone_config, "n_embd", HIDDEN_DIM))
        num_layers = getattr(backbone_config, "num_hidden_layers",
                             getattr(backbone_config, "n_layer", 12))

        if self.task == "gsm8k":
            self.chunk_size = GSM8K_CHUNK_SIZE
            # Paper: L_IN = 1 (second layer, 0-indexed), L_OUT = num_layers - 2
            self.l_in = 1
            self.l_out = num_layers - 2
        else:
            self.chunk_size = chunk_size
            self.l_in = l_in
            self.l_out = l_out

        # ------------------------------------------------------------------
        # Add special tokens for GSM8K thinking
        # ------------------------------------------------------------------
        if self.task == "gsm8k":
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [BOS_TOKEN, EOS_TOKEN]}
            )
            if added > 0:
                self.backbone.resize_token_embeddings(len(self.tokenizer))

        vocab_size = len(self.tokenizer)

        # ------------------------------------------------------------------
        # Thinking States modules
        # ------------------------------------------------------------------
        if self.task == "gsm8k":
            self.thinking_block = AutoregressiveThinkingBlock(
                vocab_size,
                hidden_dim=self.hidden_dim,
                max_thought_len=GSM8K_MAX_THOUGHT_LEN,
                chunk_size=self.chunk_size,
                backbone_config=backbone_config,
                model_name=model_name,
            )
            self.compression_block = TransformerCompressionBlock(
                vocab_size,
                hidden_dim=self.hidden_dim,
                chunk_size=self.chunk_size,
                max_thought_len=GSM8K_MAX_THOUGHT_LEN,
                backbone_config=backbone_config,
                model_name=model_name,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            self._init_gsm8k_modules_from_backbone()
        else:
            self.thinking_block = ThinkingBlock(vocab_size, self.hidden_dim)
            self.compression_block = CompressionBlock(vocab_size, self.hidden_dim, self.chunk_size)

        # State to inject (set during forward)
        self._state_to_inject = None
        self._extracted_hidden = None

        # Register hooks
        self._register_hooks()

        # Optionally freeze backbone
        if self.task == "gsm8k" and GSM8K_FREEZE_BACKBONE:
            self._freeze_backbone()

        # Enable gradient checkpointing to reduce activation memory.
        # This trades ~30% more compute for ~50% less memory during backward.
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ==================================================================
    # Architecture-agnostic helpers
    # ==================================================================

    def _get_layers(self):
        """Return the nn.ModuleList of transformer layers."""
        if is_gpt2(self.model_name):
            return self.backbone.transformer.h
        else:
            # Qwen2, LLaMA, Mistral, etc.
            return self.backbone.model.layers

    def _get_embed_tokens(self):
        """Return the token embedding module."""
        if is_gpt2(self.model_name):
            return self.backbone.transformer.wte
        else:
            return self.backbone.model.embed_tokens

    def _get_lm_head(self):
        """Return the LM head linear layer."""
        return self.backbone.lm_head

    # ==================================================================
    # Initialization
    # ==================================================================

    def _init_gsm8k_modules_from_backbone(self):
        """Initialize T and C blocks from backbone layers (paper Appendix A.1)."""
        layers = self._get_layers()
        embed = self._get_embed_tokens()
        lm_head = self._get_lm_head()

        # Init T from last backbone layer
        self.thinking_block.block.load_state_dict(layers[-1].state_dict())

        # Init C from first backbone layer
        self.compression_block.encoder.load_state_dict(layers[0].state_dict())

        # Share token embedding with backbone
        self.thinking_block.token_embedding = embed
        self.compression_block.embedding = embed

        # Init T unembedding from backbone LM head (paper: "independent copy")
        with torch.no_grad():
            self.thinking_block.lm_head.weight.copy_(lm_head.weight)
            if self.thinking_block.lm_head.bias is not None and lm_head.bias is not None:
                self.thinking_block.lm_head.bias.copy_(lm_head.bias)

    def _freeze_backbone(self):
        """Freeze all backbone parameters; only T and C remain trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Make sure T and C are trainable (they have their own copies of layer weights)
        for param in self.thinking_block.parameters():
            param.requires_grad = True
        for param in self.compression_block.parameters():
            param.requires_grad = True
        # The shared embedding is frozen (backbone owns it) but that's fine:
        # T and C don't need to update embeddings, they just use them.

    # ==================================================================
    # Hooks
    # ==================================================================

    def _register_hooks(self):
        """Register forward hooks for state injection and hidden extraction."""

        def injection_hook(module, input, output):
            """Inject state at layer L_IN via direct addition (paper Eq. 1: X̃_i = X_i + S_i)."""
            if self._state_to_inject is not None:
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Cast state to match hidden dtype (important for bf16 backbone)
                    state = self._state_to_inject.to(dtype=hidden.dtype, device=hidden.device)
                    hidden = hidden + state
                    return (hidden,) + output[1:]
                else:
                    state = self._state_to_inject.to(dtype=output.dtype, device=output.device)
                    return output + state
            return output

        def extraction_hook(module, input, output):
            """Extract hidden states at layer L_OUT."""
            if isinstance(output, tuple):
                self._extracted_hidden = output[0].clone().float()  # always float32
            else:
                self._extracted_hidden = output.clone().float()
            return output

        layers = self._get_layers()
        layers[self.l_in].register_forward_hook(injection_hook)
        layers[self.l_out].register_forward_hook(extraction_hook)

    # ==================================================================
    # State construction
    # ==================================================================

    def build_state_tensor(
        self,
        chunk_thought_ids: list,
        seq_len: int,
        batch_size: int,
        chunk_thought_masks: list = None,
    ) -> torch.Tensor:
        """
        Build the full state tensor from per-chunk thought ids.

        Per the paper (Eq. 1, 4, 5): S_1 = 0 (first chunk gets zero state),
        and S_{i+1} = C(Z*_i) -- the thought from chunk i is compressed and
        injected into chunk i+1's token positions.
        """
        device = next(self.parameters()).device
        state_tensor = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        num_chunks = len(chunk_thought_ids)

        for chunk_idx, thought_ids in enumerate(chunk_thought_ids):
            target_chunk = chunk_idx + 1
            if target_chunk >= num_chunks:
                break

            thought_ids = thought_ids.to(device)
            if chunk_thought_masks is not None:
                thought_mask = chunk_thought_masks[chunk_idx].to(device)
                chunk_state = self.compression_block(thought_ids, attention_mask=thought_mask)
            else:
                chunk_state = self.compression_block(thought_ids)

            start_pos = target_chunk * self.chunk_size
            end_pos = min(start_pos + self.chunk_size, seq_len)
            if start_pos >= seq_len:
                break
            actual_len = end_pos - start_pos

            state_tensor[:, start_pos:end_pos, :] = chunk_state[:, :actual_len, :]

        return state_tensor

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        chunk_thought_ids: list = None,
        chunk_thought_masks: list = None,
    ):
        """
        Forward pass with teacher-forced state injection.

        Returns:
            dict with lm_loss, thinking_loss, extracted_hidden
        """
        batch_size, seq_len = input_ids.shape

        # Build state tensor from teacher-forced thoughts
        if chunk_thought_ids is not None:
            self._state_to_inject = self.build_state_tensor(
                chunk_thought_ids, seq_len, batch_size, chunk_thought_masks
            )
        else:
            self._state_to_inject = None

        # Forward through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        lm_loss = outputs.loss if labels is not None else None

        # Get extracted hidden states (set by hook)
        extracted_hidden = self._extracted_hidden

        # Compute thinking loss
        thinking_loss = None
        if chunk_thought_ids is not None and extracted_hidden is not None:
            thinking_loss = self._compute_thinking_loss(
                extracted_hidden, chunk_thought_ids, chunk_thought_masks
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
        chunk_thought_ids: list,
        chunk_thought_masks: list = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for thinking block predictions."""
        batch_size, seq_len, _ = extracted_hidden.shape
        device = extracted_hidden.device

        total_loss = 0.0
        num_valid = 0

        for chunk_idx, thought_ids in enumerate(chunk_thought_ids):
            start_pos = chunk_idx * self.chunk_size
            end_pos = min(start_pos + self.chunk_size, seq_len)

            if start_pos >= seq_len:
                continue

            chunk_hidden = extracted_hidden[:, start_pos:end_pos, :]

            if chunk_hidden.shape[1] < self.chunk_size:
                pad_len = self.chunk_size - chunk_hidden.shape[1]
                padding = torch.zeros(batch_size, pad_len, self.hidden_dim, device=device)
                chunk_hidden = torch.cat([chunk_hidden, padding], dim=1)

            thought_ids = thought_ids.to(device)
            if self.task == "gsm8k":
                thought_mask = None
                if chunk_thought_masks is not None:
                    thought_mask = chunk_thought_masks[chunk_idx].to(device)

                logits = self.thinking_block(chunk_hidden, thought_ids, thought_mask)

                labels = thought_ids[:, 1:].contiguous()
                if thought_mask is not None:
                    mask = thought_mask[:, 1:].contiguous()
                    labels = labels.masked_fill(mask == 0, self.tokenizer.pad_token_id)

                loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                chunk_loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            else:
                logits = self.thinking_block(chunk_hidden, thought_ids)
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
    print("Loading model...")
    model = ThinkingStatesModel()
    model.to(DEVICE)

    print(f"Model loaded on {DEVICE}")
    print(f"Backbone: {model.model_name}")
    print(f"Hidden dim: {model.hidden_dim}")
    print(f"L_IN: {model.l_in}, L_OUT: {model.l_out}")
    print(f"Chunk size: {model.chunk_size}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Quick forward test
    tokenizer = model.tokenizer
    text = "What is 2 + 2?\nAnswer:"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(DEVICE)
    print(f"\nTest input: {text}")
    print(f"Token count: {input_ids.shape[1]}")

    outputs = model(input_ids=input_ids, labels=input_ids)
    print(f"LM loss: {outputs['lm_loss']}")
    print(f"Extracted hidden shape: {outputs['extracted_hidden'].shape}")
