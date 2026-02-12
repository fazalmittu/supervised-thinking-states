"""Configuration for Thinking States prototype."""

# Task selector: "parity" or "gsm8k"
TASK = "gsm8k"

# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------
# Set to "gpt2" for the lightweight GPT-2 baseline, or a Qwen model for the
# backbone used in the paper. Paper uses Qwen2.5-Base 0.5B and 1.5B.
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# These are auto-detected from the loaded model config at runtime; the values
# below are only used as fallbacks / for the parity task with GPT-2.
HIDDEN_DIM = 768
NUM_LAYERS = 12

# Thinking States architecture (parity)
CHUNK_SIZE = 8
L_IN = 1        # Inject state after this layer
L_OUT = 10      # Extract hidden states from this layer
MAX_THOUGHT_LEN = 8

# Training (parity)
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 20

# Data (parity)
TRAIN_SIZE = 1000
VAL_SIZE = 200
MIN_FLIPS = 4
MAX_FLIPS = 8
TEST_MIN_FLIPS = 10
TEST_MAX_FLIPS = 16

# GSM8K architecture
GSM8K_CHUNK_SIZE = 16
GSM8K_MAX_THOUGHT_LEN = 64
GSM8K_MAX_ANSWER_TOKENS = 32

# GSM8K training
GSM8K_BATCH_SIZE = 2           # reduced for MPS memory (full fine-tune 1.5B)
GSM8K_LEARNING_RATE = 5e-5     # paper uses standard fine-tuning LR
GSM8K_NUM_EPOCHS = 10
GSM8K_FREEZE_BACKBONE = False  # paper fine-tunes entire model end-to-end

# GSM8K special tokens
BOS_TOKEN = "<|thought_start|>"
EOS_TOKEN = "<|thought_end|>"

# Teacher LLM config
TEACHER_MODEL = "gemini-2.5-flash"

# GSM8K data
GSM8K_CACHE_DIR = "data/gsm8k"
GSM8K_ALIGNED_SUFFIX = ""

# Device
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
