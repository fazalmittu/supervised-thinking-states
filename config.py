"""Configuration for Thinking States prototype."""

# Model
MODEL_NAME = "gpt2"
HIDDEN_DIM = 768
NUM_LAYERS = 12

# Thinking States architecture
CHUNK_SIZE = 8
L_IN = 1        # Inject state after this layer
L_OUT = 10      # Extract hidden states from this layer
MAX_THOUGHT_LEN = 8

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 20
LAMBDA_THINK = 1.0  # Weight for thinking loss

# Data
TRAIN_SIZE = 1000
VAL_SIZE = 200
MIN_FLIPS = 4
MAX_FLIPS = 8
TEST_MIN_FLIPS = 10
TEST_MAX_FLIPS = 16

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
