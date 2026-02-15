"""Unified training script for Thinking States.

Trains either a Thinking States model or a vanilla baseline, on either the
parity task or GSM8K, depending on config.py and CLI flags.

Usage:
    python train.py                 # train Thinking States on current TASK
    python train.py --vanilla       # train vanilla baseline on current TASK
    python train.py --epochs 5      # override epoch count
"""

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import (
    DEVICE, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    TRAIN_SIZE, VAL_SIZE, MIN_FLIPS, MAX_FLIPS,
    GSM8K_BATCH_SIZE, GSM8K_NUM_EPOCHS, GSM8K_LEARNING_RATE,
    GSM8K_FREEZE_BACKBONE, TASK
)
from src.data import (
    ParityDataset, get_collate_fn,
    GSM8KDataset, get_gsm8k_collate_fn
)
from src.model import ThinkingStatesModel


CHECKPOINT_DIR = "checkpoints"
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO = 0.03


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def _checkpoint_path(task: str, vanilla: bool) -> str:
    prefix = "vanilla" if vanilla else "thinking_states"
    return os.path.join(CHECKPOINT_DIR, f"{prefix}_{task}.pt")


# =========================================================================
# Thinking States training
# =========================================================================

def train_epoch_ts(model, dataloader, optimizer, device, scheduler=None):
    """Train Thinking States for one epoch (L_LM + L_T, paper Eq. 6)."""
    model.train()
    total_lm_loss = 0.0
    total_think_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        chunk_thought_ids = batch["chunk_thought_ids"]
        chunk_thought_masks = batch.get("chunk_thought_masks")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            chunk_thought_ids=chunk_thought_ids,
            chunk_thought_masks=chunk_thought_masks
        )

        lm_loss = outputs["lm_loss"]
        think_loss = outputs["thinking_loss"]

        loss = lm_loss
        if think_loss is not None:
            loss = loss + think_loss

        (loss / GRAD_ACCUM_STEPS).backward()

        total_lm_loss += lm_loss.item()
        if think_loss is not None:
            total_think_loss += think_loss.item()
        num_batches += 1

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if device == "mps" and (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.mps.empty_cache()

    return {
        "lm_loss": total_lm_loss / num_batches,
        "think_loss": total_think_loss / num_batches,
    }


def evaluate_ts(model, dataloader, device):
    """Evaluate Thinking States on validation set."""
    model.eval()
    total_lm_loss = 0.0
    total_think_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            chunk_thought_ids = batch["chunk_thought_ids"]
            chunk_thought_masks = batch.get("chunk_thought_masks")

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chunk_thought_ids=chunk_thought_ids,
                chunk_thought_masks=chunk_thought_masks
            )

            lm_loss = outputs["lm_loss"]
            think_loss = outputs["thinking_loss"]
            logits = outputs["logits"].float()

            total_lm_loss += lm_loss.item()
            if think_loss is not None:
                total_think_loss += think_loss.item()
            num_batches += 1

            for i in range(input_ids.shape[0]):
                answer_mask = labels[i] != -100
                if answer_mask.any():
                    answer_positions = answer_mask.nonzero(as_tuple=False).view(-1)
                    preds = logits[i, answer_positions - 1].argmax(dim=-1)
                    trues = labels[i, answer_positions]
                    if preds.shape == trues.shape and torch.equal(preds, trues):
                        correct += 1
                    total += 1

            if device == "mps" and (step + 1) % 10 == 0:
                torch.mps.empty_cache()

    return {
        "lm_loss": total_lm_loss / num_batches,
        "think_loss": total_think_loss / num_batches,
        "accuracy": correct / total if total > 0 else 0.0,
    }


# =========================================================================
# Vanilla training
# =========================================================================

def train_epoch_vanilla(model, dataloader, optimizer, device, scheduler=None):
    """Train vanilla model for one epoch (standard LM loss)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        (loss / GRAD_ACCUM_STEPS).backward()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return {"lm_loss": total_loss / num_batches}


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Thinking States or vanilla baseline")
    parser.add_argument("--vanilla", action="store_true",
                        help="Train vanilla baseline (no T/C blocks)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count from config")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Task: {TASK}")
    print(f"Mode: {'vanilla' if args.vanilla else 'thinking_states'}")

    # ---- Select hyperparameters based on task ----
    if TASK == "gsm8k":
        batch_size = GSM8K_BATCH_SIZE
        num_epochs = args.epochs or GSM8K_NUM_EPOCHS
        learning_rate = GSM8K_LEARNING_RATE
    else:
        batch_size = BATCH_SIZE
        num_epochs = args.epochs or NUM_EPOCHS
        learning_rate = LEARNING_RATE

    # ---- Build model ----
    if args.vanilla:
        print(f"Loading vanilla {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # All tasks now use BOS/EOS tokens for thinking supervision
        from config import BOS_TOKEN, EOS_TOKEN
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [BOS_TOKEN, EOS_TOKEN]}
        )
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
        model.to(DEVICE)
    else:
        print("Loading Thinking States model...")
        model = ThinkingStatesModel()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params:     {total_params:>14,}")
        print(f"Trainable params: {trainable:>14,} ({100*trainable/total_params:.1f}%)")

        if GSM8K_FREEZE_BACKBONE:
            print("Backbone is FROZEN -- only training T and C blocks.")

        tokenizer = model.tokenizer
        model.to(DEVICE)

    # ---- Build datasets ----
    print("Creating datasets...")
    if TASK == "gsm8k":
        train_dataset = GSM8KDataset("train", tokenizer=tokenizer)
        val_dataset = GSM8KDataset("test", tokenizer=tokenizer)
        collate_fn = get_gsm8k_collate_fn(tokenizer)
    else:
        train_dataset = ParityDataset(TRAIN_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)
        val_dataset = ParityDataset(VAL_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)
        collate_fn = get_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ---- Optimizer + scheduler ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    if args.vanilla:
        steps_per_epoch = (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
        total_opt_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_opt_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)
    else:
        steps_per_epoch = (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
        total_opt_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_opt_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    # ---- Training loop ----
    eff_bs = batch_size * GRAD_ACCUM_STEPS
    print(f"\nStarting training: {num_epochs} epochs, "
          f"batch_size={batch_size} x {GRAD_ACCUM_STEPS} accum = {eff_bs} effective"
          f", lr={learning_rate}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        if args.vanilla:
            metrics = train_epoch_vanilla(model, train_loader, optimizer, DEVICE, scheduler)
            print(f"Train - LM Loss: {metrics['lm_loss']:.4f}")
        else:
            metrics = train_epoch_ts(model, train_loader, optimizer, DEVICE, scheduler)
            print(f"Train - LM Loss: {metrics['lm_loss']:.4f}, "
                  f"Think Loss: {metrics['think_loss']:.4f}")

            if DEVICE == "mps":
                torch.mps.empty_cache()

            val_metrics = evaluate_ts(model, val_loader, DEVICE)
            print(f"Val - LM Loss: {val_metrics['lm_loss']:.4f}, "
                  f"Think Loss: {val_metrics['think_loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.2%}")

            if DEVICE == "mps":
                torch.mps.empty_cache()

    # ---- Save checkpoint ----
    ckpt_path = _checkpoint_path(TASK, args.vanilla)
    print(f"\nSaving model to {ckpt_path}...")
    torch.save(model.state_dict(), ckpt_path)
    print("Done!")

    return model


if __name__ == "__main__":
    model = main()
