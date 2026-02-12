"""Training loop for Thinking States."""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    TRAIN_SIZE, VAL_SIZE, MIN_FLIPS, MAX_FLIPS,
    GSM8K_BATCH_SIZE, GSM8K_NUM_EPOCHS, GSM8K_LEARNING_RATE,
    GSM8K_FREEZE_BACKBONE, TASK
)
from data import (
    ParityDataset, get_collate_fn,
    GSM8KDataset, get_gsm8k_collate_fn
)
from model import ThinkingStatesModel


GRAD_ACCUM_STEPS = 4  # effective batch size = GSM8K_BATCH_SIZE * GRAD_ACCUM_STEPS


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch. Loss = L_LM + L_T (paper Eq. 6, equal weighting).

    Uses gradient accumulation to maintain a larger effective batch size
    while keeping per-step memory low.  Clears MPS cache periodically.
    """
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

        # Combined loss: L = L_LM + L_T (paper Eq. 6)
        loss = lm_loss
        if think_loss is not None:
            loss = loss + think_loss

        # Scale loss by accumulation steps so the average is correct
        (loss / GRAD_ACCUM_STEPS).backward()

        total_lm_loss += lm_loss.item()
        if think_loss is not None:
            total_think_loss += think_loss.item()
        num_batches += 1

        # Optimizer step every GRAD_ACCUM_STEPS (or at the last batch)
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        # Free MPS cache periodically to prevent OOM buildup
        if device == "mps" and (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.mps.empty_cache()

    return {
        "lm_loss": total_lm_loss / num_batches,
        "think_loss": total_think_loss / num_batches,
    }


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
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

            total_lm_loss += lm_loss.item()
            if think_loss is not None:
                total_think_loss += think_loss.item()
            num_batches += 1

            # Re-run backbone to get logits for accuracy computation
            model._state_to_inject = model.build_state_tensor(
                chunk_thought_ids, input_ids.shape[1], input_ids.shape[0], chunk_thought_masks
            )
            backbone_out = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            logits = backbone_out.logits.float()  # ensure float32 for accuracy
            model._state_to_inject = None

            # Exact-match accuracy on answer tokens
            for i in range(input_ids.shape[0]):
                answer_mask = labels[i] != -100
                if answer_mask.any():
                    answer_positions = answer_mask.nonzero(as_tuple=False).view(-1)
                    preds = logits[i, answer_positions - 1].argmax(dim=-1)
                    trues = labels[i, answer_positions]
                    if preds.shape == trues.shape and torch.equal(preds, trues):
                        correct += 1
                    total += 1

            # Free MPS cache periodically during eval
            if device == "mps" and (step + 1) % 10 == 0:
                torch.mps.empty_cache()

    return {
        "lm_loss": total_lm_loss / num_batches,
        "think_loss": total_think_loss / num_batches,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def main():
    print(f"Using device: {DEVICE}")

    # Initialize model
    print("Loading model...")
    model = ThinkingStatesModel()

    # Print parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params:     {total_params:>14,}")
    print(f"Trainable params: {trainable:>14,} ({100*trainable/total_params:.1f}%)")

    if TASK == "gsm8k" and GSM8K_FREEZE_BACKBONE:
        print("Backbone is FROZEN -- only training T and C blocks.")

    model.to(DEVICE)

    tokenizer = model.tokenizer

    # Create datasets
    print("Creating datasets...")
    if TASK == "gsm8k":
        train_dataset = GSM8KDataset("train", tokenizer=tokenizer)
        val_dataset = GSM8KDataset("test", tokenizer=tokenizer)
        collate_fn = get_gsm8k_collate_fn(tokenizer)
        batch_size = GSM8K_BATCH_SIZE
        num_epochs = GSM8K_NUM_EPOCHS
        learning_rate = GSM8K_LEARNING_RATE
    else:
        train_dataset = ParityDataset(TRAIN_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)
        val_dataset = ParityDataset(VAL_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)
        collate_fn = get_collate_fn(tokenizer)
        batch_size = BATCH_SIZE
        num_epochs = NUM_EPOCHS
        learning_rate = LEARNING_RATE

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    # Training loop
    eff_bs = batch_size * GRAD_ACCUM_STEPS
    print(f"\nStarting training: {num_epochs} epochs, "
          f"batch_size={batch_size} x {GRAD_ACCUM_STEPS} accum = {eff_bs} effective, "
          f"lr={learning_rate}")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Train - LM Loss: {train_metrics['lm_loss']:.4f}, "
              f"Think Loss: {train_metrics['think_loss']:.4f}")

        # Clear memory before eval
        if DEVICE == "mps":
            torch.mps.empty_cache()

        val_metrics = evaluate(model, val_loader, DEVICE)
        print(f"Val - LM Loss: {val_metrics['lm_loss']:.4f}, "
              f"Think Loss: {val_metrics['think_loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.2%}")

        # Clear memory before next epoch
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), "thinking_states_model.pt")
    print("Done!")

    return model


if __name__ == "__main__":
    model = main()
