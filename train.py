"""Training loop for Thinking States."""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    LAMBDA_THINK, TRAIN_SIZE, VAL_SIZE, MIN_FLIPS, MAX_FLIPS
)
from data import ParityDataset, get_collate_fn
from model import ThinkingStatesModel


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_lm_loss = 0.0
    total_think_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        chunk_thought_ids = batch["chunk_thought_ids"]

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            chunk_thought_ids=chunk_thought_ids
        )

        lm_loss = outputs["lm_loss"]
        think_loss = outputs["thinking_loss"]

        # Combined loss
        loss = lm_loss
        if think_loss is not None:
            loss = loss + LAMBDA_THINK * think_loss

        loss.backward()
        optimizer.step()

        total_lm_loss += lm_loss.item()
        if think_loss is not None:
            total_think_loss += think_loss.item()
        num_batches += 1

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
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            chunk_thought_ids = batch["chunk_thought_ids"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chunk_thought_ids=chunk_thought_ids
            )

            lm_loss = outputs["lm_loss"]
            think_loss = outputs["thinking_loss"]

            total_lm_loss += lm_loss.item()
            if think_loss is not None:
                total_think_loss += think_loss.item()
            num_batches += 1

            # Check accuracy - run forward again to get logits with state injection
            # The model.forward() sets up the injection hooks
            model._state_to_inject = model.build_state_tensor(
                chunk_thought_ids, input_ids.shape[1], input_ids.shape[0]
            )
            backbone_out = model.backbone(input_ids=input_ids)
            logits = backbone_out.logits
            model._state_to_inject = None

            # Find answer position (where labels != -100)
            for i in range(input_ids.shape[0]):
                answer_mask = labels[i] != -100
                if answer_mask.any():
                    answer_pos = answer_mask.nonzero()[0].item()
                    pred_token = logits[i, answer_pos - 1].argmax().item()
                    true_token = labels[i, answer_pos].item()

                    if pred_token == true_token:
                        correct += 1
                    total += 1

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
    model.to(DEVICE)

    tokenizer = model.tokenizer

    # Create datasets
    print("Creating datasets...")
    train_dataset = ParityDataset(TRAIN_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)
    val_dataset = ParityDataset(VAL_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)

    collate_fn = get_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Train - LM Loss: {train_metrics['lm_loss']:.4f}, "
              f"Think Loss: {train_metrics['think_loss']:.4f}")

        val_metrics = evaluate(model, val_loader, DEVICE)
        print(f"Val - LM Loss: {val_metrics['lm_loss']:.4f}, "
              f"Think Loss: {val_metrics['think_loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.2%}")

    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), "thinking_states_model.pt")
    print("Done!")

    return model


if __name__ == "__main__":
    model = main()
