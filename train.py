"""Training loop for Thinking States."""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    LAMBDA_THINK, TRAIN_SIZE, VAL_SIZE, MIN_FLIPS, MAX_FLIPS,
    GSM8K_BATCH_SIZE, GSM8K_NUM_EPOCHS, GSM8K_LEARNING_RATE, GSM8K_LAMBDA_THINK,
    TASK
)
from data import (
    ParityDataset, get_collate_fn,
    GSM8KDataset, get_gsm8k_collate_fn
)
from model import ThinkingStatesModel


def train_epoch(model, dataloader, optimizer, device, lambda_think):
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
        chunk_thought_masks = batch.get("chunk_thought_masks")

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            chunk_thought_ids=chunk_thought_ids,
            chunk_thought_masks=chunk_thought_masks
        )

        lm_loss = outputs["lm_loss"]
        think_loss = outputs["thinking_loss"]

        # Combined loss
        loss = lm_loss
        if think_loss is not None:
            loss = loss + lambda_think * think_loss

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

            # Check accuracy - run forward again to get logits with state injection
            # The model.forward() sets up the injection hooks
            model._state_to_inject = model.build_state_tensor(
                chunk_thought_ids, input_ids.shape[1], input_ids.shape[0], chunk_thought_masks
            )
            backbone_out = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            logits = backbone_out.logits
            model._state_to_inject = None

            # Exact-match accuracy on answer tokens
            for i in range(input_ids.shape[0]):
                answer_mask = labels[i] != -100
                if answer_mask.any():
                    answer_positions = answer_mask.nonzero().squeeze(-1)
                    preds = logits[i, answer_positions - 1].argmax(dim=-1)
                    trues = labels[i, answer_positions]
                    if torch.equal(preds, trues):
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
    if TASK == "gsm8k":
        train_dataset = GSM8KDataset("train", tokenizer=tokenizer)
        val_dataset = GSM8KDataset("test", tokenizer=tokenizer)
        collate_fn = get_gsm8k_collate_fn(tokenizer)
        batch_size = GSM8K_BATCH_SIZE
        num_epochs = GSM8K_NUM_EPOCHS
        learning_rate = GSM8K_LEARNING_RATE
        lambda_think = GSM8K_LAMBDA_THINK
    else:
        train_dataset = ParityDataset(TRAIN_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)
        val_dataset = ParityDataset(VAL_SIZE, MIN_FLIPS, MAX_FLIPS, tokenizer)
        collate_fn = get_collate_fn(tokenizer)
        batch_size = BATCH_SIZE
        num_epochs = NUM_EPOCHS
        learning_rate = LEARNING_RATE
        lambda_think = LAMBDA_THINK

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

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, lambda_think)
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
