# Thinking States (Prototype)

This repository contains an experimental implementation of the method proposed in:

**“Latent Reasoning with Supervised Thinking States” (Amos et al., 2026)**

---

## What the Paper Proposes

Modern language models can solve multi-step reasoning tasks by generating explicit chain-of-thought (CoT) text. However:

* Explicit CoT increases inference cost (long outputs).
* Latent (continuous) reasoning methods often require expensive backpropagation through time (BPTT).
* Scaling latent reasoning depth can be unstable.

The paper introduces **Thinking States**, a method that:

1. Performs reasoning in **hidden states**, not in the visible token stream.
2. Supervises those hidden reasoning states using **natural-language thoughts**.
3. Avoids BPTT during training via **teacher-forced state injection**.

---

## Core Idea

The model processes input in fixed-size **chunks**. After each chunk:

1. It generates a short natural-language “thought” describing the current reasoning step.
2. That thought is compressed into a fixed-size **state tensor**.
3. The state is injected into the hidden representations of the next chunk.

Crucially:

* The thought text is **not appended to the context**.
* The reasoning state flows through hidden activations.
* Context length does not grow with reasoning depth.

This creates a recurrent reasoning mechanism inside a standard Transformer.

---

## Training Trick (Why It’s Efficient)

At training time:

* The model does **not** use its own predicted reasoning states.
* Instead, it injects **gold (teacher-provided) reasoning states**.

Because the states are known in advance:

* All chunks can be processed in parallel.
* No backpropagation through time is required.
* Training cost is similar to standard language modeling.

At inference time, recurrence is restored and reasoning states are generated sequentially.

---

## Why This Matters

Thinking States combines the benefits of:

* Explicit chain-of-thought (interpretable supervision)
* Latent reasoning (compact, hidden internal state)
* Efficient training (no BPTT)
* Stable scaling of reasoning depth

The paper shows improvements in:

* Length generalization
* State tracking tasks
* Multi-hop reasoning
* Mathematical reasoning (GSM-style benchmarks)

---

## Conceptual Summary

Instead of writing its thoughts into the token stream, the model:

> Thinks in text → compresses into hidden state → injects into future computation.

This allows reasoning to occur in latent space while still being supervised by language.

---

## Citation

If you use this method, please cite:

```
Amos et al., 2026.
Latent Reasoning with Supervised Thinking States.
```
