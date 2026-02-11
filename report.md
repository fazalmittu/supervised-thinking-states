# Supervised Thinking States on GPT‑2: Implementation Report

## 1) Why I Chose This Paper
[Write your “why” here.]

## 2) Paper in a Nutshell (3–4 sentences)
Thinking States adds a recurrent “hidden scratchpad” to LLMs without growing the context window: the model generates short thought sequences while processing input, compresses them into a fixed‑size state, and injects that state into future tokens. This recovers the sequential reasoning benefits of chain‑of‑thought, but the thoughts are generated during input processing and never appended to the main context. Crucially, because the thoughts are natural‑language tokens, the method can use teacher forcing and parallel training rather than backpropagation through time. citeturn10view0

## 3) My Implementation (What I Built + Assumptions)
I focused on the GSM task and implemented the full Thinking States loop on top of GPT‑2. The backbone is GPT‑2, and I wired in two extra modules:

- **Thinking Block T**: a single causal Transformer block (GPT‑2 block) that autoregressively generates a thought sequence from chunk representations. It’s initialized as an independent copy of the last layer of the backbone, its unembedding is copied from the backbone, and its embedding layer is shared with the backbone. citeturn13view0
- **Compression Block C**: a single causal Transformer block that maps the variable‑length thought sequence into a fixed‑size state by taking the last `c` hidden states; it’s initialized as an independent copy of the first layer of the backbone (excluding embeddings). citeturn13view0

For data/alignment, I implemented a teacher‑LLM alignment pipeline for GSM8K. It parses GSM8K solutions into steps, asks a teacher model for the earliest question position where each step is inferable, converts those to token positions, and assigns steps to chunks. Each chunk’s target thought is the concatenation of assigned steps, terminated with an EOS token. This matches the paper’s idea of chunk‑level natural‑language supervision for thoughts. citeturn11view0

## 4) Training Pipeline (Equations + Flow)
The paper’s core equations are implemented directly. The input is split into `K` chunks of length `c`, with embeddings \(X_i \in \mathbb{R}^{c \times d}\). At each step we inject the current state into the shallow‑layer representations and forward through the backbone:

\[
\tilde{X}_i = X_i + S_i
\]
\[
H^{out}_i = M_{\theta}(\tilde{X}_i \mid \tilde{X}_{<i})
\]

The Thinking Block generates a variable‑length thought sequence:

\[
Z_{i+1} = T(H^{out}_i)
\]

The Compression Block maps thoughts to the next state:

\[
S_{i+1} = C(Z_{i+1}) \in \mathbb{R}^{c \times d}
\]

For training, we use gold thought targets \(Z^*_i\). The target state is:

\[
S^*_i = C(Z^*_i)
\]
and we inject it instead of the model’s state:
\[
\tilde{X}_i = X_i + S^*_i
\]

Because all \(Z^*_i\) are known up front, all chunks can be processed in parallel (no BPTT). The objective is:
\[
\mathcal{L} = \mathcal{L}_{LM} + \sum_{i=1}^{K} \mathcal{L}_T(Z_i, Z^*_i)
\]
This is exactly what the training code does: teacher‑forced state injection for the backbone, plus a cross‑entropy loss on T’s predicted thought tokens. citeturn11view0

## 5) Why This Paper Matters (Latent Reasoning, Without BPTT)
Most latent‑reasoning approaches that iterate internal “thinking” require backpropagation through time because the latent state depends on previous latent predictions. Thinking States avoids that by supervising the thoughts in natural language and injecting teacher‑forced states during training, so all chunks can be processed in parallel. This keeps training costs close to standard LM fine‑tuning, while still giving the model recurrent reasoning state. The paper explicitly contrasts this with BPTT‑based approaches (e.g., Coconut as a baseline) and shows the training cost scales much more favorably. citeturn10view0turn11view0

## 6) Inference Pipeline
At inference time, we run the chunk‑recurrent loop sequentially:
1) inject the current state at a shallow layer,  
2) forward through the backbone to a deep layer,  
3) generate thought tokens autoregressively with T until EOS,  
4) compress them with C into the next state,  
5) repeat for the next chunk, then generate the final answer.

The paper also describes a speculative prefill algorithm that exploits the fact that many chunks emit trivial thoughts (just EOS), allowing faster prefill without changing results. I do **not** implement speculative prefill yet, and I also do not keep a KV‑cache across chunks, so inference is slower than the paper’s optimized pipeline. citeturn11view0

## 7) Code Walkthrough (How This Maps to the Logic)
- `config.py`: task selection, chunk sizes, loss weights, special tokens.
- `align_gsm8k.py`: teacher alignment for GSM8K; builds chunk‑level thought targets.
- `data.py`: GSM8K dataset + collate, builds BOS/EOS‑wrapped thought targets.
- `modules.py`: Thinking Block T and Compression Block C (GPT‑2 blocks, paper‑style).
- `model.py`: hook‑based state injection/extraction, teacher‑forced training, loss computation, initialization from backbone.
- `train.py`: training loop with LM + thinking loss, teacher‑forced state injection.
- `inference.py`: sequential chunk‑by‑chunk inference and answer generation.

## 8) What’s Missing vs. the Paper (Current Gaps)
- Speculative prefill optimization for faster prefill. citeturn11view0turn13view0
- KV‑cache reuse across chunks.
- Full‑scale GSM8K training and rigorous evaluation of thought quality.
- Alignment quality evaluation (how often thoughts match teacher reasoning).
