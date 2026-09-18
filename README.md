# FengXiao (风小)

> A small experimental model based on the SP-JEPA architecture.
> Character-level text prediction in vector space.

**Status**: `v0.4.4-preflight` — all checks passed, waiting for hardware.

---

## Overview

FengXiao is a small experimental model that learns by predicting in
**vector space** rather than in token space. It is not a chatbot, and it
does not generate text autoregressively.

Instead, it:

1. Encodes context into a continuous representation.
2. Predicts the representation of the *next position*.
3. Retrieves the closest span from a pre-encoded corpus.
4. Appends the span and repeats.

This is an attempt to apply LeCun's JEPA paradigm to character-level
text. It is not expected to outperform large language models. It is
expected to **fail in interesting ways**.
> **Note on "SP"**: SP-JEPA is not a standard academic term.
> The `SP` stands for **Self-Play**, referring to the planned
> difficulty-driven self-play mechanism (see `Explorer`).
> The Self-Play loop is not yet active; currently it functions
> as hard-example mining only.

---

## Architecture

```
                  ┌────────────────────┐
   context  ───►│  context_encoder   │ ───►  ctx_emb
                  └────────────────────┘
                                                │
                                                ▼
                 ┌────────────────────┐
                 │     predictor      │ ───►  pred_emb  (next position)
                 └────────────────────┘
                                                │
                                                ▼
 raw text ───► ┌────────────────────┐
                 │  target_encoder    │ ───►  tgt_emb  (EMA of context)
                 └────────────────────┘
                                                │
                                                ▼
                                   L_JEPA = smooth_l1(pred, tgt)
                                   L_VISREG = variance + sliced-W
```

- **context_encoder**: reads masked input, trained with gradient.
- **target_encoder**: reads full input, EMA-updated, no gradient.
- **predictor**: MLP with LayerNorm + GELU, predicts next-position vector.
- **VISReg**: variance + sliced Wasserstein regularization on predictions,
  to prevent representation collapse.

At inference, the predictor acts as a **navigator** in span space, and a
retrieval layer (L4) resolves the trajectory into text.

---

## Project Structure

```
fengxiao-jepa/
│
├── identity_zh.txt              Self-description corpus (Chinese, 80 lines)
├── identity_en.txt              Self-description corpus (English, 90 lines)
├── input.txt                    General Chinese corpus (short essays, fairy tales)
├── qa_corpus.txt                QA corpus (Chinese)
├── qa_corpus_en.txt             QA corpus (English, simplified)
├── logic_corpus.txt             Generated logic QA pairs
├── logic_qa.jsonl               Structured logic QA (for verifier)
│
├── logic_generator.py           Logic corpus generator + self-verifier
├── prepare_corpus.py            Corpus builder (merges all sources)
├── sp_jepa.py                   Training (single-file, hardware-adaptive)
├── retrieve.py                  Retrieval + calibration (V1.5)
├── l4_decode.py                 Retrieval-based decoding (L4)
│
├── L4_DESIGN.md                 L4 design document
├── CHANGELOG.md                 Project history
│
├── checkpoints_spjepa/          Trained checkpoints (empty for now)
└── l4_cache/                    Span cache (empty for now)
```

---

## Quick Start

### 1. Requirements

```
Python 3.8 ~ 3.12
torch>=2.4.0
numpy>=1.26.0,<2.0
psutil>=5.9.0
```

Install:

```bash
pip install torch numpy psutil
```

### 2. Build the corpus

```bash
python logic_generator.py     # generate logic_corpus.txt (optional)
python prepare_corpus.py      # produce input_augmented.txt
```

### 3. Train

```bash
python sp_jepa.py
```

The script auto-detects CPU / CUDA and adjusts batch size, sequence
length, and AMP settings accordingly. Watch `EMA_diff` in the log — it
should stay below `1e-2`.

### 4. Calibrate retrieval

```bash
python retrieve.py
```

This loads the checkpoint, builds a per-language identity index, runs
the calibration suite, and reports `sep = min(HIT) - max(MISS)`.

- `sep >= 0.10` — healthy
- `sep >= 0.02` — marginal
- `sep <  0.00` — overlapping

### 5. L4 decoding

```python
from l4_decode import RetrievalDecoder

decoder = RetrievalDecoder(
    checkpoint_path="checkpoints_spjepa/fengxiao_best.pth",
    corpus_path="input_augmented.txt",
)
print(decoder.generate("小狐狸走进了", max_steps=10))
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Character-level tokenization** | Simplest vocabulary; no tokenizer drift between train and inference. |
| **Scheduled masking** | `mask_ratio ~ Uniform(0, 0.3)`, so the predictor sees the full visibility spectrum. Needed for L4, which feeds unmasked context. |
| **EMA target encoder** | Updated every step. Momentum ramps `0.990 → 0.996` over the first 5 epochs. |
| **Shared batch mask** | One mask per batch. Required for `view(B, -1)` to work; standard in I-JEPA / V-JEPA. |
| **VISReg loss** | Prevents representation collapse without contrastive negatives. |
| **Explorer** | Difficulty-weighted sampler over training sequences. Optional reward hook for future self-play. |
| **Span library cache** | Built once, keyed by (corpus path, size, mtime, span params). |
| **Rule-based routing** | Identity / math queries bypass retrieval and go straight to templates. |

---

## Known Limitations

- **Output is bounded by the corpus.** FengXiao cannot generate content
  that does not exist in `input_augmented.txt`.
- **Generation quality is limited** by model size (0.74M params) and
  corpus size. Spans may repeat or semantically drift.
- **Retrieval `sep` may be negative** if the corpus contains no
  question-answer structure. Adding `qa_corpus*.txt` improves this.
- **OOV characters** (uppercase Latin, rare Chinese) are silently
  skipped during encoding.
- **Character-level model cannot encode long professional terms**
  (e.g. medical / chemical names). A BPE tokenizer would be required.

---

## Roadmap

| Version | Goal |
|---|---|
| `v0.4.4-preflight` | *(current)* All code sealed, syntax verified, waiting for hardware. |
| `v0.5.0` | First full training run. `retrieve.py` reports `sep > 0`. |
| `v0.6.0` | Update `identity_zh.txt`: "Explorer and Internal Judge are now running." |
| `v0.7.0` | L4 self-play with `verify()` rewards. |
| `v1.0.0` | Tokenizer-agnostic (char / BPE), bilingual, reproducible. |

---

## Philosophy

FengXiao is not trying to be a large language model.

It is a probe into a specific question: **can a predictive model trained
in vector space produce useful text without ever predicting a token?**

The answer, so far, is: *partially, and not gracefully.* That is exactly
what makes it worth building.

---

## License

MIT
