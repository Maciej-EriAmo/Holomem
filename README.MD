# EriAmo — Holographic Context Architecture

> *"Oddziel przestrzeń obliczeń od przestrzeni interpretacji."*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/numpy-pure-green.svg)](https://numpy.org)
[![Gemini API](https://img.shields.io/badge/gemini-embedding--001-orange.svg)](https://ai.google.dev)

---

## What is EriAmo?

EriAmo is a memory architecture for conversational LLM systems that separates context into two ontologically distinct classes:

- **T-class (Temporal):** current code, active problem, last N turns — lives in the context window
- **P-class (Permanent):** user thinking patterns, dialogue geometry, structural history — compressed into a fixed-size perception matrix **Φ ∈ R^(k×d)** outside the window

Φ modulates the attention key space through a cross-attention bias. It does not compete with task content for tokens. It shapes *how the model listens* — not *what the model does*.

```
Without EriAmo:  window = Identity + Memory_1..n + Code   O(n), grows forever
With EriAmo:     window = Code + Last_N_turns              O(N), constant
                 Φ      = fixed perception matrix, off-window
```

---

## Benchmark Results

**50-turn coding session** with recurring pattern (`"sortowanie logów"`), returning every ~7 turns with noise in between. Embeddings: **gemini-embedding-001** (dim=768).

| Metric | EriAmo | Baseline (sliding window) |
|---|---|---|
| Avg context drift | **0.2421** | 0.2935 |
| Drift reduction | **−17.5%** | — |
| Avg window size | **2.6** | 4.6 |
| Window reduction | **−43%** | — |
| Pattern retention | **50/50 (100%)** | 50/50 |

Key observation: the advantage grows with session length. At turn 50, EriAmo drift is 0.1801 vs baseline 0.2654 — a 32% difference. Vacuum and recall compound over time.

---

## Architecture

### Modified Attention

```python
# Cross-attention: each K token queries the perception geometry
Φ_weights = softmax(K · Φᵀ / √d)      # [T, k]
Φ_context = Φ_weights · Φ              # [T, d]
K'        = K + α · Φ_context          # [T, d]

scores = softmax(Q · K'ᵀ / √d)
```

Zero architectural changes. Additive pre-Softmax bias — structurally analogous to [Harmonic Attention (QRM)](https://github.com/Maciej615/harmonic-attention). Potentially composable.

### Information Lifecycle

```
            recall
    Φ  ------------>  T
    ▲                  │
    │  consolidation   │ vacuum
    │                  ▼
    └──────────── [decay]
```

### Turn Cycle

```
1. recall_from_phi(Φ[t], query)     # reactivate relevant patterns
2. append new item (age=0)
3. vacuum_filter(store, Φ[t])       # remove low-relevance items
4. build_context_window(store, Φ[t])
5. update_patterns(Φ[t], store, window)
6. age++ for all items
```

---

## Files

```
core.py      — Item, cosine_sim (pure numpy), Embedder, init_phi
memory.py    — vacuum_filter, build_context_window
patterns.py  — recall_from_phi, update_patterns
context.py   — update_context (full turn cycle)
embedder.py  — GeminiEmbedder (gemini-embedding-001), factory
session.py   — EriAmoSession (loop + LLM call + metrics)
benchmark.py — EriAmo vs sliding window baseline, 50 turns
```

---

## Quick Start

```bash
# Install (no compilation — works on Termux ARM)
pip install numpy google-genai

export GEMINI_API_KEY="your_key"
python benchmark.py
```

**Offline mode** (no API key): placeholder embedder for structural testing.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| k | 4 | perception vectors in Φ |
| N | 5 | context window size |
| threshold | 0.45 | vacuum threshold |
| lr | 0.01 | Φ learning rate |
| α | 0.05 | hologram strength |
| top_n_recall | 2 | items reactivated by recall |

---

## Relation to Harmonic Attention

| | QRM (Harmonic Attention) | EriAmo (Holographic) |
|---|---|---|
| Bias type | Symmetric (token-token) | Asymmetric (token-hologram) |
| Tested | ✓ ~9% perplexity | Memory layer ✓, attention open |
| Composable? | — | Direct research question |

---

## Open Questions

- **Attention layer:** K' = K + α·Φ_context requires model internals. Perplexity impact unknown. Natural next step for teams with full model access (SpeakLeash/Bielik).
- **Φ at scale:** k=4, lr=0.01 sufficient for 50 turns. Behavior at 500+ turns: open.
- **Composability with QRM:** both additive pre-Softmax — direct research question.

---

## Paper

[`eriamo_paper.md`](eriamo_paper.md) (English) · [`eriamo_paper_pl.md`](eriamo_paper_pl.md) (Polish)

---

## License

Apache 2.0 — non-exclusive. Attribution required on redistribution.

---

## Author

**Maciej Mazur** — concept, architecture, research direction
GitHub: [@Maciej-EriAmo](https://github.com/Maciej-EriAmo) · Medium: [@drwisz](https://medium.com/@drwisz)
*Independent AI Researcher & Systems Architect*

*Mathematical formalization: Claude (Anthropic)*
