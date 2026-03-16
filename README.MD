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
Without EriAmo:  window = Identity + Memory_1..n + Code   O(n)
With EriAmo:     window = Code + Last_N_turns              O(N), constant
                 Φ      = fixed perception matrix, off-window
```

---

## Benchmark Results

Evaluated on a 7-turn coding session with a recurring pattern, using **gemini-embedding-001** (dim=768):

| Metric | EriAmo | Baseline (sliding window) |
|---|---|---|
| Avg context drift | **0.1761** | 0.1863 |
| Drift reduction | **+5.5%** | — |
| Avg window size | **2.6** | 3.6 |
| Window reduction | **−28%** | — |
| Pattern retention | **7/7** | 7/7 |

Full benchmark output: see [`benchmark.py`](benchmark.py)

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

Zero architectural changes. Additive pre-Softmax bias — structurally analogous to [Harmonic Attention (QRM)](https://github.com/Maciej615/harmonic-attention).

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
benchmark.py — EriAmo vs sliding window baseline
```

---

## Quick Start

```bash
# Install (no compilation — works on Termux ARM)
pip install numpy google-genai

# Set API key
export GEMINI_API_KEY="your_key"

# Run benchmark
python benchmark.py

# Interactive session
from session import EriAmoSession
session = EriAmoSession()
response = session.turn("Napisz funkcję sortującą logi.")
print(session.report())
```

**Offline mode** (no API key): runs with a placeholder embedder for structural testing.

---

## Relation to Harmonic Attention

Both EriAmo and [Harmonic Attention (QRM)](https://github.com/Maciej615/harmonic-attention) are additive pre-Softmax biases with zero architectural change:

| | QRM (Harmonic Attention) | EriAmo (Holographic) |
|---|---|---|
| Bias type | Symmetric (token-token) | Asymmetric (token-hologram) |
| Source | Phase geometry of hidden states | Compressed perception matrix Φ |
| Tested | ✓ ~9% perplexity improvement | Memory layer ✓, attention layer open |
| Composable? | — | Direct research question |

---

## Open Questions

- **Attention layer empirical validation:** K' = K + α·Φ_context requires model internals access. Does it yield perplexity improvements comparable to QRM (~9%)? Natural next step for teams with full model access.
- **Φ collapse at scale:** k=4, lr=0.01 may collapse toward mean over 100+ turns. Momentum or orthogonality constraint may be needed.
- **Composability with QRM:** both are additive pre-Softmax — compositional testing is a direct research question.

---

## Paper

Available in [`eriamo_paper.md`](eriamo_paper.md) (English) and [`eriamo_paper_pl.md`](eriamo_paper_pl.md) (Polish).

---

## Author

**Maciej Mazur** — concept, architecture, research direction
GitHub: [@Maciej-Er# EriAmo — Holographic Context Architecture

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
Without EriAmo:  window = Identity + Memory_1..n + Code   O(n)
With EriAmo:     window = Code + Last_N_turns              O(N), constant
                 Φ      = fixed perception matrix, off-window
```

---

## Benchmark Results

Evaluated on a 7-turn coding session with a recurring pattern, using **gemini-embedding-001** (dim=768):

| Metric | EriAmo | Baseline (sliding window) |
|---|---|---|
| Avg context drift | **0.1761** | 0.1863 |
| Drift reduction | **+5.5%** | — |
| Avg window size | **2.6** | 3.6 |
| Window reduction | **−28%** | — |
| Pattern retention | **7/7** | 7/7 |

Full benchmark output: see [`benchmark.py`](benchmark.py)

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

Zero architectural changes. Additive pre-Softmax bias — structurally analogous to [Harmonic Attention (QRM)](https://github.com/Maciej615/harmonic-attention).

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
benchmark.py — EriAmo vs sliding window baseline
```

---

## Quick Start

```bash
# Install (no compilation — works on Termux ARM)
pip install numpy google-genai

# Set API key
export GEMINI_API_KEY="your_key"

# Run benchmark
python benchmark.py

# Interactive session
from session import EriAmoSession
session = EriAmoSession()
response = session.turn("Napisz funkcję sortującą logi.")
print(session.report())
```

**Offline mode** (no API key): runs with a placeholder embedder for structural testing.

---

## Relation to Harmonic Attention

Both EriAmo and [Harmonic Attention (QRM)](https://github.com/Maciej615/harmonic-attention) are additive pre-Softmax biases with zero architectural change:

| | QRM (Harmonic Attention) | EriAmo (Holographic) |
|---|---|---|
| Bias type | Symmetric (token-token) | Asymmetric (token-hologram) |
| Source | Phase geometry of hidden states | Compressed perception matrix Φ |
| Tested | ✓ ~9% perplexity improvement | Memory layer ✓, attention layer open |
| Composable? | — | Direct research question |

---

## Open Questions

- **Attention layer empirical validation:** K' = K + α·Φ_context requires model internals access. Does it yield perplexity improvements comparable to QRM (~9%)? Natural next step for teams with full model access.
- **Φ collapse at scale:** k=4, lr=0.01 may collapse toward mean over 100+ turns. Momentum or orthogonality constraint may be needed.
- **Composability with QRM:** both are additive pre-Softmax — compositional testing is a direct research question.

---

## Paper

Available in [`eriamo_paper.md`](eriamo_paper.md) (English) and [`eriamo_paper_pl.md`](eriamo_paper_pl.md) (Polish).

---

## Author

**Maciej Mazur** — concept, architecture, research direction
GitHub: [@Maciej-EriAmo](https://github.com/Maciej-EriAmo) · Medium: [@drwisz](https://medium.com/@drwisz)

*Mathematical formalization: Claude (Anthropic)*

---

## See Also

- [Harmonic Attention (QRM)](https://github.com/Maciej-EriAmo/holomem) — additive phase-geometry bias for transformer attention
                 ](https://github.com/Maciej615) · Medium: [@drwisz](https://medium.com/@drwisz)

*Mathematical formalization: Claude (Anthropic)*

---

## See Also

- [Harmonic Attention (QRM)](https://github.com/Maciej615/harmonic-attention) — additive phase-geometry bias for transformer attention
