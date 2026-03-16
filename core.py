"""
EriAmo — Holograficzna Pamięć Kontekstu
Krok 1: Struktury danych i embedding
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Struktura itemu
# ---------------------------------------------------------------------------

@dataclass
class Item:
    content:   str
    embedding: np.ndarray
    age:       int  = 0
    recalled:  bool = False

    def __repr__(self):
        return f"Item(age={self.age}, recalled={self.recalled}, content={self.content[:40]!r})"


# ---------------------------------------------------------------------------
# Cosine similarity (wektor vs wektor lub wektor vs macierz)
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray):
    """
    a: [d]
    b: [d] lub [k, d]
    zwraca: float lub np.ndarray [k]
    Czyste numpy — brak zależności od sklearn.
    """
    a = a.flatten()
    norm_a = np.linalg.norm(a) + 1e-8
    if b.ndim == 1:
        return float(np.dot(a, b) / (norm_a * (np.linalg.norm(b) + 1e-8)))
    # b: [k, d]
    norms_b = np.linalg.norm(b, axis=1) + 1e-8
    return np.dot(b, a) / (norm_a * norms_b)


# ---------------------------------------------------------------------------
# Embedding — wymienne źródło
# ---------------------------------------------------------------------------

class Embedder:
    """
    Wrapper nad źródłem embeddingów.
    Domyślnie: deterministyczny hash → wektor (do testów, bez API).
    W produkcji: podmień encode() na wywołanie modelu.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """
        Placeholder: deterministyczny pseudo-embedding z tekstu.
        Zachowuje podobieństwo semantyczne słabo, ale pozwala testować logikę.
        Podmień na prawdziwy model przed produkcją.
        """
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)   # normalizacja do sfery jednostkowej


# ---------------------------------------------------------------------------
# Inicjalizacja Φ
# ---------------------------------------------------------------------------

def init_phi(k: int, dim: int) -> np.ndarray:
    """
    Φ ∈ R^(k × d) — k wektorów percepcji, losowo zainicjalizowane na sferze.
    """
    rng = np.random.default_rng(seed=42)
    phi = rng.standard_normal((k, dim)).astype(np.float32)
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    return phi / norms


# ---------------------------------------------------------------------------
# Szybki test struktur
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    emb = Embedder(dim=64)

    # Test Item
    v = emb.encode("Sortowanie logów w pipeline")
    item = Item(content="Sortowanie logów w pipeline", embedding=v)
    print(item)
    print(f"  embedding shape: {item.embedding.shape}, norm: {np.linalg.norm(item.embedding):.4f}")

    # Test cosine_sim wektor vs wektor
    v2 = emb.encode("Filtrowanie logów w systemie")
    sim = cosine_sim(v, v2)
    print(f"\ncosine_sim(log1, log2) = {sim:.4f}  (powinno być między -1 a 1)")

    # Test cosine_sim wektor vs macierz
    phi = init_phi(k=4, dim=64)
    scores = cosine_sim(v, phi)
    print(f"cosine_sim(v, Phi[4x64]) = {scores}  (shape: {scores.shape})")

    # Test init_phi
    print(f"\nPhi shape: {phi.shape}")
    print(f"Phi row norms: {np.linalg.norm(phi, axis=1)}")  # powinno być ~1.0
