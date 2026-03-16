"""
EriAmo — Holograficzna Pamięć Kontekstu
Krok 5: Prawdziwy embedder — Gemini API
"""

import os
import numpy as np
from typing import Optional
from core import Embedder


# ---------------------------------------------------------------------------
# Gemini Embedder
# ---------------------------------------------------------------------------

class GeminiEmbedder(Embedder):
    """
    Model: gemini-embedding-001
      - wymiar: 768 (przez output_dimensionality, domyślnie 3072)
      - ta sama przestrzeń dla wszystkich itemów i Φ
      - normalizacja do sfery jednostkowej
    """

    MODEL = "gemini-embedding-001"
    DIM   = 768

    def __init__(self, api_key=None):
        super().__init__(dim=self.DIM)

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Brak klucza API. Ustaw GEMINI_API_KEY lub przekaż api_key=..."
            )

        from google import genai
        self._client = genai.Client(api_key=key)

    def encode(self, text):
        from google.genai import types

        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=self.DIM,
            ),
        )

        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(api_key=None, dim=64):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if key:
        print("Embedder: Gemini gemini-embedding-001 (dim=768)")
        return GeminiEmbedder(api_key=key)
    else:
        print(f"Embedder: placeholder hash-based (dim={dim}) — brak GEMINI_API_KEY")
        return Embedder(dim=dim)


# ---------------------------------------------------------------------------
# Testy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core import init_phi, cosine_sim
    from context import update_context

    emb = get_embedder()
    dim = emb.dim
    phi = init_phi(k=4, dim=dim)

    print(f"\n=== TEST ENCODERA (dim={dim}) ===")
    v1 = emb.encode("sortowanie logów w pipeline")
    v2 = emb.encode("filtrowanie błędów w systemie logów")
    v3 = emb.encode("architektura holograficzna transformera")

    print(f"  shape:   {v1.shape}")
    print(f"  norm v1: {np.linalg.norm(v1):.6f}  (powinno być ~1.0)")

    sim_12 = cosine_sim(v1, v2)
    sim_13 = cosine_sim(v1, v3)
    print(f"  cosine_sim(logi, błędy_logów)    = {sim_12:.4f}")
    print(f"  cosine_sim(logi, transformer)    = {sim_13:.4f}")

    if isinstance(emb, GeminiEmbedder):
        assert sim_12 > sim_13, f"BŁĄD semantyczny: {sim_12:.4f} powinno być > {sim_13:.4f}"
        print("  semantyka OK: podobne > odległe")
    else:
        print("  (placeholder: semantyka niezgwarantowana)")

    print(f"\n=== MINI SESJA (3 turny) ===")
    store = []
    turns = [
        ("Piszę funkcję sortującą logi.",         "sortowanie logów"),
        ("Dodaję obsługę wyjątków do pipeline.",  "wyjątki pipeline"),
        ("Wracam do sortowania — znalazłem bug.", "bug sortowanie"),
    ]
    for t, (content, query) in enumerate(turns):
        window, store, phi = update_context(
            store, phi,
            new_turn_content=content,
            query_content=query,
            embedder=emb,
            n=4,
            threshold=0.3,
        )
        print(f"  Turn {t+1}: store={len(store)}  okno={len(window)}")
        for item in window:
            print(f"    age={item.age}  {item.content[:55]!r}")

    print("\nWszystkie testy przeszły.")
