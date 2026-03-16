"""
EriAmo — Holograficzna Pamięć Kontekstu
Krok 7: Test porównawczy — EriAmo vs baseline

Mierzone metryki:
  1. Dryf kontekstu     — jak bardzo okno oddala się od query (cosine distance)
  2. Retencja wzorców   — czy itemy semantycznie bliskie query przeżywają vacuum
  3. Rozmiar okna       — ile tokenów "zużywa" każdy system
  4. Stabilność Φ       — jak szybko geometria percepcji się zmienia
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from core import Item, init_phi, cosine_sim
from embedder import get_embedder
from context import update_context


# ---------------------------------------------------------------------------
# Baseline: płaskie okno bez pamięci (klasyczny sliding window)
# ---------------------------------------------------------------------------

class BaselineSession:
    """
    Klasyczny transformer context: ostatnie N turnów w oknie, bez vacuum,
    bez Φ, bez recall. Rosnąca lista tokenów.
    """

    def __init__(self, embedder, n: int = 5):
        self.embedder = embedder
        self.n        = n
        self.window: List[Item] = []

    def turn(self, user_input: str) -> List[Item]:
        item = Item(
            content   = user_input,
            embedding = self.embedder.encode(user_input),
            age       = 0,
        )
        self.window.append(item)
        # Sliding window: zawsze ostatnie N
        if len(self.window) > self.n:
            self.window = self.window[-self.n:]
        for i, it in enumerate(self.window):
            it.age = len(self.window) - 1 - i
        return list(self.window)


# ---------------------------------------------------------------------------
# Metryki porównawcze
# ---------------------------------------------------------------------------

@dataclass
class CompareMetrics:
    turn:               int
    query:              str

    # Dryf kontekstu: średnia cosine distance między query a itemami w oknie
    # Niższy = okno bardziej skupione na query
    eriamo_context_drift:   float
    baseline_context_drift: float

    # Rozmiar okna (liczba itemów)
    eriamo_window_size:   int
    baseline_window_size: int

    # Czy najważniejszy stary item (najbliższy query) jest w oknie
    eriamo_retention:   bool
    baseline_retention: bool


def context_drift(window: List[Item], query_emb: np.ndarray) -> float:
    """Średnia cosine distance okna od query. Niższy = lepiej skupiony."""
    if not window:
        return 1.0
    sims = [cosine_sim(item.embedding, query_emb) for item in window]
    return float(1.0 - np.mean(sims))


def retention_check(window: List[Item], target_emb: np.ndarray,
                    threshold: float = 0.1) -> bool:
    """Czy okno zawiera item podobny do target (cosine_sim > threshold)?"""
    for item in window:
        if cosine_sim(item.embedding, target_emb) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Scenariusz testowy
# ---------------------------------------------------------------------------

# Symulacja sesji kodowania z powracającymi tematami.
# Kluczowy wzorzec: "sortowanie logów" pojawia się w turnie 1 i wraca w 5 i 7.
# Między nimi — różne tematy (szum kontekstowy).

TURNS = [
    # (user_input, query_do_pomiaru_dryfu)
    ("Piszę funkcję sortującą logi według timestampu.",         "sortowanie logów"),
    ("Dodaję obsługę wyjątków — co jeśli log jest pusty?",     "wyjątki"),
    ("Refaktoring: wydzielam parser do osobnego modułu.",       "parser moduł"),
    ("Jak skonfigurować logging w Django?",                     "logging Django"),
    ("Wracam do sortowania — bug przy tym samym timestampie.",  "sortowanie bug"),
    ("Dodaję testy jednostkowe do parsera.",                    "testy parser"),
    ("Znowu sortowanie — problem ze strefami czasowymi.",       "sortowanie strefy"),
]

# Embedding "sortowania" — wzorzec który powinien być retencjonowany
KEY_PATTERN = "sortowanie logów"


def run_benchmark():
    emb = get_embedder()
    dim = emb.dim

    phi    = init_phi(k=4, dim=dim)
    store: List[Item] = []

    baseline = BaselineSession(embedder=emb, n=5)

    key_emb    = emb.encode(KEY_PATTERN)
    all_metrics: List[CompareMetrics] = []

    print(f"{'Turn':<5} {'Query':<30} {'Drift E':>8} {'Drift B':>8} "
          f"{'Win E':>6} {'Win B':>6} {'Ret E':>6} {'Ret B':>6}")
    print("-" * 80)

    for t, (content, query) in enumerate(TURNS):
        query_emb = emb.encode(query)

        # EriAmo
        window_e, store, phi = update_context(
            store, phi,
            new_turn_content = content,
            query_content    = query,
            embedder         = emb,
            n                = 5,
            threshold        = 0.15,
            lr               = 0.01,
            top_n_recall     = 2,
        )

        # Baseline
        window_b = baseline.turn(content)

        # Metryki
        drift_e = context_drift(window_e, query_emb)
        drift_b = context_drift(window_b, query_emb)
        ret_e   = retention_check(window_e, key_emb)
        ret_b   = retention_check(window_b, key_emb)

        m = CompareMetrics(
            turn                    = t + 1,
            query                   = query,
            eriamo_context_drift    = drift_e,
            baseline_context_drift  = drift_b,
            eriamo_window_size      = len(window_e),
            baseline_window_size    = len(window_b),
            eriamo_retention        = ret_e,
            baseline_retention      = ret_b,
        )
        all_metrics.append(m)

        ret_e_str = "TAK" if ret_e else "NIE"
        ret_b_str = "TAK" if ret_b else "NIE"
        drift_marker = " <" if drift_e < drift_b else ("  >" if drift_e > drift_b else "  =")

        print(f"{t+1:<5} {query:<30} {drift_e:>8.4f} {drift_b:>8.4f}"
              f"{drift_marker} {len(window_e):>6} {len(window_b):>6} "
              f"{ret_e_str:>6} {ret_b_str:>6}")

    return all_metrics


# ---------------------------------------------------------------------------
# Podsumowanie
# ---------------------------------------------------------------------------

def summarize(metrics: List[CompareMetrics]):
    avg_drift_e = np.mean([m.eriamo_context_drift   for m in metrics])
    avg_drift_b = np.mean([m.baseline_context_drift for m in metrics])
    avg_win_e   = np.mean([m.eriamo_window_size      for m in metrics])
    avg_win_b   = np.mean([m.baseline_window_size    for m in metrics])
    ret_e       = sum(1 for m in metrics if m.eriamo_retention)
    ret_b       = sum(1 for m in metrics if m.baseline_retention)

    print("\n" + "=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)
    print(f"  Avg drift kontekstu  — EriAmo: {avg_drift_e:.4f}  |  Baseline: {avg_drift_b:.4f}")
    improvement = (avg_drift_b - avg_drift_e) / (avg_drift_b + 1e-8) * 100
    print(f"  Redukcja dryfu:        {improvement:+.1f}%  ({'EriAmo lepszy' if improvement > 0 else 'Baseline lepszy'})")
    print(f"  Avg rozmiar okna     — EriAmo: {avg_win_e:.1f}     |  Baseline: {avg_win_b:.1f}")
    print(f"  Retencja wzorca      — EriAmo: {ret_e}/{len(metrics)}  |  Baseline: {ret_b}/{len(metrics)}")
    print()
    print("  Legenda kolumn:")
    print("   Drift: cosine distance okna od query (niższy = lepiej skupiony)")
    print("   <      EriAmo drift niższy (lepiej)")
    print("   >      Baseline drift niższy")
    print("   Ret:   czy kluczowy wzorzec 'sortowanie logów' jest w oknie")
    print()
    print("  Uwaga: z placeholder embedderem wyniki semantyczne są losowe.")
    print("  Podłącz GEMINI_API_KEY żeby uzyskać miarodajne porównanie.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== BENCHMARK: EriAmo vs Baseline ===\n")
    print(f"Scenariusz: {len(TURNS)} turnów, powracający wzorzec '{KEY_PATTERN}'\n")

    metrics = run_benchmark()
    summarize(metrics)
