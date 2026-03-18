"""
EriAmo — Holograficzna Pamięć Kontekstu
Benchmark: EriAmo vs Baseline (sliding window)

Scenariusz: 50-turnowa sesja kodowania z powracającym wzorcem.
Kluczowy wzorzec: "sortowanie logów" — pojawia się w turnie 1,
wraca co ~7 turnów, między nimi szum kontekstowy.
"""

import numpy as np
from typing import List
from dataclasses import dataclass

from core import Item, init_phi, cosine_sim
from embedder import get_embedder
from context import update_context


class BaselineSession:
    def __init__(self, embedder, n=5):
        self.embedder = embedder
        self.n        = n
        self.window: List[Item] = []

    def turn(self, user_input):
        item = Item(content=user_input, embedding=self.embedder.encode(user_input), age=0)
        self.window.append(item)
        if len(self.window) > self.n:
            self.window = self.window[-self.n:]
        for i, it in enumerate(self.window):
            it.age = len(self.window) - 1 - i
        return list(self.window)


@dataclass
class CompareMetrics:
    turn:                   int
    query:                  str
    eriamo_context_drift:   float
    baseline_context_drift: float
    eriamo_window_size:     int
    baseline_window_size:   int
    eriamo_retention:       bool
    baseline_retention:     bool
    phi_drift:              float


def context_drift(window, query_emb):
    if not window:
        return 1.0
    sims = [cosine_sim(item.embedding, query_emb) for item in window]
    return float(1.0 - np.mean(sims))


def retention_check(window, target_emb, threshold=0.1):
    return any(cosine_sim(item.embedding, target_emb) > threshold for item in window)


def build_turns():
    recurring = [
        ("Piszę funkcję sortującą logi według timestampu.",          "sortowanie logów"),
        ("Wracam do sortowania — bug przy tym samym timestampie.",   "sortowanie bug"),
        ("Znowu sortowanie — problem ze strefami czasowymi.",        "sortowanie strefy"),
        ("Sortowanie v2 — przepisuję na algorytm stabilny.",         "sortowanie v2"),
        ("Sortowanie final — testy przeszły, mergujemy.",            "sortowanie final"),
        ("Sortowanie refactor — wydzielam do osobnej klasy.",        "sortowanie refactor"),
        ("Sortowanie cache — dodaję memoizację wyników.",            "sortowanie cache"),
        ("Sortowanie async — wersja asynchroniczna.",                "sortowanie async"),
    ]
    noise = [
        ("Dodaję obsługę wyjątków — co jeśli log jest pusty?",      "wyjątki"),
        ("Refaktoring: wydzielam parser do osobnego modułu.",        "parser moduł"),
        ("Jak skonfigurować logging w Django?",                      "logging Django"),
        ("Dodaję testy jednostkowe do parsera.",                     "testy parser"),
        ("Optymalizacja zapytań do bazy danych.",                    "baza danych"),
        ("Konfiguracja CI/CD w GitHub Actions.",                     "CI/CD"),
        ("Code review — komentarze do PR.",                          "code review"),
        ("Dokumentacja API w OpenAPI.",                              "dokumentacja"),
    ]
    turns = []
    noise_idx = 0
    recurring_idx = 0
    for i in range(50):
        if i == 0 or (i > 0 and i % 7 == 0 and recurring_idx < len(recurring)):
            turns.append(recurring[recurring_idx % len(recurring)])
            recurring_idx += 1
        else:
            turns.append(noise[noise_idx % len(noise)])
            noise_idx += 1
    return turns


KEY_PATTERN = "sortowanie logów"
PRINT_TURNS = {1, 5, 7, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50}


def run_benchmark(turns):
    emb  = get_embedder()
    dim  = emb.dim
    phi  = init_phi(k=4, dim=dim)
    store: List[Item] = []
    baseline = BaselineSession(embedder=emb, n=5)
    key_emb  = emb.encode(KEY_PATTERN)
    all_metrics = []

    print(f"{'Turn':<5} {'Query':<28} {'Drift E':>8} {'Drift B':>8} "
          f"{'Win E':>6} {'Win B':>6} {'Ret E':>6} {'Ret B':>6} {'Φ drift':>8}")
    print("-" * 90)

    for t, (content, query) in enumerate(turns):
        query_emb  = emb.encode(query)
        phi_before = phi.copy()

        window_e, store, phi = update_context(
            store, phi,
            new_turn_content=content,
            query_content=query,
            embedder=emb,
            n=5, threshold=0.45, lr=0.01, top_n_recall=2,
        )
        window_b = baseline.turn(content)

        drift_e = context_drift(window_e, query_emb)
        drift_b = context_drift(window_b, query_emb)
        ret_e   = retention_check(window_e, key_emb)
        ret_b   = retention_check(window_b, key_emb)
        phi_d   = float(np.linalg.norm(phi - phi_before, axis=1).max())

        m = CompareMetrics(t+1, query, drift_e, drift_b,
                           len(window_e), len(window_b), ret_e, ret_b, phi_d)
        all_metrics.append(m)

        if (t + 1) in PRINT_TURNS:
            marker  = " <" if drift_e < drift_b else ("  >" if drift_e > drift_b else "  =")
            print(f"{t+1:<5} {query[:26]:<28} {drift_e:>8.4f} {drift_b:>8.4f}"
                  f"{marker} {len(window_e):>6} {len(window_b):>6} "
                  f"{'TAK' if ret_e else 'NIE':>6} {'TAK' if ret_b else 'NIE':>6} {phi_d:>8.6f}")

    return all_metrics


def summarize(metrics):
    avg_de  = np.mean([m.eriamo_context_drift   for m in metrics])
    avg_db  = np.mean([m.baseline_context_drift for m in metrics])
    avg_we  = np.mean([m.eriamo_window_size      for m in metrics])
    avg_wb  = np.mean([m.baseline_window_size    for m in metrics])
    ret_e   = sum(1 for m in metrics if m.eriamo_retention)
    ret_b   = sum(1 for m in metrics if m.baseline_retention)
    avg_phi = np.mean([m.phi_drift for m in metrics])
    imp     = (avg_db - avg_de) / (avg_db + 1e-8) * 100
    win_red = (avg_wb - avg_we) / (avg_wb + 1e-8) * 100

    print("\n" + "=" * 90)
    print("PODSUMOWANIE — 50 turnów")
    print("=" * 90)
    print(f"  Avg drift kontekstu  — EriAmo: {avg_de:.4f}  |  Baseline: {avg_db:.4f}")
    print(f"  Redukcja dryfu:        {imp:+.1f}%  ({'EriAmo lepszy' if imp > 0 else 'Baseline lepszy'})")
    print(f"  Avg rozmiar okna     — EriAmo: {avg_we:.1f}  |  Baseline: {avg_wb:.1f}")
    print(f"  Redukcja okna:         {win_red:+.1f}%")
    print(f"  Retencja wzorca      — EriAmo: {ret_e}/{len(metrics)}  |  Baseline: {ret_b}/{len(metrics)}")
    print(f"  Avg Φ drift:           {avg_phi:.6f}")
    print()
    print("  Legenda: Drift = cosine distance (niższy = lepiej) | < EriAmo lepszy | > Baseline lepszy")
    print("  Ustaw GEMINI_API_KEY żeby uzyskać miarodajne wyniki semantyczne.")


if __name__ == "__main__":
    turns = build_turns()
    print("=== BENCHMARK: EriAmo vs Baseline (sliding window) ===\n")
    print(f"Scenariusz: {len(turns)} turnów · k=4 · α=0.05 · τ=0.45 · n=5\n")
    metrics = run_benchmark(turns)
    summarize(metrics)
  
