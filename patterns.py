"""
EriAmo — Holograficzna Pamięć Kontekstu
Krok 3: Recall (Φ → T) i aktualizacja wzorców (T → Φ)
"""

import numpy as np
from typing import List
from core import Item, cosine_sim


# ---------------------------------------------------------------------------
# Recall: Φ → T
# ---------------------------------------------------------------------------

def recall_from_phi(store: List[Item], phi: np.ndarray,
                    query_embedding: np.ndarray, top_n: int = 2) -> List[Item]:
    """
    Przywołuje itemy ze store które są najbliższe aktywowanemu wierszowi Φ.

    Przepływ:
      1. Znajdź wiersz Φ najbliższy query_embedding  → best_phi [d]
      2. Znajdź top_n starych itemów najbliższych best_phi → oznacz recalled=True

    Tylko stare itemy (age > 0) są kandydatami — nowe i tak wejdą do okna.

    phi:             [k, d]
    query_embedding: [d]
    """
    # Krok 1: który wiersz Φ odpowiada temu query
    phi_scores = cosine_sim(query_embedding, phi)   # [k]
    best_phi   = phi[np.argmax(phi_scores)]          # [d]

    # Krok 2: itemy w store które ukształtowały ten wzorzec
    old_items = [(item, cosine_sim(item.embedding, best_phi))
                 for item in store if item.age > 0]

    if not old_items:
        return store

    old_items.sort(key=lambda x: -x[1])   # score DESC

    for item, score in old_items[:top_n]:
        item.recalled = True

    return store


# ---------------------------------------------------------------------------
# Aktualizacja Φ: T → Φ
# ---------------------------------------------------------------------------

def update_patterns(phi: np.ndarray, store: List[Item],
                    context_window: List[Item], lr: float = 0.01) -> np.ndarray:
    """
    Aktualizuje Φ na podstawie całego T[t+1]:
      - age == 0       (nowe)
      - recalled       (reaktywowane — właśnie okazały się relewantne)
      - in context_window (wszystko co weszło do okna)

    Zastępuje najsłabszy wiersz Φ (najmniejsza norma po ewolucji)
    w kierunku nowego wzorca z lr.

    phi:            [k, d]  — modyfikowany in-place (kopia zwracana)
    context_window: lista itemów wybranych przez build_context_window
    """
    window_set = set(id(item) for item in context_window)

    active_embs = [
        item.embedding for item in store
        if item.age == 0
        or item.recalled
        or id(item) in window_set
    ]

    if not active_embs:
        return phi

    new_pattern = np.mean(active_embs, axis=0)               # [d]
    new_pattern /= (np.linalg.norm(new_pattern) + 1e-8)      # normalizacja

    phi = phi.copy()
    weakest = int(np.argmin(np.linalg.norm(phi, axis=1)))
    phi[weakest] = (1 - lr) * phi[weakest] + lr * new_pattern
    # Renormalizuj zaktualizowany wiersz
    phi[weakest] /= (np.linalg.norm(phi[weakest]) + 1e-8)

    return phi


# ---------------------------------------------------------------------------
# Testy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core import Embedder, init_phi
    from memory import vacuum_filter, build_context_window

    emb = Embedder(dim=64)
    phi = init_phi(k=4, dim=64)

    def make_item(text, age=1, recalled=False):
        return Item(
            content=text,
            embedding=emb.encode(text),
            age=age,
            recalled=recalled
        )

    store = [
        make_item("architektura transformera",  age=5),
        make_item("wzorzec projektu EriAmo",    age=3),
        make_item("obsługa błędów w pipeline",  age=7),
        make_item("nowy turn: pytanie o kod",   age=0),
    ]

    query_emb = emb.encode("transformer attention layers")

    # --- Test recall ---
    print("=== RECALL ===")
    print("Przed recall:")
    for item in store:
        print(f"  {item}")

    store = recall_from_phi(store, phi, query_emb, top_n=2)

    print("\nPo recall:")
    recalled = [i for i in store if i.recalled]
    for item in store:
        marker = " ← RECALLED" if item.recalled else ""
        print(f"  {item}{marker}")
    print(f"Oznaczonych recalled: {len(recalled)} (oczekiwane: ≤2, tylko age>0)")
    assert all(i.age > 0 for i in recalled), "BŁĄD: recalled item z age==0!"

    # --- Test update_patterns ---
    print("\n=== UPDATE PATTERNS ===")
    kept, _ = vacuum_filter(store, phi)
    window   = build_context_window(kept, phi, n=3)

    phi_before = phi.copy()
    phi_after  = update_patterns(phi, kept, window, lr=0.01)

    diff = np.linalg.norm(phi_after - phi_before, axis=1)
    print(f"Normy zmian wierszy Φ: {diff.round(6)}")
    print(f"Zmieniony wiersz (max diff): {int(np.argmax(diff))}")
    print(f"Nowe normy wierszy Φ: {np.linalg.norm(phi_after, axis=1).round(6)}")
    assert np.allclose(np.linalg.norm(phi_after, axis=1), 1.0, atol=1e-5), \
        "BŁĄD: wiersze Φ nie są znormalizowane!"

    # Sprawdź że tylko jeden wiersz się zmienił
    changed = np.sum(diff > 1e-8)
    print(f"Liczba zmienionych wierszy: {changed} (oczekiwane: 1)")
    assert changed == 1, "BŁĄD: zmieniono więcej niż jeden wiersz Φ!"

    # --- Test: brak aktywnych embeddingów ---
    print("\n=== EDGE CASE: pusty store ===")
    phi_unchanged = update_patterns(phi_after, [], [], lr=0.01)
    assert np.array_equal(phi_after, phi_unchanged), "BŁĄD: Φ zmienione przy pustym store!"
    print("  Φ niezmienione przy pustym store — OK")

    print("\nWszystkie testy przeszły.")
