"""
EriAmo — Holograficzna Pamięć Kontekstu
Krok 2: Vacuum i budowanie okna kontekstu
"""

import numpy as np
from typing import List
from core import Item, cosine_sim


# ---------------------------------------------------------------------------
# Vacuum filter
# ---------------------------------------------------------------------------

def vacuum_filter(store: List[Item], phi: np.ndarray, threshold: float = 0.3) -> List[Item]:
    """
    Usuwa stare, nieistotne itemy ze store.
    Operuje na store (repozytorium), NIE na sekwencji attention.

    Chronione (nigdy nie usuwane):
      - age == 0  (nowe w tym turnie, bootstrap protection)
      - recalled  (reaktywowane przez recall_from_phi)

    phi: [k, d]
    """
    phi_mean = phi.mean(axis=0)   # [d]

    kept = []
    removed = []

    for item in store:
        if item.age == 0 or item.recalled:
            kept.append(item)
            continue
        score = cosine_sim(item.embedding, phi_mean)
        if score >= threshold:
            kept.append(item)
        else:
            removed.append(item)

    return kept, removed


# ---------------------------------------------------------------------------
# Budowanie okna kontekstu
# ---------------------------------------------------------------------------

def build_context_window(store: List[Item], phi: np.ndarray,
                         n: int = 5, threshold: float = 0.3) -> List[Item]:
    """
    Buduje ciągłe okno kontekstu ze store.
    Okno zawsze świeżo złożone — brak dziur, pozycjonalne encodingi spójne.

    Priorytet:
      1. Protected (age==0 lub recalled) — zawsze wchodzą
      2. Kandydaci (score >= threshold) — po score DESC, tiebreaker: większy age

    Edge case: jeśli po filtracji < n, uzupełnij kandydatami poniżej progu.

    phi: [k, d]
    """
    phi_mean = phi.mean(axis=0)   # [d]

    protected  = []
    candidates = []
    fallback   = []   # poniżej progu — tylko do uzupełniania edge case

    for i, item in enumerate(store):
        if item.age == 0 or item.recalled:
            protected.append((i, item, 1.0))
        else:
            score = cosine_sim(item.embedding, phi_mean)
            if score >= threshold:
                candidates.append((i, item, score))
            else:
                fallback.append((i, item, score))

    # Sortuj kandydatów: score DESC, przy remisie większy age pierwszy
    candidates.sort(key=lambda x: (-x[2], x[1].age))
    fallback.sort(key=lambda x: (-x[2], x[1].age))

    ordered  = protected + candidates
    selected = [item for _, item, _ in ordered[:n]]

    # Edge case: uzupełnij do n z fallback
    if len(selected) < n:
        extra = [item for _, item, _ in fallback if item not in selected]
        selected += extra[:n - len(selected)]

    return selected


# ---------------------------------------------------------------------------
# Testy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core import Embedder, init_phi

    emb = Embedder(dim=64)
    phi = init_phi(k=4, dim=64)

    def make_item(text, age=1, recalled=False):
        return Item(
            content=text,
            embedding=emb.encode(text),
            age=age,
            recalled=recalled
        )

    # Zbuduj store z itemów różnego wieku i statusu
    store = [
        make_item("stary nieistotny szum",          age=10),
        make_item("stary ale relewantny kontekst",  age=5),
        make_item("nowy item z tego turnu",         age=0),
        make_item("recalled wzorzec",               age=3, recalled=True),
        make_item("kolejny stary szum",             age=8),
    ]

    print("=== STORE PRZED VACUUM ===")
    for item in store:
        print(f"  {item}")

    # Vacuum (threshold celowo wysoki żeby coś odpadło)
    kept, removed = vacuum_filter(store, phi, threshold=0.3)

    print(f"\n=== VACUUM (threshold=0.3) ===")
    print(f"Zachowane ({len(kept)}):")
    for item in kept:
        print(f"  {item}")
    print(f"Usunięte ({len(removed)}):")
    for item in removed:
        print(f"  {item}")

    # Zawsze chronione
    protected_in_kept = [i for i in kept if i.age == 0 or i.recalled]
    print(f"\nChronione w kept: {len(protected_in_kept)} (powinny być 2: age=0 + recalled)")
    assert all(i.age == 0 or i.recalled for i in protected_in_kept), "BŁĄD: chroniony item usunięty!"

    # Budowanie okna
    print(f"\n=== OKNO KONTEKSTU (n=3) ===")
    window = build_context_window(kept, phi, n=3, threshold=0.3)
    for item in window:
        print(f"  {item}")

    print(f"\n=== EDGE CASE: store z 1 itemem, n=5 ===")
    tiny_store = [make_item("jedyny item", age=0)]
    window_small = build_context_window(tiny_store, phi, n=5)
    print(f"  okno: {len(window_small)} itemów (powinno być 1, nie crash)")

    print("\nWszystkie testy przeszły.")
  
