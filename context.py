"""
EriAmo — Holograficzna Pamięć Kontekstu
Krok 4: Pełny cykl turnu — update_context
"""

import numpy as np
from typing import List, Tuple
from core import Item, Embedder, cosine_sim
from memory import vacuum_filter, build_context_window
from patterns import recall_from_phi, update_patterns


# ---------------------------------------------------------------------------
# Pełny cykl turnu
# ---------------------------------------------------------------------------

def update_context(
    store:             List[Item],
    phi:               np.ndarray,
    new_turn_content:  str,
    query_content:     str,
    embedder:          Embedder,
    n:                 int   = 5,
    threshold:         float = 0.3,
    lr:                float = 0.01,
    top_n_recall:      int   = 2,
) -> Tuple[List[Item], List[Item], np.ndarray]:
    """
    Jeden turn systemu. Kolejność zgodna z teorią (sekcja 4):

      Recall(Φ[t]) → dodaj nowe → Vacuum → okno z Φ[t] → Pattern(T[t+1]) → wiek++

    Zwraca:
      context_window  — sekwencja itemów do attention (ciągła, max n)
      store           — zaktualizowane repozytorium
      phi             — zaktualizowana macierz wzorców Φ[t+1]
    """

    # Krok 1: Recall — Φ[t] → T
    # Szukamy powiązanych wzorców PRZED dodaniem nowego turnu
    query_emb = embedder.encode(query_content)
    store = recall_from_phi(store, phi, query_emb, top_n=top_n_recall)

    # Krok 2: Dodaj nowy item (age=0, chroniony przed vacuum)
    new_item = Item(
        content=new_turn_content,
        embedding=embedder.encode(new_turn_content),
        age=0,
        recalled=False,
    )
    store.append(new_item)

    # Krok 3: Vacuum — T_raw → T[t+1]
    store, removed = vacuum_filter(store, phi, threshold=threshold)

    # Krok 4: Zbuduj okno z Φ[t] — attention turnu widzi starą geometrię
    context_window = build_context_window(store, phi, n=n, threshold=threshold)

    # Krok 5: Pattern(T[t+1]) — aktualizuj Φ po vacuum i po złożeniu okna
    phi = update_patterns(phi, store, context_window, lr=lr)

    # Krok 6: Resetuj flagi recalled, zwiększ wiek
    for item in store:
        item.recalled = False
        item.age += 1

    return context_window, store, phi


# ---------------------------------------------------------------------------
# Testy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core import init_phi

    emb = Embedder(dim=64)
    phi = init_phi(k=4, dim=64)
    store: List[Item] = []

    turns = [
        ("Piszę funkcję sortującą logi.",          "sortowanie logów"),
        ("Dodaję obsługę wyjątków do pipeline.",   "obsługa błędów"),
        ("Refaktoring modułu parsowania.",          "parsowanie pipeline"),
        ("Pytanie o architekturę transformera.",   "transformer attention"),
        ("Wracam do sortowania — znalazłem bug.",  "sortowanie logów bug"),
    ]

    print("=== SYMULACJA 5 TURNÓW ===\n")

    for t, (content, query) in enumerate(turns):
        phi_before = phi.copy()
        window, store, phi = update_context(
            store, phi,
            new_turn_content=content,
            query_content=query,
            embedder=emb,
            n=4,
            threshold=0.1,   # niski próg — placeholder embeddingi słabo korelują
        )

        phi_drift = np.linalg.norm(phi - phi_before, axis=1).max()
        recalled  = [i for i in store if i.recalled]  # po resecie — zawsze 0 tu

        print(f"Turn {t+1}: {content[:45]!r}")
        print(f"  store: {len(store)} itemów  |  okno: {len(window)} itemów  |  Φ drift: {phi_drift:.6f}")
        print(f"  okno contents:")
        for item in window:
            print(f"    age={item.age}  {item.content[:50]!r}")
        print()

    # --- Asercje końcowe ---
    print("=== ASERCJE ===")

    # Store rośnie i nie jest pusty
    assert len(store) > 0, "BŁĄD: store pusty po turnach!"
    print(f"  store po 5 turnach: {len(store)} itemów — OK")

    # Wszystkie age > 0 po zakończeniu (wiek inkrementowany)
    assert all(i.age > 0 for i in store), "BŁĄD: item z age==0 po turnie!"
    print(f"  wszystkie age > 0 — OK")

    # Żaden item nie ma recalled=True po turnie (resetowane w kroku 6)
    assert not any(i.recalled for i in store), "BŁĄD: recalled nie zresetowane!"
    print(f"  wszystkie recalled=False po turnie — OK")

    # Φ zmieniło się przez sesję
    phi_init = init_phi(k=4, dim=64)
    total_drift = np.linalg.norm(phi - phi_init)
    assert total_drift > 0, "BŁĄD: Φ nie ewoluowało!"
    print(f"  Φ ewoluowało (total drift={total_drift:.4f}) — OK")

    # Normy wierszy Φ ~1.0
    assert np.allclose(np.linalg.norm(phi, axis=1), 1.0, atol=1e-5), \
        "BŁĄD: wiersze Φ nie znormalizowane!"
    print(f"  normy wierszy Φ ~1.0 — OK")

    print("\nWszystkie testy przeszły.")
