"""
EriAmo — Holograficzna Pamięć Kontekstu
Krok 6: Główna pętla sesji — LLM call + metryki
"""

import os
import time
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from core import Item, init_phi
from embedder import get_embedder
from context import update_context


# ---------------------------------------------------------------------------
# Metryki jednego turnu
# ---------------------------------------------------------------------------

@dataclass
class TurnMetrics:
    turn:          int
    store_size:    int
    window_size:   int
    phi_drift:     float
    recall_count:  int
    vacuum_removed:int
    latency_ms:    float


# ---------------------------------------------------------------------------
# Sesja
# ---------------------------------------------------------------------------

class EriAmoSession:
    """
    Główna pętla sesji. Łączy warstwę pamięci z wywołaniem LLM.

    Architektura:
      user → [recall + dodaj + vacuum + okno] → LLM API → output → [update Φ]

    Parametry:
      api_key_gemini  — klucz Gemini (embeddingi + generacja)
      k               — liczba wektorów percepcji w Φ
      n               — rozmiar okna kontekstu
      threshold       — próg vacuum
      lr              — learning rate aktualizacji Φ
      top_n_recall    — ile itemów przywołuje recall
    """

    def __init__(
        self,
        api_key_gemini: Optional[str] = None,
        k:              int   = 4,
        n:              int   = 5,
        threshold:      float = 0.3,
        lr:             float = 0.01,
        top_n_recall:   int   = 2,
        llm_model:      str   = "gemini-2.0-flash",
    ):
        self.embedder     = get_embedder(api_key_gemini)
        self.phi          = init_phi(k=k, dim=self.embedder.dim)
        self.store:  List[Item] = []
        self.metrics: List[TurnMetrics] = []
        self.history: List[dict] = []   # {role, content} dla LLM

        self.n            = n
        self.threshold    = threshold
        self.lr           = lr
        self.top_n_recall = top_n_recall
        self.llm_model    = llm_model

        # LLM client — ten sam klucz co embeddingi
        key = api_key_gemini or os.environ.get("GEMINI_API_KEY")
        if key:
            from google import genai
            self._llm_client = genai.Client(api_key=key)
        else:
            self._llm_client = None

    # -----------------------------------------------------------------------
    # Główna metoda
    # -----------------------------------------------------------------------

    def turn(self, user_input: str) -> str:
        """
        Jeden turn sesji.
          1. Aktualizacja pamięci (recall → vacuum → okno → Φ)
          2. Budowanie promptu z okna kontekstu
          3. Wywołanie LLM
          4. Zapis metryk
        """
        t0 = time.perf_counter()

        phi_before   = self.phi.copy()
        store_before = len(self.store)

        # Krok 1: aktualizacja pamięci
        window, self.store, self.phi = update_context(
            store            = self.store,
            phi              = self.phi,
            new_turn_content = user_input,
            query_content    = user_input,
            embedder         = self.embedder,
            n                = self.n,
            threshold        = self.threshold,
            lr               = self.lr,
            top_n_recall     = self.top_n_recall,
        )

        vacuum_removed = store_before + 1 - len(self.store)  # +1 = nowy item
        recall_count   = sum(1 for i in self.store if i.recalled)  # po resecie = 0
        phi_drift      = float(np.linalg.norm(self.phi - phi_before, axis=1).max())

        # Krok 2: budowanie promptu z okna
        context_text = self._build_prompt(window, user_input)

        # Krok 3: wywołanie LLM
        response = self._call_llm(context_text, user_input)

        # Krok 4: metryki
        latency = (time.perf_counter() - t0) * 1000
        self.metrics.append(TurnMetrics(
            turn           = len(self.metrics) + 1,
            store_size     = len(self.store),
            window_size    = len(window),
            phi_drift      = phi_drift,
            recall_count   = recall_count,
            vacuum_removed = max(0, vacuum_removed),
            latency_ms     = latency,
        ))

        # Zapisz do historii LLM
        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": response})

        return response

    # -----------------------------------------------------------------------
    # Budowanie promptu
    # -----------------------------------------------------------------------

    def _build_prompt(self, window: List[Item], user_input: str) -> str:
        """
        Składa okno kontekstu w tekst dla LLM.
        Tylko T-class — bez systemu, bez listy zasad.
        """
        if not window:
            return user_input

        context_parts = []
        for item in window:
            if item.content != user_input:   # pomiń bieżący input (jest w window jako age=0)
                label = "[przywołane]" if item.recalled else f"[t-{item.age}]"
                context_parts.append(f"{label} {item.content}")

        if not context_parts:
            return user_input

        context_str = "\n".join(context_parts)
        return f"Kontekst sesji:\n{context_str}\n\nBieżące: {user_input}"

    # -----------------------------------------------------------------------
    # Wywołanie LLM
    # -----------------------------------------------------------------------

    def _call_llm(self, context_text: str, user_input: str) -> str:
        if self._llm_client is None:
            # Tryb bez API — zwraca echo z info o oknie
            return f"[tryb offline] Otrzymano: {user_input!r}"

        from google.genai import types

        contents = []
        # Historia poprzednich turnów (bez bieżącego — jest w context_text)
        for msg in self.history[-6:]:   # ostatnie 3 pary max
            contents.append(types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["content"])]
            ))
        # Bieżący turn z kontekstem pamięci
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=context_text)]
        ))

        response = self._llm_client.models.generate_content(
            model=self.llm_model,
            contents=contents,
        )
        return response.text

    # -----------------------------------------------------------------------
    # Raport metryk
    # -----------------------------------------------------------------------

    def report(self) -> str:
        if not self.metrics:
            return "Brak turnów."

        lines = [
            f"{'Turn':>5} {'store':>6} {'okno':>5} {'Φ drift':>9} "
            f"{'vacuum':>7} {'ms':>7}",
            "-" * 48,
        ]
        for m in self.metrics:
            lines.append(
                f"{m.turn:>5} {m.store_size:>6} {m.window_size:>5} "
                f"{m.phi_drift:>9.6f} {m.vacuum_removed:>7} {m.latency_ms:>7.1f}"
            )

        avg_drift   = np.mean([m.phi_drift for m in self.metrics])
        avg_latency = np.mean([m.latency_ms for m in self.metrics])
        lines += [
            "-" * 48,
            f"  avg Φ drift: {avg_drift:.6f}   avg latency: {avg_latency:.1f} ms"
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Testy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SESJA OFFLINE (bez klucza API) ===\n")

    session = EriAmoSession(n=4, threshold=0.15)

    turns = [
        "Piszę funkcję sortującą logi według timestampu.",
        "Dodaję obsługę wyjątków — co jeśli log jest pusty?",
        "Refaktoring: wydzielam parser do osobnego modułu.",
        "Wracam do sortowania — bug przy logach z tym samym timestampem.",
        "Jak działa stabilne sortowanie w Pythonie?",
        "Dodaję testy jednostkowe do parsera.",
        "Znowu bug w sortowaniu — tym razem przy strefach czasowych.",
    ]

    for user_input in turns:
        response = session.turn(user_input)
        m = session.metrics[-1]
        print(f"User:  {user_input[:60]!r}")
        print(f"Resp:  {response[:60]!r}")
        print(f"       store={m.store_size} okno={m.window_size} "
              f"vacuum={m.vacuum_removed} drift={m.phi_drift:.6f}")
        print()

    print("=== METRYKI SESJI ===")
    print(session.report())

    # Asercje
    print("\n=== ASERCJE ===")
    assert len(session.metrics) == len(turns), "BŁĄD: liczba metryk != liczba turnów"
    print(f"  metryk: {len(session.metrics)} — OK")

    assert all(m.store_size > 0 for m in session.metrics), "BŁĄD: pusty store"
    print(f"  store zawsze > 0 — OK")

    assert all(m.window_size <= session.n for m in session.metrics), "BŁĄD: okno > n"
    print(f"  okno zawsze <= n={session.n} — OK")

    phi_drifts = [m.phi_drift for m in session.metrics]
    assert all(d >= 0 for d in phi_drifts), "BŁĄD: ujemny drift"
    assert max(phi_drifts) > 0, "BŁĄD: Φ nie ewoluowało"
    print(f"  Φ ewoluowało (max drift={max(phi_drifts):.6f}) — OK")

    print("\nWszystkie testy przeszły.")
