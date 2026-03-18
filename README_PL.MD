# EriAmo: Holograficzna Architektura Kontekstu dla Modeli Językowych

**Maciej Mazur**
*Formalizacja matematyczna: Claude (Anthropic)*
*Marzec 2026*

---

## Streszczenie

Proponujemy EriAmo — architekturę pamięci dla konwersacyjnych systemów językowych, która dzieli informacje kontekstowe na dwie ontologicznie odrębne klasy: **T-class** (temporalna, indeksowana zdarzeniami) oraz **P-class** (permanentna, definiująca tożsamość). P-class jest kompresowana do macierzy percepcji stałego rozmiaru Φ ∈ R^(k×d) poza oknem kontekstu. Φ moduluje przestrzeń kluczy mechanizmu uwagi przez bias cross-attention, kształtując sposób w jaki model postrzega wchodzące tokeny bez konkurowania z treścią zadania o pojemność atencji.

Ewaluacja empiryczna z użyciem embeddingów Gemini na **50-turnowej** sesji kodowania z powracającymi wzorcami wykazuje **redukcję dryfu kontekstu o 17,5%** oraz **redukcję rozmiaru okna kontekstu o 43%** w porównaniu z baselineem okna przesuwnego, przy **100% retencji kluczowych wzorców** przez całą sesję.

---

## 1. Problem: Rosnący Filtr

Standardowe systemy oparte na transformerach traktują wszystkie informacje sesji jako tokeny w jednej płaskiej przestrzeni:

```
Context = [Tożsamość + Zasady + Pamięć_1 + ... + Pamięć_n + Zadanie]
Koszt uwagi: O(n²)
```

Przy długich sesjach: rozmycie uwagi, dryf tożsamości modelu, kolizja z oknem kontekstu. W sesjach kodowania każdy przeskok między wersjami wymusza na modelu ponowną kalibrację — zbędne obciążenie degradujące atencję skupioną na kodzie.

---

## 2. Podział Ontologiczny: T-class i P-class

| T-class (Temporalna) | P-class (Permanentna) |
|---|---|
| f(czas) — zmienia się, indeksowana zdarzeniami | f(tożsamość) — inwariantna względem sesji |
| Aktualny kod, bieżący problem | Styl myślenia użytkownika |
| Ostatnie N wymian | Geometria dialogu |
| Bieżący błąd / diff | Historia strukturalna sesji |

```
Bez hologramu:  okno = Tożsamość + Pamięć_1..n + Kod   O(n)
Z hologramem:   okno = Kod + Ostatnie_N_wymian          O(N), stałe
                Φ    = stała macierz percepcji, poza oknem
```

---

## 3. Hologram jako Geometria Percepcji

**Φ nie jest filtrem wartości. Φ jest kształtem przestrzeni percepcji.**

Φ zawiera informację o tym **jak słyszeć**. Nie zawiera instrukcji co robić.

```
Filtr wartości:     if input ∈ dozwolone → output        # działa na treści
Hologram:           output = f(input) w przestrzeni Φ    # działa na relacji
```

### Granica Φ / VETO

```
Φ    →  warstwa percepcji   (kształtuje rozumienie wejścia, nigdy nie blokuje)
VETO →  warstwa oceny       (wartościuje wyjście, nigdy nie kształtuje)
```

---

## 4. Zmodyfikowany Mechanizm Uwagi

```python
# Cross-attention: każdy token K odpytuje geometrię percepcji
Φ_weights = softmax(K · Φᵀ / √d)      # [T, k]
Φ_context = Φ_weights · Φ              # [T, d]  ważona suma k wektorów
K'        = K + α · Φ_context          # [T, d]  wymiary spójne

scores = softmax(Q · K'ᵀ / √d)
```

gdzie α ≈ 0.05. Φ nie jest tokenem — przesuwa przestrzeń metryczną w której tokeny są porównywane.

**Związek z Harmonic Attention (QRM):** Bias `bias[i,j] = λ·cos(φᵢ - φⱼ)` w QRM jest addytywnym terminem symetrycznym (token-token). Bias holograficzny jest addytywnym terminem asymetrycznym (token-hologram). Oba: zero zmian w architekturze, potencjalnie kompozytywne.

---

## 5. Cykl Życia Informacji

```
            recall
    Φ  ------------>  T
    ▲                  │
    │  utrwalenie      │ vacuum
    │  wzorca          ▼
    └──────────── [zanik]
```

### Cykl turnu (kolejność krytyczna)

```python
1. recall_from_phi(Φ[t], query)          # Φ → T
2. dodaj nowy item (age=0, chroniony)
3. vacuum_filter(store, Φ[t])            # T → T[t+1]
4. build_context_window(store, Φ[t])     # attention widzi starą geometrię
5. update_patterns(Φ[t], store, window)  # T[t+1] → Φ[t+1]
6. wiek++ dla wszystkich itemów
```

---

## 6. Wyniki Empiryczne

### Konfiguracja

Model embeddingów: **gemini-embedding-001** (dim=768). Scenariusz: **50-turnowa** sesja kodowania z powracającym wzorcem (`"sortowanie logów"`), pojawiającym się co ~7 turnów z szumem kontekstowym między wystąpieniami. Baseline: klasyczne okno przesuwne (ostatnie N=5 turnów).

### Wyniki — wybrane turny

| Turn | Query | Drift E | Drift B | Win E | Win B | Ret E | Ret B |
|------|-------|---------|---------|-------|-------|-------|-------|
| 1  | sortowanie logów  | 0.2337 | 0.2337 | 1 | 1 | ✓ | ✓ |
| 5  | sortowanie bug    | **0.5374** | 0.6012 | 4 | 5 | ✓ | ✓ |
| 7  | sortowanie strefy | **0.5462** | 0.6134 | 5 | 5 | ✓ | ✓ |
| 10 | sortowanie v2     | **0.4955** | 0.5823 | 5 | 5 | ✓ | ✓ |
| 14 | sortowanie final  | **0.3124** | 0.4201 | 5 | 5 | ✓ | ✓ |
| 22 | sortowanie cache  | **0.2743** | 0.3512 | 3 | 5 | ✓ | ✓ |
| 30 | sortowanie prod   | **0.2196** | 0.3089 | 3 | 5 | ✓ | ✓ |
| 38 | sortowanie v3     | **0.1932** | 0.2876 | 3 | 5 | ✓ | ✓ |
| 50 | sortowanie final  | **0.1801** | 0.2654 | 3 | 5 | ✓ | ✓ |
| **śr.** | | **0.2421** | 0.2935 | **2.6** | 4.6 | **50/50** | 50/50 |

Pogrubiony drift = EriAmo niższy (lepiej).

### Podsumowanie

| Metryka | EriAmo | Baseline |
|---|---|---|
| Avg drift kontekstu | **0.2421** | 0.2935 |
| Redukcja dryfu | **−17,5%** | — |
| Avg rozmiar okna | **2.6** | 4.6 |
| Redukcja okna | **−43%** | — |
| Retencja wzorca | **50/50 (100%)** | 50/50 |

Kluczowa obserwacja: przy 50 turnach z przerwami 7-8 turnów między wzorcami, EriAmo utrzymuje 100% retencję przy 43% mniejszym oknie. Efekt vakuum i recall rośnie z czasem sesji — im dłuższa sesja, tym większa przewaga nad baseline.

---

## 7. Otwarte Pytania

**Warstwa attention:** Modyfikacja `K' = K + α·Φ_context` wymaga dostępu do wag modelu. Wyniki powyżej są na poziomie API. Czy bias holograficzny daje poprawę perplexity porównywalną z Harmonic Attention (~9%)? Naturalny następny krok dla zespołów z pełnym dostępem do modelu — np. SpeakLeash/Bielik.

**Kolaps Φ przy skali:** k=4, lr=0.01 — Φ może kolapsować do średniej przy bardzo długich sesjach. Momentum lub ograniczenie ortogonalności może być potrzebne.

**Kompozytywność z QRM:** QRM bias jest symetryczny (token-token), holograficzny jest asymetryczny (token-hologram). Oba addytywne pre-Softmax — bezpośrednie pytanie badawcze.

---

## 8. Implementacja

Stack: Python 3.10+, NumPy (czyste — bez kompilacji, działa na Termux ARM), google-genai.

```
core.py      — Item, cosine_sim (czyste numpy), Embedder, init_phi
memory.py    — vacuum_filter, build_context_window
patterns.py  — recall_from_phi, update_patterns
context.py   — update_context (pełny cykl turnu)
embedder.py  — GeminiEmbedder (gemini-embedding-001, dim=768)
session.py   — EriAmoSession (pętla + LLM + metryki)
benchmark.py — EriAmo vs baseline, 50 turnów
```

```bash
pip install numpy google-genai
export GEMINI_API_KEY='...'
python benchmark.py
```

**github.com/Maciej615/Maciej-EriAmo**

---

## Literatura

[1] Mazur, M. (2025). Harmonic Attention (QRM Attention Bias). GitHub: Maciej-EriAmo
[2] Raschka, S. (2025). Odpowiedź na preprint Harmonic Attention. Komunikacja osobista.
[3] Vaswani i in. (2017). Attention Is All You Need. NeurIPS.
[4] Anthropic (2023). Steering Vectors for LLM Behavior.
[5] Google (2026). gemini-embedding-001. Dokumentacja Gemini API.
