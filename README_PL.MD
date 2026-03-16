# EriAmo: Holograficzna Architektura Kontekstu dla Modeli Językowych

**Maciej Mazur**
*Formalizacja matematyczna: Claude (Anthropic)*
*Marzec 2026*

---

## Streszczenie

Proponujemy EriAmo — architekturę pamięci dla konwersacyjnych systemów językowych, która dzieli informacje kontekstowe na dwie ontologicznie odrębne klasy: **T-class** (temporalna, indeksowana zdarzeniami) oraz **P-class** (permanentna, definiująca tożsamość). Informacje P-class są kompresowane do macierzy percepcji stałego rozmiaru Φ ∈ R^(k×d), zamiast zajmować pozycje tokenów w oknie kontekstu. Φ moduluje przestrzeń kluczy mechanizmu uwagi poprzez bias cross-attention, kształtując sposób w jaki model postrzega wchodzące tokeny bez konkurowania z treścią zadania o pojemność atencji. Mechanizm vacuum selektywnie zachowuje istotne itemy T-class między turnami, a mechanizm recall reaktywuje historycznie relewantne wzorce gdy semantycznie odpowiadają nowemu zapytaniu.

Ewaluacja empiryczna z użyciem embeddingów Gemini na scenariuszu 7-turnowej sesji kodowania wykazuje **redukcję dryfu kontekstu o 5,5%** oraz **redukcję rozmiaru okna kontekstu o 28%** w porównaniu z baselineem okna przesuwnego, przy pełnej retencji kluczowych wzorców we wszystkich turnach.

---

## 1. Problem: Rosnący Filtr

Standardowe systemy oparte na transformerach traktują wszystkie informacje sesji jako tokeny w jednej płaskiej przestrzeni:

```
Context = [Tożsamość + Zasady + Pamięć_1 + ... + Pamięć_n + Zadanie]
Koszt uwagi: O(n²)
```

Wraz z rosnącą długością sesji pojawiają się trzy tryby awarii. Po pierwsze, **rozmycie uwagi**: tokeny relewantne dla zadania konkurują równorzędnie z tokenami tożsamości i historii o wagę atencji. Po drugie, **dryf tożsamości**: model musi ponownie odczytywać swoją rolę i ograniczenia przy każdym przejściu w przód. Po trzecie, **kolizja kontekstu**: przy długich sesjach okno się zapełnia, wymuszając obcięcie historii lub treści zadania.

W sesjach kodowania każdy przeskok między wersjami wymusza na modelu ponowną kalibrację w narracji kontekstu — zbędne obciążenie degradujące atencję skupioną na kodzie.

---

## 2. Podział Ontologiczny: T-class i P-class

Definiujemy binarną partycję informacji sesji:

| T-class (Temporalna) | P-class (Permanentna) |
|---|---|
| f(czas) — zmienia się, indeksowana zdarzeniami | f(tożsamość) — inwariantna względem sesji |
| Aktualny kod, bieżący problem | Styl myślenia użytkownika |
| Ostatnie N wymian | Geometria dialogu |
| Bieżący błąd / diff | Historia strukturalna sesji |

**Reguła podziału:** jeśli informacja ma indeks czasowy → T-class (w oknie). W przeciwnym razie → P-class (hologram).

```
Bez hologramu:  okno = Tożsamość + Pamięć_1..n + Kod   O(n)
Z hologramem:   okno = Kod + Ostatnie_N_wymian          O(N), stałe
                Φ    = stała macierz percepcji, poza oknem
```

---

## 3. Hologram jako Geometria Percepcji

Optyczny hologram koduje pełną informację obiektu we wzorcu interferencji stałego rozmiaru. Odczyt przez dowolną wiązkę referencyjną rekonstruuje obiekt bez zajmowania przestrzeni proporcjonalnej do złożoności obiektu.

Kluczowe rozróżnienie w naszej architekturze: **Φ nie jest filtrem wartości. Φ jest kształtem przestrzeni percepcji.**

```
Filtr wartości (standardowy):     Hologram (proponowany):
  if input ∈ dozwolone:             output = f(input)
      output = f(input)               w przestrzeni zdefiniowanej przez Φ
  else:
      odrzuć

  Działa na treści.                 Działa na relacji między
                                    treścią a odbiorcą.
```

Φ zawiera informację o tym **jak słyszeć**. Nie zawiera instrukcji co robić. Analogia do system promptu: Φ nie jest listą zasad — jest układem współrzędnych orientującym model bez zajmowania jego pamięci roboczej.

### Granica Φ / VETO

```
Φ    →  warstwa percepcji   (kształtuje rozumienie wejścia, nigdy nie blokuje)
VETO →  warstwa oceny       (wartościuje wyjście, nigdy nie kształtuje)

wejście → [Φ moduluje percepcję] → LLM → [VETO ocenia output] → wyjście
```

---

## 4. Zmodyfikowany Mechanizm Uwagi

Standardowy scaled dot-product attention:

```
scores = softmax(Q·Kᵀ / √d)
```

Proponowany mechanizm z Φ ∈ R^(k×d):

```python
# Cross-attention: każdy token K odpytuje geometrię percepcji
Φ_weights = softmax(K · Φᵀ / √d)      # [T, k]
Φ_context = Φ_weights · Φ              # [T, d]  ważona suma k wektorów
K'        = K + α · Φ_context          # [T, d]  wymiary spójne

scores = softmax(Q · K'ᵀ / √d)
```

gdzie α ≈ 0.05 kontroluje siłę hologramu. Φ nie staje się tokenem. Przesuwa przestrzeń metryczną w której tokeny są porównywane.

**Związek z Harmonic Attention (QRM):** Addytywny bias `bias[i,j] = λ·cos(φᵢ - φⱼ)` w QRM dodaje termin geometrii fazowej przed Softmaxem. Oba mechanizmy są addytywnymi biasami pre-Softmax bez zmiany architektury. Różnica: bias QRM jest symetryczny (relacja token-token); bias holograficzny jest asymetryczny (relacja token-hologram). Oba są potencjalnie kompozytywne.

---

## 5. Cykl Życia Informacji

Trzy ścieżki dla itemów T-class:

```
T → Φ       (utrwalenie wzorca przez update_patterns)
T → Vacuum  (zanik itemów o niskiej relewancji)
Φ → T       (recall: reaktywacja historycznie relewantnych wzorców)
```

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

**Ochrona bootstrap:** itemy z `age=0` i `recalled=True` nigdy nie są usuwane przez vacuum, chroniąc nowe informacje przed odrzuceniem zanim zdążą wpłynąć na Φ.

**Wymaganie przestrzeni embeddingów:** `embedding()` i Φ muszą dzielić tę samą przestrzeń wektorową — mean-pool ostatniej warstwy ukrytej tego samego modelu. Użycie zewnętrznego encodera złamałoby geometrię cross-attention.

---

## 6. Wyniki Empiryczne

### Konfiguracja

Model embeddingów: **gemini-embedding-001** (dim=768, Gemini API). Scenariusz: 7-turnowa sesja kodowania z powracającym wzorcem (`"sortowanie logów"`, wprowadzonym w turnie 1, przywoływanym ponownie w turnach 5 i 7, z niepowiązanymi turnami pomiędzy). Baseline: klasyczne okno przesuwne (ostatnie N=5 turnów, bez vacuum, bez recall).

### Metryki

- **Dryf kontekstu:** średnia cosine distance między itemami okna a embeddingiem zapytania. Niższy = lepiej skupiony.
- **Rozmiar okna:** liczba itemów przekazywanych do LLM na turn.
- **Retencja wzorca:** czy kluczowy powracający item jest obecny w oknie.

### Wyniki

| Turn | Query | Drift E | Drift B | Win E | Win B | Ret E |
|------|-------|---------|---------|-------|-------|-------|
| 1 | sortowanie logów  | 0.1016 | 0.1016 | 1 | 1 | ✓ |
| 2 | wyjątki           | 0.2182 | 0.2182 | 2 | 2 | ✓ |
| 3 | parser moduł      | 0.1743 | 0.1743 | 3 | 3 | ✓ |
| 4 | logging Django    | **0.1803** | 0.1835 | 3 | 4 | ✓ |
| 5 | sortowanie bug    | **0.1814** | 0.1984 | 3 | 5 | ✓ |
| 6 | testy parser      | **0.1838** | 0.2186 | 3 | 5 | ✓ |
| 7 | sortowanie strefy | **0.1932** | 0.2092 | 3 | 5 | ✓ |
| **śr.** | | **0.1761** | 0.1863 | **2.6** | 3.6 | **7/7** |

Pogrubiony drift = EriAmo niższy (lepiej). E = EriAmo, B = Baseline, Ret = retencja kluczowego wzorca.

### Podsumowanie

| Metryka | EriAmo | Baseline |
|---|---|---|
| Średni dryf kontekstu | **0.1761** | 0.1863 |
| Redukcja dryfu | **+5,5%** | — |
| Średni rozmiar okna | **2,6** | 3,6 |
| Redukcja okna | **−28%** | — |
| Retencja wzorca | **7/7** | 7/7 |

EriAmo wygrywa na dryf w 4 z 7 turnów. Turny 1–3 są identyczne — store jest zbyt mały żeby vacuum i recall miały co różnicować. Od turnu 4 EriAmo konsekwentnie lepszy. Pełna retencja wzorca osiągnięta przy 28% niższym koszcie kontekstu.

---

## 7. Otwarte Pytania

**Warstwa attention (empiryczne):** Modyfikacja attention (`K' = K + α·Φ_context`) wymaga dostępu do wnętrza modelu. Powyższe wyniki są na poziomie API. Czy bias holograficzny daje poprawę perplexity porównywalną z Harmonic Attention (~8–9%) — to otwarte pytanie empiryczne wymagające dostępu do wag modelu. Naturalny kolejny krok dla zespołów z pełnym dostępem do modelu (np. SpeakLeash/Bielik).

**Kolaps Φ przy skali:** Przy k=4 wektorach percepcji i lr=0.01, Φ może kolapsować do średniej przy sesjach liczących setki turnów. Może być potrzebny człon momentum lub ograniczenie ortogonalności.

**Kalibracja progu:** Próg vacuum τ=0.3 ustawiony heurystycznie. Optymalna wartość zależy od zadania i powinna być kalibrowana względem precyzji i pełności retencji.

**Kompozytywność z Harmonic Attention:** Bias QRM jest symetryczny względem par tokenów; bias holograficzny jest asymetryczny względem globalnej kotwicy percepcji. Oba to addytywne biasy pre-Softmax bez zmiany architektury — kompozytywność jest bezpośrednim pytaniem badawczym.

---

## 8. Implementacja

Pełna implementacja dostępna. Stack: Python 3.10+, NumPy (czyste — bez zależności Fortran/Rust, działa na Termux ARM), google-genai.

```
core.py      — Item, cosine_sim (czyste numpy), Embedder, init_phi
memory.py    — vacuum_filter, build_context_window
patterns.py  — recall_from_phi, update_patterns
context.py   — update_context (pełny cykl turnu)
embedder.py  — GeminiEmbedder (gemini-embedding-001), factory
session.py   — EriAmoSession (pętla + wywołanie LLM + metryki)
benchmark.py — EriAmo vs baseline okna przesuwnego
```

Szybki start:

```bash
pip install numpy google-genai
export GEMINI_API_KEY='...'
python benchmark.py
```

---

## Literatura

[1] Mazur, M. (2025). Harmonic Attention (QRM Attention Bias). GitHub: Maciej-EriAmo.

[2] Raschka, S. (2025). Odpowiedź na preprinkt Harmonic Attention. Komunikacja osobista.

[3] Vaswani i in. (2017). Attention Is All You Need. NeurIPS.

[4] Anthropic (2023). Steering Vectors for LLM Behavior.

[5] Google (2026). gemini-embedding-001. Dokumentacja Gemini API.
