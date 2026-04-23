# Milestone 2 — Modele boostingowe i walidacja (20 pkt)

## Kontekst

W Milestone 1 zbudowałeś pierwsze rozwiązanie oparte na modelach liniowych i nieliniowych. Teraz rozszerzysz je o metody drzewiaste i boostingowe (XGBoost, LightGBM lub CatBoost) i rzetelnie ocenisz model za pomocą walidacji krzyżowej.

---

## Zadanie

Notebook powinien zawierać kolejno:

### 1. Rozszerzona inżynieria cech

Dodaj co najmniej kilka nowych cech wykraczających poza proste średnie z M1. Mogą to być agregaty per drużyna, różnice między drużynami, cechy pochodne (np. zabójstwa na minutę) lub cokolwiek, co uzasadnisz jako potencjalnie informatywne.

### 2. Modele boostingowe i walidacja krzyżowa

- Wytrenuj podstawowy model drzewiasty i wstępnie zinterpretuj na jego podstawie wpływ cech
- Wytrenuj co najmniej jeden model boostingowy (XGBoost, LightGBM lub CatBoost) i porównaj z wynikiem z M1
- Zastosuj stratyfikowany k-fold (min. 5 foldów) zamiast pojedynczego podziału — podaj wyniki jako średnia ± odchylenie
- Dobierz hiperparametry wybranego modelu (grid search, random search lub inna metoda)

### 3. Skąd pochodzą dane?

Zbiór treningowy pochodzi z API OpenDota — zewnętrznego serwisu agregującego dane o meczach Dota 2. Dane były pobierane automatycznie, partiami, według rang.

**Zbieranie danych odbywało się w kilku sesjach, a w każdej z nich priorytetyzowano różne rangi.**

Zastanów się, jakie konsekwencje ma ten sposób zbierania dla zbioru treningowego. Czy wszystkie kolumny w danych opisują to, co chcesz przewidywać — czyli przebieg rozgrywki — czy może któraś z nich przechowuje informację o randze z zupełnie innego powodu?

### 4. Predykcja i submission

Wygeneruj plik CSV z predykcjami na zbiorze testowym.

---

## Punktacja

| Element | Pkt |
|---------|-----|
| Rozszerzona inżynieria cech z uzasadnieniem | 3 |
| Model + CV + porównanie z M1 | 7 |
| Tuning hiperparametrów | 3 |
| Wynik na leaderboardzie (z korektą za generalizację) | 4 |
| Czytelność notebooka i opisanie kroków | 3 |
| Suma | 20 |

### Progi za wynik na leaderboardzie (2 pkt)

Wynik bazowy na podstawie public accuracy:

| Public accuracy | Pkt |
|-----------------|-----|
| ≥ 60% | 3 |
| 50–59% | 2 |
| 30–49% | 1 |
| < 25% lub brak | 0 |

Korekta za generalizację na podstawie |public − private|, ujawnianego po terminie:

| Różnica | Pkt |
|---------|-----|
| ≤ 8 pp | 1 |
| 9–15 pp | 0 |
| > 15 pp | -1 |

---

## Format oddania

Jak ostatnio - notebook html na upel i submission na Kaggle
