import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier as SklearnDTC, export_text
    from ISLP import load_data

    return (
        SklearnDTC,
        accuracy_score,
        export_text,
        load_data,
        np,
        pd,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 6: Drzewa decyzyjne — implementacja od podstaw

    Drzewa decyzyjne dzielą przestrzeń cech na rozłączne prostokąty (równoległe do osi) i w każdym dopasowują prosty model — zwykle stałą, choć później poznamy też ciekawsze warianty. Idea jest prosta, modele są elastyczne i — co odróżnia je od tych z poprzednich zajęć — nie zakładają żadnej globalnej postaci funkcyjnej. Są też podstawowym elementem metod składanych (ensemble, czyli lasy losowe, boosting), którym poświęcimy kolejne laby.

    W tym laboratorium natomiast zadaniem jest implementacja klasyfikatora CART (Classification and Regression Trees) od podstaw. Treść jest podzielona na pięć zadań odpowiadających krokom algorytmu:

    1. **Miary nieczystości** węzła (Gini, entropia).
    2. **Najlepszy podział dla pojedynczej cechy** — wybór progu $s$.
    3. **Najlepszy podział po wszystkich cechach** — wybór pary $(j, s)$.
    4. **Rekurencyjna budowa drzewa** — klasy `Node` i `DecisionTreeClassifier`.
    5. **Predykcja** — przejście drzewa dla nowej obserwacji.

    Na końcu porównamy napisaną od podstaw implementację z `DecisionTreeClassifier` z `scikit-learn`.

    Źródło teoretyczne i podstawa do pracy przy tym laboratorium to *The Elements of Statistical Learning*, rozdział 9.2 (Hastie, Tibshirani, Friedman). Książka jest dostępna online pod url: https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CART w pigułce

    CART buduje drzewo zachłannie: w każdym węźle szuka podziału binarnego (cecha + próg), który najmocniej redukuje nieczystość klas w potomkach, a potem powtarza to samo na każdym dziecku. Nieczystość mierzymy indeksem Giniego albo entropią; kryterium podziału minimalizuje ważoną nieczystość dzieci, a rekurencję zatrzymuje warunek stopu (głębokość, liczność, czysty węzeł). Wzory i ich uzasadnienie — ESL 9.2.

    Pełny CART zawiera jeszcze etap przycinania (pruning) z walidacją krzyżową współczynnika kary — w tym labie go pomijamy i sterujemy złożonością drzewa tylko przez warunki stopu.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dane: Carseats

    Używamy zbioru `Carseats` z ISLP — 400 obserwacji sprzedaży fotelików samochodowych w sieci sklepów (dane są symulowane). Zmienna objaśniana `Sales` jest ciągła, ale na potrzeby klasyfikacji przekształcamy ją w binarną etykietę `High = Sales > 8`. Opis zbioru można znaleźć tutaj: https://rdrr.io/cran/ISLR/man/Carseats.html.

    Zmienne kategoryczne (`ShelveLoc`, `Urban`, `US`) zamieniamy na kolumny zerojedynkowe przez `pd.get_dummies()`, bo dla prostoty nasza implementacja obsługuje wyłącznie cechy numeryczne.
    """)
    return


@app.cell
def _(load_data):
    Carseats = load_data('Carseats')
    print(Carseats.info())
    return (Carseats,)


@app.cell
def _(Carseats, pd):
    y_full = (Carseats['Sales'] > 8).astype(int).values
    X_full_df = pd.get_dummies(
        Carseats.drop(columns=['Sales']),
        drop_first=True
    ).astype(float)
    feature_names = list(X_full_df.columns)
    X_full = X_full_df.values
    print(f"Liczba cech po zakodowaniu: {X_full.shape[1]}")
    print(f"Udział klasy pozytywnej (High): {y_full.mean():.2%}")
    return X_full, feature_names, y_full


@app.cell
def _(X_full, train_test_split, y_full):
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
    )
    print(f"Trening: {X_train.shape}, test: {X_test.shape}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 1: Miary nieczystości

    Zaimplementuj dwie miary nieczystości węzła. Wejście to wektor etykiet `y` obserwacji w danym węźle (liczbę klas wyznacz na podstawie `y`).

    - Gini: $\; Q_{\text{Gini}}(y) = \sum_k \hat p_k (1 - \hat p_k) = 1 - \sum_k \hat p_k^2$
    - Entropia: $\; Q_{\text{ent}}(y) = -\sum_k \hat p_k \log_2 \hat p_k$ (dla $\hat p_k = 0$ przyjmujemy $0 \log 0 = 0$)

    Obie miary przyjmują wartość $0$ gdy węzeł jest czysty (wszystkie obserwacje tej samej klasy) a maksymalną przy równomiernym rozkładzie klas.
    """)
    return


@app.cell
def _():
    def gini(y):
        # TODO: zwróć Gini index dla wektora etykiet y
        ...

    def entropy(y):
        # TODO: zwróć entropię (log o podstawie 2) dla wektora etykiet y
        #       pamiętaj, żeby zignorować klasy o zerowej liczności
        ...

    return entropy, gini


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weryfikacja

    Dla `y = [0, 0, 0, 0]` (czysty węzeł) obie miary powinny dać 0.
    Dla `y = [0, 0, 1, 1]` (po równo dwie klasy) oczekujemy Gini = 0.5 oraz entropia = 1.0.
    Dla `y = [0, 1, 2, 3]` (po równo cztery klasy) oczekujemy Gini = 0.75 oraz entropia = 2.0.
    """)
    return


@app.cell
def _(entropy, gini, np):
    for _y in [np.array([0, 0, 0, 0]),
               np.array([0, 0, 1, 1]),
               np.array([0, 1, 2, 3])]:
        print(f"y={_y}  Gini={gini(_y)}  entropy={entropy(_y)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2: Najlepszy podział pojedynczej cechy

    Mając jedną cechę (wektor `x` wartości numerycznych) i etykiety `y`, chcemy znaleźć próg $s$ minimalizujący ważoną nieczystość potomków:

    $$L(s) = \frac{N_L}{N} Q(y_L) + \frac{N_R}{N} Q(y_R)$$

    gdzie $y_L = y[x \le s]$ oraz $y_R = y[x > s]$. Wagi $N_L / N$, $N_R / N$ są równoważne samym $N_L$, $N_R$ przy stałym $N$ — proponuję użycie znormalizowanej wersji, bo jest czytelniejsza i bezpośrednio porównywalna między węzłami o różnej liczności (potrzebne w Zadaniu 4).

    Kandydaci na $s$: zbiór unikalnych wartości `x` jest skończony, więc istnieje skończenie wiele "istotnie różnych" podziałów. Standardowo bierzemy punkty środkowe między kolejnymi unikalnymi wartościami po posortowaniu — to gwarantuje, że każdy próg realnie rozdziela obserwacje. Dla cechy o jednej unikalnej wartości nie ma sensownego podziału — zwróć `None`.

    Funkcja ma zwrócić parę `(best_threshold, best_loss)` albo `(None, np.inf)` jeśli podział jest niemożliwy.
    """)
    return


@app.cell
def _(np):
    def best_split_feature(x, y, impurity_fn):
        """
        x: 1D wektor wartości cechy (float)
        y: 1D wektor etykiet
        impurity_fn: funkcja nieczystości (gini lub entropy)

        Zwraca: (threshold, weighted_impurity). Jeśli split jest niemożliwy
        (np. cecha ma 1 unikalną wartość), zwraca (None, np.inf).
        """
        unique_vals = np.unique(x)
        if unique_vals.size < 2:
            return None, np.inf

        # TODO: wyznacz wektor kandydatów na próg (punkty środkowe między
        #       kolejnymi unikalnymi wartościami x)
        thresholds = ...

        best_threshold = None
        best_loss = np.inf
        n = y.size

        # TODO: dla każdego kandydata:
        #         - podziel y na y_L (x <= s) i y_R (x > s),
        #         - policz ważoną nieczystość L(s) = (N_L/N)*Q(y_L) + (N_R/N)*Q(y_R),
        #         - pomiń podział jeśli którykolwiek potomek jest pusty,
        #         - zaktualizuj best_threshold / best_loss.
        ...

        return best_threshold, best_loss

    return (best_split_feature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weryfikacja

    Dla danych zaprojektowanych tak, że klasa zmienia się w punkcie $x = 5$ (wartości $x \in \{1, \ldots, 9\}$, etykiety `0` dla $x \le 5$ i `1` dla $x > 5$), optymalny próg powinien leżeć między 5 a 6 — czyli s = 5.5, a ważona nieczystość w idealnym podziale wyniesie 0.
    """)
    return


@app.cell
def _(best_split_feature, gini, np):
    _x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    _y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    print(f"best_split_feature (Gini): threshold={best_split_feature(_x, _y, gini)[0]}, "
          f"loss={best_split_feature(_x, _y, gini)[1]:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 3: Najlepszy podział po wszystkich cechach

    Mając macierz cech `X` o kształcie $(N, p)$ iterujemy po kolumnach, dla każdej wywołujemy `best_split_feature`, i wybieramy parę $(j^*, s^*)$ dającą najmniejszą ważoną nieczystość.

    Funkcja ma zwrócić `(best_feature_index, best_threshold, best_loss)` albo `(None, None, np.inf)` jeśli żadna cecha nie pozwala na podział.
    """)
    return


@app.cell
def _(np):
    def best_split(X, y, impurity_fn):
        """
        X: macierz cech (N, p)
        y: etykiety (N,)

        Zwraca: (j*, s*, L*) — indeks cechy, próg i ważoną nieczystość.
        """
        best_feature = None
        best_threshold = None
        best_loss = np.inf

        # TODO: iteruj po kolumnach X, wywołuj best_split_feature
        #       i aktualizuj best_feature / best_threshold / best_loss
        ...

        return best_feature, best_threshold, best_loss

    return (best_split,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weryfikacja

    Konstruujemy zbiór 2D, w którym cecha 1 (druga kolumna) idealnie separuje klasy wokół progu $x_1 = 0$, a cecha 0 jest czystym szumem. Oczekujemy `j* = 1`, próg blisko 0 (nie dokładnie — zależy od konkretnych losowych wartości w okolicy zera), `loss* = 0`.
    """)
    return


@app.cell
def _(best_split, gini, np):
    _rng = np.random.default_rng(0)
    _X = np.column_stack([
        _rng.normal(size=40),
        np.concatenate([_rng.uniform(-2, 0, 20), _rng.uniform(0, 2, 20)])
    ])
    _y = np.array([0] * 20 + [1] * 20)
    _j, _s, _L = best_split(_X, _y, gini)
    print(f"best_split: feature={_j}, threshold={_s:.4f}, loss={_L:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 4: Rekurencyjna budowa drzewa

    Mając funkcję wyboru podziału możemy zbudować całe drzewo. Reprezentacja węzła:

    - węzeł wewnętrzny zawiera `feature` (indeks cechy), `threshold` (próg) oraz referencje do poddrzew `left` (dla $x_j \le s$) i `right` (dla $x_j > s$);
    - liść zawiera `value` — predykcję (klasę większościową) oraz opcjonalnie rozkład klas do diagnostyki.

    Klasa `Node` jest już podana niżej razem ze szkieletem klasy `DecisionTreeClassifier`. W tym zadaniu uzupełnij metodę `_grow_tree`; metodę `_predict_one` zostaw nietkniętą — to zadanie 5.

    Warunki stopu (zamień węzeł na liść):

    - osiągnięto `max_depth`;
    - liczność węzła poniżej `min_samples_split`;
    - węzeł jest czysty (`impurity == 0`);
    - `best_split` nie znalazł sensownego podziału (każda cecha ma jedną wartość).

    W przeciwnym razie znajdujemy najlepszy podział, dzielimy dane i rekurencyjnie budujemy poddrzewa.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 5: Predykcja

    Predykcja dla pojedynczej obserwacji $x$ to przejście drzewa od korzenia w dół: w każdym węźle wewnętrznym sprawdzamy warunek $x_j \le s$ i wchodzimy do `left` lub `right`, aż natrafimy na liść — zwracamy jego `value`.

    Uzupełnij metodę `_predict_one` klasy `DecisionTreeClassifier` poniżej. Metoda `predict`, która iteruje po obserwacjach, jest już gotowa.
    """)
    return


@app.cell
def _(np):
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, n_samples=0, class_counts=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.n_samples = n_samples
            self.class_counts = class_counts

        @property
        def is_leaf(self):
            return self.value is not None


    def majority_class(y):
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)]), dict(zip(values.tolist(), counts.tolist()))


    class DecisionTreeClassifier:
        def __init__(self, impurity_fn, max_depth=5, min_samples_split=2):
            self.impurity_fn = impurity_fn
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.root = None

        def fit(self, X, y):
            self.root = self._grow_tree(np.asarray(X, dtype=float), np.asarray(y), depth=0)
            return self

        def _make_leaf(self, y):
            cls, counts = majority_class(y)
            return Node(value=cls, n_samples=y.size, class_counts=counts)

        def _grow_tree(self, X, y, depth):
            # --- Zadanie 4 ---
            # TODO: 1. Sprawdź warunki stopu i zwróć self._make_leaf(y) jeśli są spełnione:
            #          - depth >= self.max_depth,
            #          - y.size < self.min_samples_split,
            #          - self.impurity_fn(y) == 0.
            ...

            # TODO: 2. Znajdź najlepszy podział przez best_split(X, y, self.impurity_fn).
            #          Jeśli nie ma sensownego podziału (feature is None), zwróć liść.
            ...

            # TODO: 3. Podziel dane na lewe/prawe za pomocą maski x[:, feature] <= threshold,
            #          rekurencyjnie zbuduj poddrzewa i zwróć węzeł wewnętrzny Node(...).
            ...

        def _predict_one(self, node, x):
            # --- Zadanie 5 ---
            # TODO: rekurencyjnie przejdź drzewo aż do liścia i zwróć node.value
            ...

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return np.array([self._predict_one(self.root, xi) for xi in X])

    return (DecisionTreeClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weryfikacja

    Na tych samych syntetycznych danych 2D, co w Zadaniu 3, drzewo o maksymalnej głębokości 3 powinno być bardzo płytkie — już pierwszy podział (cecha 1, próg blisko 0) dzieli dane na dwa czyste liście.

    Funkcja `print_tree` (podana niżej) pozwala obejrzeć strukturę drzewa tekstowo.
    """)
    return


@app.function
def print_tree(node, feature_names=None, depth=0):
    indent = "|   " * depth
    if node.is_leaf:
        print(f"{indent}|--- klasa: {node.value}  (n={node.n_samples}, rozkład={node.class_counts})")
        return
    fname = f"x[{node.feature}]" if feature_names is None else feature_names[node.feature]
    print(f"{indent}|--- {fname} <= {node.threshold:.4f}")
    print_tree(node.left, feature_names, depth + 1)
    print(f"{indent}|--- {fname} >  {node.threshold:.4f}")
    print_tree(node.right, feature_names, depth + 1)


@app.cell
def _(DecisionTreeClassifier, gini, np):
    _rng = np.random.default_rng(0)
    _X = np.column_stack([
        _rng.normal(size=40),
        np.concatenate([_rng.uniform(-2, 0, 20), _rng.uniform(0, 2, 20)])
    ])
    _y = np.array([0] * 20 + [1] * 20)
    _tree = DecisionTreeClassifier(impurity_fn=gini, max_depth=3).fit(_X, _y)
    print_tree(_tree.root)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weryfikacja predykcji

    Dla drzewa zbudowanego na naszych syntetycznych danych 2D predykcje na zbiorze treningowym powinny być dokładnie równe etykietom — drzewo głębokości 3 w zupełności je separuje.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, gini, np):
    _rng = np.random.default_rng(0)
    _X = np.column_stack([
        _rng.normal(size=40),
        np.concatenate([_rng.uniform(-2, 0, 20), _rng.uniform(0, 2, 20)])
    ])
    _y = np.array([0] * 20 + [1] * 20)
    _tree = DecisionTreeClassifier(impurity_fn=gini, max_depth=3).fit(_X, _y)
    _pred = _tree.predict(_X)
    print(f"Dokładność treningowa na danych syntetycznych: {(_pred == _y).mean():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zastosowanie: Carseats

    Trenujemy drzewo na zbiorze `Carseats` (binarna etykieta `High`), z Gini jako miarą nieczystości i `max_depth=3` — żeby drzewo dało się wygodnie obejrzeć i porównać z wersją z `sklearn` przy tych samych hiperparametrach.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    X_test,
    X_train,
    feature_names,
    gini,
    y_test,
    y_train,
):
    our_tree = DecisionTreeClassifier(impurity_fn=gini, max_depth=3).fit(X_train, y_train)
    our_train_acc = (our_tree.predict(X_train) == y_train).mean()
    our_test_acc = (our_tree.predict(X_test) == y_test).mean()
    print(f"Nasze drzewo (Gini, max_depth=3):  train acc = {our_train_acc:.4f}, test acc = {our_test_acc:.4f}\n")
    print("Struktura drzewa:")
    print_tree(our_tree.root, feature_names=feature_names)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Porównanie ze sklearn

    `sklearn.tree.DecisionTreeClassifier` z tymi samymi hiperparametrami powinien dać bardzo zbliżone dopasowanie — drobne różnice mogą wynikać z obsługi remisów przy wyborze podziału, doboru kandydatów na próg albo szczegółów implementacyjnych przy małych węzłach.
    """)
    return


@app.cell
def _(
    SklearnDTC,
    X_test,
    X_train,
    accuracy_score,
    export_text,
    feature_names,
    y_test,
    y_train,
):
    sk_tree = SklearnDTC(criterion='gini', max_depth=3, random_state=0).fit(X_train, y_train)
    sk_train_acc = accuracy_score(y_train, sk_tree.predict(X_train))
    sk_test_acc = accuracy_score(y_test, sk_tree.predict(X_test))
    print(f"sklearn (Gini, max_depth=3):       train acc = {sk_train_acc:.4f}, test acc = {sk_test_acc:.4f}\n")
    print("Struktura drzewa sklearn:")
    print(export_text(sk_tree, feature_names=feature_names))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 6.
    Przełącz miarę nieczystości na `entropy` i zbuduj drzewo na `Carseats` — czy struktura się zmieniła?
    ## Zadanie 7.
    Zwiększ `max_depth` do 10 i porównaj dokładność treningową i testową — co można zaobserwować?
    """)
    return


@app.cell
def _():
    # miejsce na Twoje rozwiązanie
    return


if __name__ == "__main__":
    app.run()
