import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 7: Interpretacja drzew, lasy losowe i BART

    Po zaimplementowaniu drzewa skupimy się na trzech zagadnieniach:

    - jak czytać i porównywać miary ważności cech,
    - jak agregacja drzew przez bagging i lasy losowe zmienia jakość i interpretowalność modelu,
    - czym jest BART — bayesowskie złożenie drzew z wbudowaną miarą niepewności.

    Dane:
    - **Boston** z ISLP (regresja — mediana cen domów w tys. USD) - główny zbiór na tych zajęciach. Jego opis i konkretne cechy można znaleźć pod url https://islp.readthedocs.io/en/latest/datasets/Boston.html. Z tego, co widzę, niektóre _problematyczne_ zmienne zostały z niego odfiltrowane.
    - **Carseats** z ISLP (klasyfikacja — czy sprzedaż jest wysoka) powraca do pokazania przestrzeni decyzyjnej
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from ISLP import load_data
    from ISLP.bart import BART

    return (
        BART,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        accuracy_score,
        go,
        load_data,
        mean_squared_error,
        mo,
        np,
        pd,
        permutation_importance,
        plot_tree,
        plt,
        train_test_split,
    )


@app.cell
def _(load_data, pd, train_test_split):
    Boston = load_data('Boston')
    X_bos = Boston.drop(columns=['medv'])
    y_bos = Boston['medv']
    X_bos_tr, X_bos_te, y_bos_tr, y_bos_te = train_test_split(
        X_bos, y_bos, test_size=0.5, random_state=1
    )

    Carseats = load_data('Carseats')
    y_car = (Carseats['Sales'] > 8).astype(int)
    X_car = pd.get_dummies(Carseats.drop(columns=['Sales']), drop_first=True).astype(float)

    X_w = Carseats[['Price', 'Advertising']].values
    y_w = y_car.values
    X_w_tr, X_w_te, y_w_tr, y_w_te = train_test_split(
        X_w, y_w, test_size=0.3, random_state=42, stratify=y_w
    )

    print(f"Boston:   {X_bos_tr.shape[0]} trening / {X_bos_te.shape[0]} test  ({X_bos.shape[1]} cech)")
    print(f"Carseats: {X_car.shape[0]} obserwacji ({X_car.shape[1]} cech po kodowaniu)")
    return (
        X_bos_te,
        X_bos_tr,
        X_w,
        X_w_te,
        X_w_tr,
        y_bos_te,
        y_bos_tr,
        y_w,
        y_w_te,
        y_w_tr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretowalność pojedynczego drzewa

    Drzewo o niskiej głębokości można przeczytać wprost. Na zbiorze Boston budujemy drzewo regresyjne — zmienna objaśniana `medv` to mediana wartości domów w danym rejonie w tys. USD.
    Każdy węzeł zawiera warunek podziału, średni błąd kwadratowy (`squared_error`), czyli alternatywę dla miary czystości węzła wykorzystywana przy regresji, liczbę obserwacji (`samples`) i średnią wartość odpowiedzi (`value`).
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    X_bos_te,
    X_bos_tr,
    mean_squared_error,
    y_bos_te,
    y_bos_tr,
):
    tree_bos_viz = DecisionTreeRegressor(max_depth=3, random_state=0)
    tree_bos_viz.fit(X_bos_tr, y_bos_tr)
    mse_tree_viz = mean_squared_error(y_bos_te, tree_bos_viz.predict(X_bos_te))
    print(f"MSE testowe (drzewo głębokość 3): {mse_tree_viz:.2f}")
    return (tree_bos_viz,)


@app.cell
def _(plot_tree, plt, tree_bos_viz):
    fig_tree, ax_tree = plt.subplots(figsize=(16, 6))
    plot_tree(
        tree_bos_viz, max_depth=4, filled=True,
        feature_names=tree_bos_viz.feature_names_in_,
        fontsize=8, ax=ax_tree,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Można odczytać, że dominująca jest cecha `lstat` (odsetek ludności o niskim statusie). Oceniamy to wizualnie po tym, że występuje w wielu podziałach i szczególnie w tych bliżej korzenia. Natomiast można też odczytać ciekawsze przypadki - przechodzimy od korzenia w lewo do poddrzewa z niższą liczbą osób o niskim statusie. Mamy tam podział oparty o wielkość mieszkań i co ciekawe w lewym poddrzewie z mniejszymi mieszkaniami mamy liść z najwyższą przewidywaną wartością i niskim błędem. Jest tam warunek na dystans od centrum, czyli model znalazł małe apartamenty w najdroższej dzielnicy. Takie przechodzenie przez poziomy jest charakterystyczną cechą interpretowalności drzew - bardzo intuicyjną i pozwalającą oceniać ciekawe, nieliniowe interakcje między zmiennymi, czego np. w regresji brakuje.

    ### Ważność cech — MDI

    Mamy dostępne też liczbowe miary istotności cech. `feature_importances_` (Mean Decrease in Impurity) mierzy średnią redukcję nieczystości przez podziały na danej cesze, ważoną liczbą obserwacji w węźle (czyli mniej więcej to, co zrobiliśmy wizualnie analizując drzewo). Suma po wszystkich cechach wynosi 1. MDI ma tendencję do zawyżania ważności cech ciągłych lub o dużej liczbie unikalnych wartości, bo dają więcej kandydatów na próg podziału i przez to bywają nadreprezentowane.
    """)
    return


@app.cell
def _(go, pd, tree_bos_viz):
    mdi_tree_s = pd.Series(
        tree_bos_viz.feature_importances_, index=tree_bos_viz.feature_names_in_
    ).sort_values(ascending=False)

    fig_mdi_tree = go.Figure(go.Bar(x=mdi_tree_s.index, y=mdi_tree_s.values))
    fig_mdi_tree.update_layout(
        title='MDI — drzewo regresyjne głębokość 3 (Boston)',
        xaxis_title='Cecha', yaxis_title='MDI',
    )
    fig_mdi_tree.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bagging i lasy losowe

    ### Baggingu — próbkowanie bootstrap

    **Bagging** (bootstrap aggregating) trenuje $B$ drzew, każde na osobnej próbce bootstrap rozmiaru $n$ losowanej ze zwracaniem z oryginalnych $n$ obserwacji. Część obserwacji pojawi się wielokrotnie, część nie pojawi się wcale — te ostatnie tworzą zbiór *out-of-bag* (OOB) służący do oceny modelu.
    Predykcje końcowe to średnia po wszystkich drzewach.
    """)
    return


@app.cell
def _(go, np):
    rng_demo = np.random.default_rng(1)
    n_demo = 15

    fig_boot = go.Figure()
    for _b in range(3):
        _sample = rng_demo.choice(n_demo, size=n_demo, replace=True)
        _counts = np.bincount(_sample, minlength=n_demo)
        fig_boot.add_bar(
            x=list(range(n_demo)), y=_counts,
            name=f'Bootstrap {_b + 1}', offsetgroup=_b,
        )
    fig_boot.update_layout(
        barmode='group',
        title='Wystąpienia każdej obserwacji w trzech próbkach bootstrap (n=15)',
        xaxis=dict(title='Indeks obserwacji', tickmode='linear', dtick=1),
        yaxis_title='Liczba wystąpień',
    )
    fig_boot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Obserwacje z liczbą wystąpień = 0 (tutaj obserwacja 10) to OOB dla danego drzewa. Uśrednianie predykcji OOB po wszystkich drzewach daje bezstronną ocenę błędu generalizacji bez oddzielnego zbioru testowego.

    **Las losowy** dekoreluje drzewa losując przy każdym podziale węzła tylko podzbiór $m$ cech. Dla regresji typowo $m = \lfloor p/3 \rfloor$, dla klasyfikacji $m = \lfloor\sqrt{p}\rfloor$. Bagging to szczególny przypadek z $m = p$.
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    RandomForestRegressor,
    X_bos_te,
    X_bos_tr,
    mean_squared_error,
    y_bos_te,
    y_bos_tr,
):
    n_p = X_bos_tr.shape[1]

    tree_bos = DecisionTreeRegressor(random_state=0)
    tree_bos.fit(X_bos_tr, y_bos_tr)
    mse_tree = mean_squared_error(y_bos_te, tree_bos.predict(X_bos_te))

    bag_bos = RandomForestRegressor(
        n_estimators=500, max_features=n_p, oob_score=True, random_state=1
    )
    bag_bos.fit(X_bos_tr, y_bos_tr)
    mse_bag = mean_squared_error(y_bos_te, bag_bos.predict(X_bos_te))

    rf_bos = RandomForestRegressor(
        n_estimators=500, max_features='sqrt', oob_score=True, random_state=1
    )
    rf_bos.fit(X_bos_tr, y_bos_tr)
    mse_rf = mean_squared_error(y_bos_te, rf_bos.predict(X_bos_te))

    print(f"MSE testowe:")
    print(f"  Pojedyncze drzewo : {mse_tree:.2f}")
    print(f"  Bagging (m=p)     : {mse_bag:.2f}")
    print(f"  Las losowy (m=√p) : {mse_rf:.2f}")
    print(f"\nOOB R²: bagging {bag_bos.oob_score_:.4f}  las losowy {rf_bos.oob_score_:.4f}")
    return mse_bag, mse_rf, mse_tree, rf_bos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Z dokumentacji wynika, że domyślną opcją w sklearn dla lasów losowych jest zwykły bagging, który co ciekawe poradził sobie na Bostonie lepiej:
    ```
        max_features : {"sqrt", "log2", None}, int or float, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and
          max(1, int(max_features * n_features_in_)) features are considered at each
          split.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None or 1.0, then max_features=n_features.
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MDI a Permutation Importance

    MDI jest liczone na danych uczących i jest stronnicze względem cech ciągłych. **Permutation Importance** (PI) mierzy, o ile pogarsza się wynik na zbiorze **testowym** po losowym przetasowaniu wartości danej cechy — jest wolniejsze, ale bezpośrednio odzwierciedla wkład cechy w zdolność generalizacji.
    Oba rankingi znormalizowano do sumy 1 dla porównania na jednym wykresie.
    """)
    return


@app.cell
def _(X_bos_te, X_bos_tr, go, pd, permutation_importance, rf_bos, y_bos_te):
    perm_res = permutation_importance(
        rf_bos, X_bos_te, y_bos_te, n_repeats=10, random_state=0
    )

    feat_names = X_bos_tr.columns.tolist()
    mdi_rf_s = pd.Series(rf_bos.feature_importances_, index=feat_names)
    pi_rf_s = pd.Series(perm_res.importances_mean, index=feat_names)

    order = mdi_rf_s.sort_values(ascending=False).index
    mdi_norm = mdi_rf_s[order] / mdi_rf_s.sum()
    pi_norm = pi_rf_s[order] / pi_rf_s.sum()

    fig_imp = go.Figure()
    fig_imp.add_bar(x=order, y=mdi_norm.values, name='MDI', offsetgroup=0)
    fig_imp.add_bar(x=order, y=pi_norm.values, name='Permutation', offsetgroup=1)
    fig_imp.update_layout(
        barmode='group',
        title='MDI vs Permutation Importance — las losowy (Boston)',
        xaxis_title='Cecha', yaxis_title='Ważność (znormalizowana)',
    )
    fig_imp.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Przestrzeń decyzyjna lasu losowego

    Model klasyfikuje obserwacje z Carseats używając tylko dwóch cech — `Price` i `Advertising` — co pozwala narysować przestrzeń decyzyjną w 2D. Suwaki sterują liczbą drzew i głębokością. Jak zmienia się granica decyzyjna i dokładność train/test? Czy da się zaobserwować przeuczenie (generalnie bias-variance tradeoff)?
    """)
    return


@app.cell
def _(mo):
    sl_trees = mo.ui.slider(1, 100, value=50, step=10, label="Liczba drzew", show_value=True)
    sl_depth = mo.ui.slider(1, 15, value=5, step=2, label="Głębokość", show_value=True)
    mo.vstack([sl_trees, sl_depth])
    return sl_depth, sl_trees


@app.cell
def _(
    RandomForestClassifier,
    X_w,
    X_w_te,
    X_w_tr,
    accuracy_score,
    go,
    np,
    sl_depth,
    sl_trees,
    y_w,
    y_w_te,
    y_w_tr,
):
    rf_w = RandomForestClassifier(
        n_estimators=sl_trees.value, max_depth=sl_depth.value, random_state=0
    )
    rf_w.fit(X_w_tr, y_w_tr)
    acc_tr = accuracy_score(y_w_tr, rf_w.predict(X_w_tr))
    acc_te = accuracy_score(y_w_te, rf_w.predict(X_w_te))

    px_lin = np.linspace(X_w[:, 0].min() - 5, X_w[:, 0].max() + 5, 200)
    adv_lin = np.linspace(X_w[:, 1].min() - 1, X_w[:, 1].max() + 1, 200)
    xx, yy = np.meshgrid(px_lin, adv_lin)
    Z = rf_w.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig_w = go.Figure()
    fig_w.add_heatmap(
        x=px_lin, y=adv_lin, z=Z,
        colorscale=[[0, 'rgba(100,149,237,0.4)'], [1, 'rgba(250,128,114,0.4)']],
        showscale=False,
    )
    for _cls, _name, _col in [(0, 'Low', 'royalblue'), (1, 'High', 'tomato')]:
        _mask = y_w == _cls
        fig_w.add_scatter(
            x=X_w[_mask, 0], y=X_w[_mask, 1],
            mode='markers',
            marker=dict(color=_col, size=6, opacity=0.75,
                        line=dict(width=0.5, color='white')),
            name=_name,
        )
    fig_w.update_layout(
        title=(f"Drzewa: {sl_trees.value}, głębokość: {sl_depth.value}"
               f"  |  Train acc: {acc_tr:.3f}  Test acc: {acc_te:.3f}"),
        xaxis_title='Price', yaxis_title='Advertising',
        legend=dict(x=0.01, y=0.99),
    )
    fig_w.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BART — Bayesian Additive Regression Trees

    BART reprezentuje funkcję regresji jako sumę $B$ małych drzew: $f(x) = \sum_{b=1}^{B} T_b(x;\Theta_b)$. Każde drzewo wyjaśnia tylko fragment sygnału — idea zbliżona do lasu losowego. Kluczowa różnica leży w sposobie trenowania: zamiast minimalizacji funkcji straty stosujemy **MCMC**.

    **MCMC** (Markov Chain Monte Carlo) to klasa algorytmów próbkowania z rozkładów, z których nie można losować wprost. W BART generator losowy naprzemiennie aktualizuje każde z $B$ drzew, trzymając pozostałe ustalone — po każdej aktualizacji rejestrujemy bieżącą predykcję jako próbkę z posterioru. Pierwsze `burnin` kroków to jest odrzucanych jako faza rozgrzewkowa. Kolejne `ndraw` kroków tworzy aproksymację posterioru. Zamiast jednej predykcji punktowej dostajemy rozkład predykcji — możemy bezpośrednio odczytać niepewność modelu.

    Implementacja pochodzi z `ISLP.bart`.
    """)
    return


@app.cell
def _(BART, X_bos_te, X_bos_tr, mean_squared_error, y_bos_te, y_bos_tr):
    bart_bos = BART(num_trees=200, ndraw=100, burnin=100, random_state=0)
    bart_bos.fit(X_bos_tr, y_bos_tr)
    mse_bart = mean_squared_error(y_bos_te, bart_bos.predict(X_bos_te))
    print(f"BART MSE testowe: {mse_bart:.2f}")
    return bart_bos, mse_bart


@app.cell
def _(go, mse_bag, mse_bart, mse_rf, mse_tree):
    labels_cmp = ['Drzewo', 'Bagging', 'Las losowy', 'BART']
    vals_cmp = [mse_tree, mse_bag, mse_rf, mse_bart]
    fig_cmp = go.Figure(go.Bar(
        x=labels_cmp, y=vals_cmp,
        text=[f'{v:.2f}' for v in vals_cmp], textposition='outside',
    ))
    fig_cmp.update_layout(
        title='MSE testowe — porównanie modeli (Boston)',
        yaxis_title='MSE', yaxis=dict(range=[0, max(vals_cmp) * 1.2]),
    )
    fig_cmp.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Niepewność predykcji

    Każde wywołanie `bart_bos.staged_predict(X)` zwraca generator kolejnych próbek z posterioru — po zebraniu ich do macierzy kształtu `(ndraw, n_test)` możemy policzyć percentyle i uzyskać bayesowskie przedziały ufności. Poniżej widać 40 losowych obserwacji testowych posortowanych rosnąco wg predykowanej wartości: niebieskie słupki błędów to 95% CI, czerwone krzyże to prawdziwe wartości.
    """)
    return


@app.cell
def _(X_bos_te, bart_bos, go, np, y_bos_te):
    _samples = np.array(list(bart_bos.staged_predict(X_bos_te)))
    _lower = np.percentile(_samples, 2.5, axis=0)
    _upper = np.percentile(_samples, 97.5, axis=0)
    _mean = _samples.mean(axis=0)

    rng_vis = np.random.default_rng(7)
    idx_vis = rng_vis.choice(len(y_bos_te), size=40, replace=False)
    idx_vis = idx_vis[np.argsort(_mean[idx_vis])]
    pos = np.arange(40)

    fig_unc = go.Figure()
    fig_unc.add_scatter(
        x=pos, y=_mean[idx_vis],
        error_y=dict(
            type='data',
            array=(_upper - _mean)[idx_vis],
            arrayminus=(_mean - _lower)[idx_vis],
        ),
        mode='markers', name='BART: predykcja + 95% CI',
        marker=dict(color='steelblue', size=7),
    )
    fig_unc.add_scatter(
        x=pos, y=y_bos_te.iloc[idx_vis].values,
        mode='markers', name='Prawdziwa wartość',
        marker=dict(color='tomato', size=8, symbol='x-thin', line=dict(width=2)),
    )
    fig_unc.update_layout(
        title='Posterior predykcji BART — 95% przedziały ufności (Boston, podzbiór testowy)',
        xaxis_title='Obserwacja (posortowana wg predykcji)',
        yaxis_title='medv (tys. USD)',
    )
    fig_unc.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variable inclusion proportions

    BART śledzi, które cechy pojawiają się jako podziały w poszczególnych próbkach MCMC. `variable_inclusion_` to macierz kształtu `(ndraw, p)` — po uśrednieniu wierszy dostajemy proporcję próbek, w których dana cecha była użyta do podziału. Porównaj ranking z MDI i Permutation Importance z sekcji o lasach losowych.
    """)
    return


@app.cell
def _(X_bos_tr, bart_bos, go, pd):
    incl = pd.Series(
        bart_bos.variable_inclusion_.mean(0),
        index=X_bos_tr.columns,
    ).sort_values(ascending=False)
    incl_norm = incl / incl.sum()

    fig_incl = go.Figure(go.Bar(x=incl_norm.index, y=incl_norm.values))
    fig_incl.update_layout(
        title='Variable inclusion proportions — BART (Boston)',
        xaxis_title='Cecha', yaxis_title='Proporcja włączeń (znorm.)',
    )
    fig_incl.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Zadania

    ## Zadanie 1 — OOB jako przybliżenie błędu testowego

    Wytrenuj las losowy z `oob_score=True` na Boston dla `n_estimators` ∈ {1, 5, 10, 20, 50, 100, 200, 500}. Narysuj OOB R² i testowy R² (z `r2_score`) w funkcji liczby drzew na jednym wykresie. Przy ilu drzewach błąd OOB stabilizuje się i czy dobrze przybliża wynik testowy? Użyj `max_features='sqrt'` i `random_state=1`.
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2 — wpływ `max_features`

    **a)** Wytrenuj las losowy (`n_estimators=200`) na Boston dla `max_features`
    ∈ {1, 2, 3, 4, 6, 8, 10, 12}, gdzie 12 to bagging. Narysuj testowe MSE.
    Czy tak zwany _knee curve_ jest widoczny?

    **b)** Dla tych samych wartości `max_features` wytrenuj lasy po 50 drzew i oblicz
    średnią korelację Pearsona między predykcjami poszczególnych drzew na zbiorze
    testowym (`model.estimators_`; przed predykcją skonwertuj X do numpy). Narysuj
    krzywe MSE z a) i średniej korelacji obok siebie. Jak korelacja zmienia się
    z `max_features` i jak tłumaczy to kształt krzywej MSE?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej — podpunkt a)
    ...
    return


@app.cell
def _():
    # Uzupełnij kod poniżej — podpunkt b)
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 3 — MDI: eksperyment z szumem

    Do zbioru Boston dodaj 5 kolumn z czystym szumem (rozkład normalny) i wytrenuj las losowy. Porównaj MDI i Permutation Importance — czy MDI zawyża ważność szumu względem PI? Poniżej gotowa podstawa do uzupełnienia.
    """)
    return


@app.cell
def _(RandomForestRegressor, X_bos_te, X_bos_tr, np, y_bos_tr):
    _rng = np.random.default_rng(42)
    _noise_cols = [f'noise_{i}' for i in range(5)]
    _X_tr_n = X_bos_tr.copy()
    _X_te_n = X_bos_te.copy()
    for _c in _noise_cols:
        _X_tr_n[_c] = _rng.normal(size=len(X_bos_tr))
        _X_te_n[_c] = _rng.normal(size=len(X_bos_te))

    _rf_n = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=1)
    _rf_n.fit(_X_tr_n, y_bos_tr)

    # Uzupełnij: oblicz MDI i Permutation Importance, narysuj wykres słupkowy grupowany
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 4 — BART: ważność cech i analiza niepewności

    **a)** Narysuj na jednym wykresie słupkowym grupowanym trzy rankingi ważności cech: MDI (`mdi_rf_s`), Permutation Importance (`pi_rf_s`) i variable inclusion proportions (`incl_norm`) z BART — wszystkie znormalizowane do sumy 1. Przy których cechach rankingi najbardziej się różnią i jak to zinterpretować?

    **b)** Obserwacje z szerokim 95% CI mają wysoką niepewność predykcji. Wyznacz szerokość CI (`upper - lower`) dla wszystkich obserwacji testowych (wywołaj `bart_bos.staged_predict(X_bos_te)` ponownie). Narysuj wykres rozrzutu `lstat` × `rm` dla zbioru testowego, kodując kolor punktu szerokością CI. Czy obserwacje z szerokim CI koncentrują się na krańcach rozkładu cech lub poza gęsto próbkowanym obszarem treningowym?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
