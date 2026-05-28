import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 9: Modele graficzne — sieci bayesowskie, pola Markowa, grafy czynników

    Pełen rozkład łączny zmiennych dyskretnych rośnie wykładniczo wraz z liczbą zmiennych. Modele graficzne dostarczają kompaktowej reprezentacji rozkładu łącznego (jako iloczynu lokalnych czynników) oraz języka wnioskowania o niezależnościach warunkowych w oparciu o własności topologiczne grafu.

    Trzy reprezentacje, którymi zajmujemy się w tym laboratorium:

    - **Sieci bayesowskie** (skierowane) — naturalne, gdy istnieje porządek przyczynowy lub czasowy.
    - **Pola losowe Markowa** (MRF, nieskierowane) — naturalne przy zależnościach symetrycznych (piksele obrazu, atomy w sieci krystalicznej).
    - **Grafy czynników** (bipartytowe) — uogólnienie obu poprzednich reprezentacji, wygodne przy algorytmach przekazywania komunikatów.

    Podstawową lekturą jest podręcznik *Probabilistic Graphical Models: Principles and Techniques* Daphne Koller i Nira Friedmana (MIT Press, 2009): http://mcb111.org/w06/KollerFriedman.pdf. Rozdziały 3, 4 i 11 pokrywają — odpowiednio — sieci bayesowskie, pola Markowa i wnioskowanie dokładne. W tym laboratorium ograniczamy się do ujęcia intuicyjnego, a po formalne wyprowadzenia odsyłamy do podręcznika.

    Narzędzia:

    - **pgmpy** — https://pgmpy.org
    - **networkx** — https://networkx.org

    Zbiór: ten sam **Adult** z UCI, którego używaliśmy w lab8.
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='pgmpy')

    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, TreeSearch
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference import VariableElimination, BeliefPropagation

    return (
        BeliefPropagation,
        DiscreteBayesianNetwork,
        HillClimbSearch,
        TreeSearch,
        VariableElimination,
        mo,
        np,
        nx,
        pd,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dane: Adult (UCI)

    Wczytujemy zbiór bezpośrednio z UCI, wybieramy osiem kolumn tworzących reprezentatywny przekrój "demografia → praca → dochód" i sprowadzamy wszystkie zmienne do dyskretnych etykiet:

    - `age` w trzech przedziałach: Young (<30), Mid (30–50), Senior (>50)
    - `hours` w trzech przedziałach: Part (<35), Full (35–45), Over (>45)
    - `education` skonsolidowane do pięciu poziomów (No-HS, HS, College, Bachelors, Advanced)
    - `marital` skonsolidowane do trzech kategorii (Married, Single, Other)
    - `occupation` skonsolidowane do czterech grup (White-collar, Blue-collar, Service, Other)
    - `relationship`, `sex` zostają w oryginale
    - `income` jako binarne (low/high)

    Konsolidacje są arbitralne, ale uzasadniają je dwa cele: ograniczenie rozmiaru tablic CPD (które rosną iloczynowo z kardynalnością rodziców) oraz czytelność grafu po nauczeniu struktury.
    """)
    return


@app.cell
def _(pd):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
    ]
    raw = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

    education_map = {
        'Preschool': 'No-HS', '1st-4th': 'No-HS', '5th-6th': 'No-HS', '7th-8th': 'No-HS',
        '9th': 'No-HS', '10th': 'No-HS', '11th': 'No-HS', '12th': 'No-HS',
        'HS-grad': 'HS',
        'Some-college': 'College', 'Assoc-voc': 'College', 'Assoc-acdm': 'College',
        'Bachelors': 'Bachelors',
        'Masters': 'Advanced', 'Prof-school': 'Advanced', 'Doctorate': 'Advanced',
    }
    marital_map = {
        'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
        'Married-spouse-absent': 'Other', 'Separated': 'Other',
        'Divorced': 'Other', 'Widowed': 'Other',
        'Never-married': 'Single',
    }
    occupation_map = {
        'Exec-managerial': 'White-collar', 'Prof-specialty': 'White-collar',
        'Tech-support': 'White-collar', 'Adm-clerical': 'White-collar', 'Sales': 'White-collar',
        'Craft-repair': 'Blue-collar', 'Machine-op-inspct': 'Blue-collar',
        'Transport-moving': 'Blue-collar', 'Handlers-cleaners': 'Blue-collar',
        'Farming-fishing': 'Blue-collar',
        'Other-service': 'Service', 'Priv-house-serv': 'Service', 'Protective-serv': 'Service',
        'Armed-Forces': 'Other',
    }

    adult = raw.dropna(subset=['occupation', 'workclass']).copy()
    adult = adult[['age', 'education', 'marital-status', 'occupation',
                   'relationship', 'sex', 'hours-per-week', 'income']]
    adult['age'] = pd.cut(adult['age'], bins=[0, 30, 50, 100],
                          include_lowest=True, labels=['Young', 'Mid', 'Senior']).astype(str)
    adult['hours'] = pd.cut(adult['hours-per-week'], bins=[0, 35, 45, 100],
                            include_lowest=True, labels=['Part', 'Full', 'Over']).astype(str)
    adult['education'] = adult['education'].map(education_map)
    adult['marital'] = adult['marital-status'].map(marital_map)
    adult['occupation'] = adult['occupation'].map(occupation_map)
    adult['income'] = adult['income'].map({'<=50K': 'low', '>50K': 'high'})
    adult = adult.drop(columns=['marital-status', 'hours-per-week'])
    adult = adult[['age', 'education', 'marital', 'occupation', 'relationship', 'sex', 'hours', 'income']]
    adult = adult.dropna()

    print(f"{adult.shape[0]} obserwacji, {adult.shape[1]} zmiennych")
    print("\nLiczba poziomów na zmienną:")
    for col in adult.columns:
        print(f"  {col:14s} {adult[col].nunique()}  ({sorted(adult[col].unique())})")
    return (adult,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sieci bayesowskie

    Sieć bayesowska to skierowany graf acykliczny, w którym każdy węzeł reprezentuje zmienną losową, a każda strzałka — bezpośrednią zależność warunkową. Rozkład łączny faktoryzuje się na iloczyn lokalnych rozkładów warunkowych „zmienna pod warunkiem swoich rodziców":

    $$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid \mathrm{Pa}(X_i)).$$

    Każdemu węzłowi $X_i$ odpowiada **tablica rozkładów warunkowych** (CPD). Im więcej rodziców ma węzeł, tym większa tablica — stąd motywacja, by struktura sieci była rzadka.

    ## Uczenie struktury — Hill Climbing

    Pełna enumeracja grafów acyklicznych jest niewykonalna nawet dla niewielu węzłów, więc uczenie struktury realizujemy heurystycznie — przeszukujemy przestrzeń grafów lokalnymi modyfikacjami (dodaj, usuń lub odwróć krawędź), kierując się funkcją oceniającą karzącą za złożoność. Standardowym wyborem jest poznane wcześniej **BIC** (Bayesian Information Criterion).
    """)
    return


@app.cell
def _(HillClimbSearch, adult, time):
    t0 = time.perf_counter()
    hc = HillClimbSearch(adult)
    bn_struct_bic = hc.estimate(scoring_method='bic-d', show_progress=False)
    t_bic = time.perf_counter() - t0

    print(f"HC + BIC: {t_bic:.1f}s, {len(list(bn_struct_bic.edges()))} krawędzi")
    print("\nKrawędzie nauczonej sieci:")
    for src, dst in bn_struct_bic.edges():
        print(f"  {src} -> {dst}")
    return bn_struct_bic, hc


@app.cell
def _(adult, bn_struct_bic, nx, plt):
    G_bn = nx.DiGraph()
    G_bn.add_nodes_from(adult.columns.tolist())
    G_bn.add_edges_from(bn_struct_bic.edges())

    pos_bn = nx.spring_layout(G_bn, seed=42, k=1.6)
    fig_bn, ax_bn = plt.subplots(figsize=(10, 7))
    nx.draw(G_bn, pos_bn, with_labels=True, node_color='#fcb1a6', node_size=2400,
            font_size=11, arrows=True, arrowsize=22, edge_color='gray',
            connectionstyle='arc3,rad=0.05', ax=ax_bn)
    ax_bn.set_title("Sieć bayesowska — Adult (HC + BIC)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Niektóre kierunki krawędzi mogą wydawać się odwrócone wobec intuicji (np. `education -> age` zamiast `age -> education`). HC z BIC nie jest w stanie odróżnić struktur z **tej samej klasy równoważności Markowa** — pary $A\to B$ i $A\leftarrow B$ generują ten sam zbiór niezależności warunkowych, gdy nie tworzą one v-struktury z trzecim węzłem. Rozróżnienie wymagałoby albo wiedzy domenowej, albo eksperymentów interwencyjnych (Koller & Friedman, rozdz. 3.4).

    ## Inna funkcja oceniająca — K2

    BIC nie jest jedyną sensowną funkcją oceny. **K2** (Cooper & Herskovits, 1992) to bayesowska funkcja oceniająca. W przeciwieństwie do BIC nie zawiera jawnego członu kary za złożoność, przez co zwykle dopuszcza nieco bogatsze grafy.
    """)
    return


@app.cell
def _(bn_struct_bic, hc):
    bn_struct_k2 = hc.estimate(scoring_method='k2', show_progress=False)
    edges_bic = set(bn_struct_bic.edges())
    edges_k2 = set(bn_struct_k2.edges())

    print(f"Liczba krawędzi: BIC = {len(edges_bic)}, K2 = {len(edges_k2)}")
    print("\nKrawędzie tylko w K2 (BIC ich nie wybrał):")
    for e in sorted(edges_k2 - edges_bic):
        print(f"  {e[0]} -> {e[1]}")
    print("\nKrawędzie tylko w BIC:")
    for e in sorted(edges_bic - edges_k2):
        print(f"  {e[0]} -> {e[1]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dopasowanie parametrów

    Mając strukturę, parametry lokalnych CPD estymujemy metodą największej wiarygodności — czyli relatywnymi częstościami w danych. `bn.fit(df)` korzysta z MLE domyślnie.
    """)
    return


@app.cell
def _(DiscreteBayesianNetwork, adult, bn_struct_bic):
    bn_model = DiscreteBayesianNetwork(bn_struct_bic.edges())
    bn_model.add_nodes_from(adult.columns.tolist())
    bn_model.fit(adult)

    print("CPD dla węzła 'income':")
    print(bn_model.get_cpds('income'))
    return (bn_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wizualizacja CPD

    Tabela liczb jest mało czytelna. Ten sam rozkład warunkowy łatwiej odczytać z wykresu — dla każdej kombinacji wartości rodziców pokazujemy słupkowy rozkład prawdopodobieństwa zmiennej. Aby wykres pozostał czytelny niezależnie od konkretnej nauczonej struktury, wybieramy węzeł z najmniejszą liczbą kombinacji rodziców.
    """)
    return


@app.cell
def _(bn_model, np, plt):
    from itertools import product

    def _combo_count(node):
        cpd = bn_model.get_cpds(node)
        parents = cpd.variables[1:]
        if not parents:
            return float('inf')
        total = 1
        for p in parents:
            total *= len(cpd.state_names[p])
        return total

    viz_node = min(bn_model.nodes(), key=_combo_count)
    cpd_viz = bn_model.get_cpds(viz_node)
    parents_viz = cpd_viz.variables[1:]
    target_states = cpd_viz.state_names[viz_node]
    parent_states = [cpd_viz.state_names[p] for p in parents_viz]

    values_viz = cpd_viz.values.reshape(len(target_states), -1)
    combos = list(product(*parent_states))
    labels_viz = ['\n'.join(c) for c in combos]

    x = np.arange(len(combos))
    width = 0.8 / len(target_states)
    fig_cpd, ax_cpd = plt.subplots(figsize=(max(8, 1.2 * len(combos)), 4.5))
    for _i, _st in enumerate(target_states):
        ax_cpd.bar(x + (_i - (len(target_states) - 1) / 2) * width,
                   values_viz[_i], width, label=f"{viz_node}={_st}")
    ax_cpd.set_xticks(x)
    ax_cpd.set_xticklabels(labels_viz, fontsize=9)
    ax_cpd.set_ylabel(f"P({viz_node} | rodzice)")
    ax_cpd.set_title(f"CPD węzła '{viz_node}' (rodzice: {', '.join(parents_viz)})")
    ax_cpd.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Otoczka Markowa

    Otoczka Markowa zmiennej $X$ to najmniejszy zbiór węzłów taki, że pod warunkiem otoczki $X$ jest niezależna od reszty grafu. W sieci bayesowskiej składa się z rodziców $X$, dzieci $X$ oraz pozostałych rodziców dzieci $X$. Znając wartości otoczki, dane spoza niej nie wnoszą nic nowego do predykcji $X$ — klasyfikator korzystający wyłącznie ze zmiennych otoczki Markowa zmiennej celu powinien osiągać dokładność porównywalną z klasyfikatorem korzystającym ze wszystkich predyktorów.
    """)
    return


@app.cell
def _(bn_model):
    mb_income = bn_model.get_markov_blanket('income')
    print(f"Otoczka Markowa zmiennej 'income': {sorted(mb_income)}")

    print("\nOtoczki Markowa pozostałych zmiennych:")
    for var in ['age', 'education', 'occupation', 'sex']:
        print(f"  {var:12s} {sorted(bn_model.get_markov_blanket(var))}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wnioskowanie — eliminacja zmiennych

    Mając kilka zmiennych w roli dowodów (`evidence`), chcemy wyznaczyć rozkład warunkowy interesującej nas zmiennej. Algorytm **eliminacji zmiennych** wyznacza go dokładnie, sumując kolejno zmienne spoza zapytania i dowodów. Kolejność eliminacji wpływa na koszt obliczeń.
    """)
    return


@app.cell
def _(VariableElimination, bn_model):
    inf = VariableElimination(bn_model)

    q1 = inf.query(['income'],
                   evidence={'sex': 'Female', 'education': 'Advanced'},
                   show_progress=False)
    print("P(income | Female, Advanced):")
    print(q1)

    q2 = inf.query(['income'],
                   evidence={'sex': 'Male', 'education': 'No-HS', 'hours': 'Part'},
                   show_progress=False)
    print("\nP(income | Male, No-HS, Part-time):")
    print(q2)

    q3 = inf.query(['occupation'],
                   evidence={'income': 'high', 'education': 'Bachelors'},
                   show_progress=False)
    print("\nP(occupation | high income, Bachelors):")
    print(q3)
    return (inf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Próbkowanie z rozkładu

    **a)** Wyświetl CPD dla wybranego węzła z większą liczbą rodziców niż `income` (np. dla `marital`, jeśli ma wielu rodziców w nauczonej strukturze; jeśli nie — wybierz inny). Ile parametrów liczy ta tablica? Porównaj z liczbą wpisów w pełnej tablicy łącznej dla tych samych zmiennych — ile rzędów wielkości oszczędności daje faktoryzacja?

    **b)** Sprawdź własność otoczki Markowa empirycznie: dla zapytania `P(income | evidence)` porównaj wynik dla `evidence` zawierającego wyłącznie zmienne z otoczki Markowa `income` względem `evidence` z dodatkowymi zmiennymi spoza otoczki. Czy odpowiedzi są identyczne (tylko gdy faktycznie obserwujesz wszystkie zmienne otoczki)?
    """)
    return


@app.cell
def _(bn_model, inf, np):


    income_parents = bn_model.get_parents('income')
    print(f"Liczba rodziców 'income': {len(income_parents)}  ({income_parents})")


    parents_count = {n: len(bn_model.get_parents(n))
                     for n in bn_model.nodes() if n != 'income'}
    target_node = max(parents_count, key=lambda n: parents_count[n])
    target_parents = bn_model.get_parents(target_node)

    print(f"\nWybrany węzeł: '{target_node}'  (rodziców: {len(target_parents)}: {target_parents})")

    cpd = bn_model.get_cpds(target_node)
    print(f"\nCPD dla '{target_node}':")
    print(cpd)


    node_card = len(cpd.state_names[target_node])
    parent_cards = [len(cpd.state_names[p]) for p in target_parents]
    parent_prod = int(np.prod(parent_cards)) if parent_cards else 1

    cpd_params = (node_card - 1) * parent_prod
    joint_entries = node_card * parent_prod
    joint_params = joint_entries - 1

    print(f"\nParametry CPD:              ({node_card}-1) × {' × '.join(map(str, parent_cards))} = {cpd_params}")
    print(f"Wpisy pełnej tablicy łącznej: {node_card} × {' × '.join(map(str, parent_cards))} = {joint_entries}")
    print(f"Stosunek (joint / CPD)      ≈ {joint_entries / cpd_params:.2f}×  "
          f"| oszczędność ~{np.log10(joint_entries / cpd_params):.2f} rzędu wielkości")


    print("\n" + "="*60)
    print("b) Własność otoczki Markowa — weryfikacja empiryczna")
    print("="*60)

    mb = bn_model.get_markov_blanket('income')
    all_non_income = [v for v in bn_model.nodes() if v != 'income']
    outside_mb = [v for v in all_non_income if v not in mb]

    print(f"\nOtoczka Markowa 'income': {sorted(mb)}")
    print(f"Zmienne spoza otoczki:    {sorted(outside_mb)}")


    evidence_mb = {v: bn_model.get_cpds(v).state_names[v][0] for v in sorted(mb)}


    extra = outside_mb[:2] if outside_mb else []
    evidence_full = {**evidence_mb,
                     **{v: bn_model.get_cpds(v).state_names[v][0] for v in extra}}

    q_mb   = inf.query(['income'], evidence=evidence_mb,   show_progress=False)
    q_full = inf.query(['income'], evidence=evidence_full, show_progress=False)

    print(f"\nEvidence — tylko otoczka Markowa: {evidence_mb}")
    print(q_mb)
    print(f"\nEvidence — otoczka + spoza ({extra}): {evidence_full}")
    print(q_full)
def _(bn_model):
    sample = bn_model.simulate(n_samples=10, show_progress=False)
    print(sample.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 1

    **a)** Wyświetl CPD dla węzła z największą liczbą rodziców w nauczonej strukturze i porównaj liczbę parametrów tej tablicy z rozmiarem pełnej tablicy łącznej dla tych samych zmiennych.

    **b)** Sprawdź empirycznie własność otoczki Markowa: czy zapytanie `P(income | evidence)` daje ten sam wynik, gdy `evidence` zawiera wyłącznie otoczkę Markowa, jak wtedy gdy zawiera dodatkowo zmienne spoza niej?
    """)
    return


@app.cell
def _(adult, np, nx, plt):
    adult_num = adult.copy()
    for c in adult_num.columns:
        adult_num[c] = adult_num[c].astype('category').cat.codes

    cor = adult_num.corr()
    threshold = 0.1
    adj = (cor.abs() > threshold).values.copy()
    np.fill_diagonal(adj, False)

    G_thr = nx.from_numpy_array(adj)
    G_thr = nx.relabel_nodes(G_thr, dict(enumerate(cor.columns)))

    fig_thr, ax_thr = plt.subplots(figsize=(8, 6))
    nx.draw(G_thr, nx.spring_layout(G_thr, seed=42, k=1.5),
            with_labels=True, node_color='#a6c8fc', node_size=2200,
            font_size=11, edge_color='gray', ax=ax_thr)
    ax_thr.set_title(f"Graf z progowania |korelacji| > {threshold}")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pola losowe Markowa

    Czasem między zmiennymi nie ma naturalnej kolejności przyczynowej — np. piksele obrazu, sąsiadujące słowa w zdaniu, atomy w sieci krystalicznej. Wtedy model nieskierowany jest wygodniejszy. **Pole losowe Markowa** (MRF, Markov Random Field) ma postać

    $$P(\mathbf{x}) = \frac{1}{Z}\prod_{C\in\mathcal{C}}\phi_C(\mathbf{x}_C),$$

    gdzie $\mathcal{C}$ to zbiór klik grafu nieskierowanego, $\phi_C$ to nieujemne funkcje potencjału na konfiguracjach zmiennych w klice $C$, a $Z$ to stała normalizacyjna (Koller & Friedman, rozdz. 4).

    ## Moralizacja sieci bayesowskiej

    Do każdej sieci bayesowskiej można skonstruować równoważny MRF poprzez **moralizację**: usuwamy kierunki krawędzi i łączymy parami wszystkich rodziców każdego węzła (stąd potoczna nazwa: „ożeń rodziców"). Operacja zachowuje rozkład łączny, ale traci informację o niezależnościach kierunkowych.
    """)
    return


@app.cell
def _(bn_model, nx, plt):
    mn = bn_model.to_markov_model()

    G_mrf = nx.Graph()
    G_mrf.add_nodes_from(mn.nodes())
    G_mrf.add_edges_from(mn.edges())

    fig_mrf, ax_mrf = plt.subplots(figsize=(10, 7))
    nx.draw(G_mrf, nx.spring_layout(G_mrf, seed=42, k=1.6),
            with_labels=True, node_color='#bca6fc', node_size=2400,
            font_size=11, edge_color='gray', ax=ax_mrf)
    ax_mrf.set_title("MRF zmoralizowany z BN — Adult")
    plt.tight_layout()
    plt.show()

    print(f"Liczba czynników w MRF: {len(mn.factors)}")
    print(f"Pierwsze trzy czynniki (zmienne, kardynalność):")
    for f in mn.factors[:3]:
        print(f"  vars={f.variables}  card={list(f.cardinality)}")
    return (mn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Separacja vs d-separacja

    Niezależności warunkowe odczytuje się z grafu inaczej w MRF i w BN. W MRF działa zwykła **separacja**: $A\perp B\mid C$ wtedy i tylko wtedy, gdy każda ścieżka między dowolnym węzłem z $A$ a dowolnym węzłem z $B$ przechodzi przez $C$. W BN trzeba używać **d-separacji**, w której v-struktury (typu $A\to C\leftarrow B$) zachowują się odwrotnie: blokują ścieżkę gdy $C$ *nie* jest obserwowane, a otwierają gdy jest.

    Praktyczna konsekwencja moralizacji: dodanie krawędzi między rodzicami zmiennej zaciera subtelność v-struktury. Po moralizacji nie da się już odczytać z grafu, że dwóch rodziców dziecka byłoby brzegowo niezależnych — graf sugeruje, że pozostają w bezpośredniej relacji. To strata informacji, którą trzeba świadomie zaakceptować przy korzystaniu z reprezentacji nieskierowanej.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 2

    **a)** Znajdź w nauczonej sieci bayesowskiej węzeł z co najmniej dwoma rodzicami i wskaż w jego strukturze v-strukturę (parę rodziców i ich wspólne dziecko). Po moralizacji rodzice tego węzła powinni być połączeni krawędzią — zweryfikuj to wizualnie, porównując graf skierowany i graf MRF.

    **b)** Wykonaj test chi-kwadrat brzegowej niezależności tych dwóch rodziców (`scipy.stats.chi2_contingency` na tabeli kontyngencji). Czy są brzegowo niezależne, jak sugerowałaby v-struktura przed obserwacją dziecka? Następnie warunkuj zbiór po wartościach dziecka i ponownie wykonaj test — czy wyniki się różnią? Dlaczego MRF zmoralizowany "ukrywa" tę różnicę?
    """)
    return


@app.cell
def _(adult, bn_model, mn, nx, pd, plt):
    from scipy.stats import chi2_contingency


    # V-struktura: Pa1 → Child ← Pa2, gdzie Pa1 i Pa2 NIE są ze sobą połączone
    # w BN (brak krawędzi bezpośredniej). Pod warunkiem NIEobserwowania Child,
    # Pa1 ⊥ Pa2 (ścieżka zablokowana). Obserwując Child "otwieramy" ścieżkę —
    # to efekt "explaining away": wiedza o jednym rodzicu zmienia prawdopodobieństwo
    # drugiego, bo oba "wyjaśniają" tę samą wartość dziecka.


    child = max(bn_model.nodes(),
                key=lambda n: len(bn_model.get_parents(n)))
    parents = bn_model.get_parents(child)
    pa1, pa2 = parents[0], parents[1]   # bierzemy dwa pierwsze

    print(f"V-struktura: '{pa1}' → '{child}' ← '{pa2}'")
    print(f"Czy {pa1}—{pa2} w BN (skierowanym)?  "
          f"{bn_model.has_edge(pa1, pa2) or bn_model.has_edge(pa2, pa1)}")
    print(f"Czy {pa1}—{pa2} w MRF (po moralizacji)?  "
          f"{mn.has_edge(pa1, pa2)}")

    # --- wizualizacja porównawcza BN vs MRF ---
    G_bn_sub = nx.DiGraph()
    G_bn_sub.add_nodes_from([pa1, pa2, child])
    G_bn_sub.add_edges_from([(p, child) for p in [pa1, pa2]
                              if bn_model.has_edge(p, child)])

    # tylko krawędzie między pa1, pa2, child — nie wciągamy ich sąsiadów
    trio = {pa1, pa2, child}
    G_mrf_sub = nx.Graph()
    G_mrf_sub.add_nodes_from(trio)
    G_mrf_sub.add_edges_from((u, v) for u, v in mn.edges() if {u, v} <= trio)

    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4))
    pos = {pa1: (-1, 0), pa2: (1, 0), child: (0, -1)}

    nx.draw(G_bn_sub, pos, with_labels=True, ax=ax_l,
            node_color='#fcb1a6', node_size=2000, font_size=10,
            arrows=True, arrowsize=22, edge_color='gray',
            connectionstyle='arc3,rad=0.05')
    ax_l.set_title(f"BN (skierowany)\n{pa1} i {pa2} NIE połączone")

    highlight = [(pa1, pa2)] if G_mrf_sub.has_edge(pa1, pa2) else []
    other_edges = [e for e in G_mrf_sub.edges() if set(e) != {pa1, pa2}]
    nx.draw(G_mrf_sub, pos, with_labels=True, ax=ax_r,
            node_color='#bca6fc', node_size=2000, font_size=10, edge_color='gray')
    nx.draw_networkx_edges(G_mrf_sub, pos, edgelist=highlight,
                           edge_color='red', width=3, ax=ax_r)
    ax_r.set_title(f"MRF po moralizacji\n{pa1}—{pa2} dodane (czerwona krawędź)")

    plt.suptitle("V-struktura: BN vs MRF", fontsize=12)
    plt.tight_layout()
    plt.show()

    # ---- b) Test chi-kwadrat: niezależność brzegowa vs warunkowa ----
    #
    # H₀ dla testu chi-kwadrat: zmienne są niezależne (P(A,B) = P(A)·P(B)).
    # Małe p-value → odrzucamy H₀ → zmienne są zależne.
    #
    # Spodziewamy się:
    #  • BRZEGOWO: jeśli Pa1 i Pa2 tworzą "czystą" v-strukturę, powinny być
    #    niezależne (p duże). W praktyce mogą być słabo zależne przez inne ścieżki.
    #  • WARUNKOWO (po Child): obserwacja dziecka otwiera v-strukturę —
    #    Pa1 staje się zależna od Pa2 (p małe). To efekt "explaining away".
    #
    # MRF zmoralizowany dodaje krawędź Pa1—Pa2, ukrywając fakt, że ta zależność
    # jest INDUKOWANA przez obserwację dziecka, a nie bezpośrednia.

    print("\n" + "="*60)
    print("b) Chi-kwadrat: niezależność Pa1 ⊥ Pa2")
    print("="*60)

    # test brzegowy
    ct_marginal = pd.crosstab(adult[pa1], adult[pa2])
    chi2_m, p_m, dof_m, _ = chi2_contingency(ct_marginal)
    print(f"\nTest BRZEGOWY P({pa1} ⊥ {pa2}):")
    print(f"  chi2={chi2_m:.2f}  df={dof_m}  p={p_m:.4f}")
    if p_m > 0.05:
        print("  => p > 0.05: NIE odrzucamy H₀ — brzegowo niezależne (zgodnie z v-strukturą)")
    else:
        print("  => p ≤ 0.05: odrzucamy H₀ — brzegowo ZALEŻNE (inne ścieżki w grafie)")

    # test warunkowy: dla każdej wartości Child
    print(f"\nTest WARUNKOWY P({pa1} ⊥ {pa2} | {child}=x):")
    for val in sorted(adult[child].unique()):
        sub = adult[adult[child] == val]
        ct_cond = pd.crosstab(sub[pa1], sub[pa2])
        chi2_c, p_c, dof_c, _ = chi2_contingency(ct_cond)
        tag = "niezależne" if p_c > 0.05 else "ZALEŻNE  ← explaining away"
        print(f"  {child}={val:<10s}  chi2={chi2_c:7.2f}  df={dof_c}  p={p_c:.4f}  → {tag}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grafy czynników

    Graf czynników to **bipartytowa** reprezentacja faktoryzacji rozkładu: dwa rodzaje węzłów — okrągłe **węzły zmiennych** i kwadratowe **węzły czynników** — przy czym krawędzie łączą czynnik z każdą zmienną, od której zależy. Reprezentacja ta nie wnosi nowej informacji względem BN czy MRF, ale ujawnia strukturę faktoryzacji w sposób jednoznaczny i jest naturalna dla algorytmów **przekazywania komunikatów** (sum-product, belief propagation; Koller & Friedman, rozdz. 11).

    ## Konwersja BN → graf czynników

    Każde CPD $P(X_i \mid \mathrm{Pa}(X_i))$ staje się czynnikiem $\phi_i$ obejmującym $X_i$ i jego rodziców.
    """)
    return


@app.cell
def _(bn_model, mn, nx, plt):
    fg = mn.to_factor_graph()

    var_nodes = [n for n in fg.nodes() if not str(n).startswith('phi_')]
    factor_nodes = [n for n in fg.nodes() if str(n).startswith('phi_')]

    pos_fg = nx.spring_layout(fg, seed=42, k=1.8)
    fig_fg, ax_fg = plt.subplots(figsize=(11, 7.5))
    nx.draw_networkx_nodes(fg, pos_fg, nodelist=var_nodes, node_color='#fcb1a6',
                           node_size=2200, node_shape='o', ax=ax_fg)
    nx.draw_networkx_nodes(fg, pos_fg, nodelist=factor_nodes, node_color='#d4d4d4',
                           node_size=900, node_shape='s', ax=ax_fg)
    nx.draw_networkx_edges(fg, pos_fg, edge_color='gray', ax=ax_fg)
    nx.draw_networkx_labels(fg, pos_fg, font_size=9, ax=ax_fg)
    ax_fg.set_title("Graf czynników z BN — kółka: zmienne, kwadraty: czynniki")
    ax_fg.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Liczba węzłów zmiennych: {len(var_nodes)}")
    print(f"Liczba węzłów czynników: {len(factor_nodes)}")
    print(f"Czynniki BN: {len(bn_model.get_cpds())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Belief Propagation — to samo zapytanie, inny algorytm

    Eliminacja zmiennych wyznacza rozkład brzegowy jednej zmiennej naraz; każde nowe zapytanie wymaga osobnego przebiegu. **Belief Propagation** w jednym przejściu wyznacza brzegi wszystkich zmiennych jednocześnie, kosztem zbudowania **drzewa złączeń** z grafu czynników. Dla pojedynczego zapytania VE bywa szybsze; przy wielu zapytaniach na tym samym modelu BP zyskuje przewagę dzięki amortyzacji kosztu kalibracji.

    Sprawdźmy, że obie metody dają identyczny wynik:
    """)
    return


@app.cell
def _(BeliefPropagation, bn_model, inf):
    bp = BeliefPropagation(bn_model)
    bp.calibrate()

    evidence = {'sex': 'Male', 'education': 'Bachelors'}
    q_ve = inf.query(['income'], evidence=evidence, show_progress=False)
    q_bp = bp.query(['income'], evidence=evidence, show_progress=False)

    print("Variable Elimination:")
    print(q_ve)
    print("\nBelief Propagation:")
    print(q_bp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 3

    Zmierz `time.perf_counter()` dla pojedynczego zapytania oraz dla 50 różnych zapytań na obiektach `VariableElimination` i `BeliefPropagation` (pamiętaj o `calibrate()` po utworzeniu BP) — czy widzisz spodziewaną amortyzację BP?
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
    # Zestawienie

    | Cecha | Sieć bayesowska | MRF | Graf czynników |
    |---|---|---|---|
    | Graf | skierowany acykliczny | nieskierowany | bipartytowy |
    | Lokalne czynniki | $P(X_i\mid\mathrm{Pa}(X_i))$ — sumują się do 1 | $\phi_C(\mathbf{x}_C)$ — dowolne nieujemne | jak źródło konwersji |
    | Stała normalizacyjna | brak (faktoryzacja już daje rozkład) | $Z$ — wymaga policzenia | dziedziczona z reprezentacji |
    | Niezależności | d-separacja | separacja | jak MRF |
    | Naturalne kiedy | zależności kierunkowe / przyczynowe | symetryczne sąsiedztwa | algorytmy przekazywania komunikatów |
    | Uczenie struktury | HC + BIC/K2, PC, GES | rzadziej automatyczne | dziedziczone |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NBC i TAN — szczególne przypadki sieci bayesowskich

    Naiwny klasyfikator bayesowski (NBC) i jego rozszerzenie TAN (Tree-Augmented Naive Bayes) to sieci bayesowskie o ustalonej, prostej strukturze:

    - **NBC** — gwiazda wokół klasy: zmienna celu jest jedynym rodzicem wszystkich pozostałych. Założenie: cechy są warunkowo niezależne pod warunkiem klasy.
    - **TAN** — rozszerza NBC o drzewo nad cechami: każda cecha może mieć dodatkowo jednego rodzica wśród innych cech. Złagodzenie założenia o niezależności cech kosztem niewielu krawędzi.

    Strukturę TAN wyznacza algorytm Chow–Liu (drzewo o maksymalnej informacji wzajemnej między cechami) uzupełniony o krawędzie z klasy do wszystkich cech. W pgmpy realizuje to `TreeSearch` z `estimator_type='tan'`.
    """)
    return


@app.cell
def _(DiscreteBayesianNetwork, adult):
    features = [c for c in adult.columns if c != 'income']
    nbc = DiscreteBayesianNetwork([('income', col) for col in features])
    nbc.fit(adult)

    print("NBC — krawędzie:")
    print("\n".join(f"  {u} -> {v}" for u, v in nbc.edges()))
    return features, nbc


@app.cell
def _(DiscreteBayesianNetwork, TreeSearch, adult):
    ts = TreeSearch(adult)
    tan_dag = ts.estimate(estimator_type='tan', class_node='income', show_progress=False)
    tan = DiscreteBayesianNetwork(tan_dag.edges())
    tan.fit(adult)

    print("TAN — krawędzie:")
    print("\n".join(f"  {u} -> {v}" for u, v in tan.edges()))
    return (tan,)


@app.cell
def _(features, nbc, np, nx, plt, tan):
    fig_clf, axes_clf = plt.subplots(1, 2, figsize=(14, 6))

    G_nbc = nx.DiGraph(nbc.edges())
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    pos_nbc = {'income': (0.0, 0.0)}
    pos_nbc.update({
        feat: (np.cos(ang), np.sin(ang))
        for feat, ang in zip(features, angles)
    })
    nx.draw(G_nbc, pos_nbc, ax=axes_clf[0], with_labels=True,
            node_color='#fcd5a6', node_size=2000, font_size=9,
            arrows=True, arrowsize=15, edge_color='gray')
    axes_clf[0].set_title("NBC — gwiazda")

    G_tan = nx.DiGraph(tan.edges())
    pos_tan = nx.spring_layout(G_tan, seed=42, k=1.5)
    nx.draw(G_tan, pos_tan, ax=axes_clf[1], with_labels=True,
            node_color='#fcd5a6', node_size=2000, font_size=9,
            arrows=True, arrowsize=15, edge_color='gray')
    axes_clf[1].set_title("TAN — gwiazda + drzewo nad cechami")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zadania

    ## Zadanie 1 — domenowa sieć bayesowska vs nauczona

    Zaprojektuj ręcznie prostą sieć bayesowską dla zbioru Adult opartą na wiedzy domenowej (np. `age -> marital`, `education -> occupation`, `occupation -> income`).

    **a)** Dopasuj parametry obu sieci (ręcznej i wyuczonej HC + BIC) i porównaj log-likelihood na zbiorze.

    **b)** Wykonaj klasyfikację `income` jako MAP zmiennej zapytania (`inf.map_query`) na zbiorze testowym i porównaj dokładność — czy ranking według log-likelihood pokrywa się z rankingiem dokładności klasyfikacji?
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
    ## Zadanie 2 — otoczka Markowa w klasyfikacji

    Wytrenuj klasyfikator (np. `RandomForestClassifier`) zmiennej `income` na trzech wariantach predyktorów: (i) wszystkie zmienne, (ii) tylko otoczka Markowa, (iii) zmienne spoza otoczki. Porównaj dokładność testową oraz sprawdź, czy ranking ważności cech (`feature_importances_`) pokrywa się z otoczką Markowa.
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
    ## Zadanie 3 — wnioskowanie na grafie czynników

    Skonstruuj graf czynników z MRF (`mn.to_factor_graph()`).

    **a)** Porównaj liczbę czynników w grafie z liczbą CPD w wyjściowej BN — skąd różnica?

    **b)** Uruchom `BeliefPropagation` zainicjalizowane na `bn_model` i porównaj wyniki z `BeliefPropagation` zainicjalizowanym na `mn` — czy zapytania `bp.query(['income'], evidence=...)` dają identyczny wynik?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
