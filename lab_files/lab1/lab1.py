import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 1: Zadania powtórkowe ze statystyki
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from scipy import stats
    from pathlib import Path
    import matplotlib.pyplot as plt

    DATA_DIR = Path(__file__).parents[2] / "data"
    return DATA_DIR, np, pd, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkłady prawdopodobieństwa w Python (scipy.stats)

    Na zajęciach ze statystyki będą nam potrzebne rozkłady prawdopodobieństwa. W Pythonie, moduł `scipy.stats` dostarcza obiekty reprezentujące rozkłady prawdopodobieństwa. Każdy rozkład ma następujące metody:

    - `.pdf(x)` lub `.pmf(x)` - gęstość (dla rozkładów ciągłych) lub funkcja prawdopodobieństwa (dla dyskretnych)
    - `.cdf(x)` - dystrybuanta (kumulatywna funkcja rozkładu)
    - `.ppf(q)` - funkcja kwantylowa (odwrotność dystrybuanty, percent point function)
    - `.rvs(size)` - losowanie zgodne z rozkładem (random variates)

    Najczęściej używane rozkłady:
    - `stats.norm` - rozkład normalny (Gaussa)
    - `stats.uniform` - rozkład jednostajny
    - `stats.binom` - rozkład dwumianowy
    - `stats.poisson` - rozkład Poissona
    - `stats.expon` - rozkład wykładniczy
    - `stats.gamma` - rozkład gamma
    - `stats.t` - rozkład t-Studenta
    - `stats.chi2` - rozkład chi-kwadrat
    - `stats.f` - rozkład F (Fishera-Snedecora)
    - `stats.geom` - rozkład geometryczny
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkład normalny

    ### Definicja

    Mówimy, że zmienna losowa \(X\) ma **rozkład normalny** z parametrami \(\mu \in \mathbb{R}\) oraz \(\sigma > 0\), jeśli

    \[
    X \sim N(\mu,\sigma)
    \]

    Funkcja gęstości:

    \[
    \varphi_{\mu,\sigma}(x) =
    \frac{1}{\sigma\sqrt{2\pi}}
    e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    \]

    Parametry:

    - \(\mu\) — średnia (położenie środka rozkładu)
    - \(\sigma\) — odchylenie standardowe (rozrzut danych)

    ---

    ### Dystrybuanta

    Dystrybuanta rozkładu normalnego:

    \[
    \Phi_{\mu,\sigma}(x) =
    \frac{1}{\sigma\sqrt{2\pi}}
    \int_{-\infty}^{x}
    e^{-\frac{(y-\mu)^2}{2\sigma^2}}dy
    \]

    Interpretacja:

    \[
    \Phi_{\mu,\sigma}(x)=P(X \le x)
    \]

    ---

    ### Symetria

    Rozkład normalny jest symetryczny względem \(\mu\).

    \[
    \Phi_{\mu,\sigma}(\mu + c) + \Phi_{\mu,\sigma}(\mu - c) = 1
    \]

    \[
    P(|X-\mu|\le c)=2\Phi_{\mu,\sigma}(\mu+c)-1
    \]

    ---

    ### Charakterystyki rozkładu

    Jeśli \(X \sim N(\mu,\sigma)\), to:

    | Wielkość | Wartość |
    |---|---|
    | Wartość oczekiwana | \(E(X)=\mu\) |
    | Mediana | q_1/2 \(=\mu\)
    | Wariancja | \(Var(X)=\sigma^2\) |
    | Odchylenie standardowe | \(\sigma = \sqrt{\lambda}\) |
    | Skośność | \(A=0\) |
    | Kurtoza  | \(Kurt(X)=3\)|

    ---

    ### Reguły prawdopodobieństwa

    Reguła \(2\sigma\):

    \[
    P(|X-\mu|>2\sigma) \approx 0.0455
    \]

    Reguła \(3\sigma\):

    \[
    P(|X-\mu|>3\sigma) \approx 0.0027
    \]

    ---

    ### Reguła 68–95–99.7

    W przybliżeniu:

    - \(P(|X-\mu| \le 1\sigma) \approx 0.68\)
    - \(P(|X-\mu| \le 2\sigma) \approx 0.95\)
    - \(P(|X-\mu| \le 3\sigma) \approx 0.997\)

    ---

    ### Standaryzacja

    Zmiana zmiennej:

    \[
    Z=\frac{X-\mu}{\sigma}
    \]

    Wtedy:

    \[
    Z \sim N(0,1)
    \]

    czyli standardowy rozkład normalny.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dyskretny rozkład jednostajny

    Rozkład jednostajny opisuje zmienną losową, która **przyjmuje wszystkie wartości w zbiorze S z jednakowym prawdopodobieństwem**.

    Dla zmiennej losowej \(X\) o wartościach:

    \[
    S = \{x_1, x_2, \dots, x_n\}
    \]

    jeśli

    \[
    P(X = x_i) = \frac{1}{n}, \quad \forall i = 1, \dots, n
    \]

    mówimy, że \(X\) ma **dyskretny rozkład jednostajny na n punktach**.

    ---

    ### Funkcja prawdopodobieństwa (PMF)

    \[
    P(X = x_i) = \frac{1}{n}, \quad i = 1, \dots, n
    \]

    - Każda wartość z \(S\) jest **równie prawdopodobna**.
    - Wartość oczekiwana i wariancja można policzyć z definicji.

    ---

    ### Wielkości opisujące rozkład

    Dla \(X \sim \text{Uniform}(x_1, \dots, x_n)\):

    | Wielkość | Wzór | Interpretacja |
    |---|---|---|
    | Średnia | \(\mu = \frac{1}{n} \sum_{i=1}^{n} x_i\) | Środek rozkładu |
    | Wariancja | \(\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2\) | Miara rozrzutu |
    | Skośność | \(A = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^3}{\left(\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2\right)^{3/2}}\) | Symetria rozkładu |
    | Kurtoza | \(K = \frac{n \sum_{i=1}^{n} (x_i - \mu)^4}{\left(\sum_{i=1}^{n} (x_i - \mu)^2\right)^2}\) | “Spiczastość” rozkładu |

    ---

    ### Przykład

    Rozkład jednostajny na punktach \(S = \{1,2,3,4,5,6\}\) (rzut kostką):

    - \(P(X=1) = P(X=2) = \dots = P(X=6) = 1/6\)
    - Średnia:

    \[
    \mu = \frac{1+2+3+4+5+6}{6} = 3.5
    \]

    - Wariancja:

    \[
    \sigma^2 = \frac{(1-3.5)^2 + (2-3.5)^2 + \dots + (6-3.5)^2}{6} \approx 2.92
    \]

    - Odchylenie standardowe: \(\sigma \approx 1.71\)

    ---

    ### Interpretacja

    - Każda wartość w zbiorze jest **równie prawdopodobna**.
    - Dyskretny rozkład jednostajny jest **symetryczny** (skośność = 0), jeśli punkty są równomiernie rozmieszczone.
    - Jest podstawą do **losowania liczb losowych**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkład Poissona

    Rozkład Poissona opisuje **liczbę zdarzeń powtarzalnych** w danym przedziale czasowym lub przestrzennym, jeśli spełnione są założenia:

    1. Zdarzenia występują **niezależnie** od siebie.
    2. Średnia liczba zdarzeń w jednostce czasu (intensywność) \(r > 0\) jest **stała**.
    3. W bardzo krótkim przedziale czasu może zajść **co najwyżej jedno zdarzenie**.

    Jeśli czas trwania przedziału to \(t\), to liczba zdarzeń \(N\) ma rozkład Poissona:

    \[
    N \sim \text{Pois}(\lambda), \quad \lambda = r t
    \]

    ---

    ### Funkcja prawdopodobieństwa (PMF)

    Dla zmiennej losowej \(X \sim \text{Pois}(\lambda)\):

    \[
    P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0,1,2,\dots
    \]

    - \(\lambda > 0\) – średnia liczba zdarzeń
    - \(k\) – liczba zdarzeń w danym przedziale

    ---

    ### Dystrybuanta (CDF)

    Dystrybuanta to:

    \[
    F(k) = P(X \le k) = \sum_{i=0}^{k} \frac{e^{-\lambda} \lambda^i}{i!}
    \]

    czyli prawdopodobieństwo, że liczba zdarzeń nie przekroczy \(k\).

    ---

    ### Charakterystyki rozkładu

    Jeśli \(X \sim \text{Pois}(\lambda)\):

    | Wielkość | Wartość |
    |---|---|
    | Wartość oczekiwana | \(E(X) = \lambda\) |
    | Wariancja | \(\text{Var}(X) = \lambda\) |
    | Odchylenie standardowe | \(\sigma = \sqrt{\lambda}\) |
    | Skośność | \(A = 1/\sqrt{\lambda}\) |
    | Kurtoza nadmiarowa | \(\gamma_2 = 1/\lambda\) |

    ---

    ### Interpretacja

    - Rozkład Poissona modeluje **rzadkie zdarzenia**, np.:
      - liczba wypadków w ciągu godziny
      - liczba telefonów przychodzących na infolinię
      - liczba błędów w tekście
    - Jeśli \(\lambda\) jest duże, rozkład Poissona przypomina **rozkład normalny** (prawo centralne graniczne).

    ---

    ### Związek z rozkładem dwumianowym

    Twierdzenie Poissona:
    Jeśli \(X_n \sim \text{Binom}(n, \theta_n)\) i \(n \to \infty\), \(\theta_n \to 0\) tak, że \(n\theta_n \to \lambda\), to:

    \[
    \lim_{n \to \infty} P(X_n = k) = \frac{e^{-\lambda} \lambda^k}{k!}
    \]

    czyli **rozkład Poissona jest granicą rozkładu dwumianowego dla rzadkich zdarzeń**.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkład dwumianowy

    Rozkład dwumianowy opisuje **liczbę sukcesów w \(n\) niezależnych próbach**, w których każda próba ma prawdopodobieństwo sukcesu \(\theta\).

    Zmienna losowa \(X\) ma rozkład dwumianowy, jeśli:

    \[
    X \sim \text{Binom}(n, \theta)
    \]

    - \(n \in \mathbb{N}\) – liczba prób
    - \(\theta \in (0,1)\) – prawdopodobieństwo sukcesu w pojedynczej próbie

    ---

    ### Funkcja prawdopodobieństwa (PMF)

    Dla \(k \in \{0,1,\dots,n\}\):

    \[
    P(X = k) = \binom{n}{k} \theta^k (1-\theta)^{\,n-k}
    \]

    - \(\binom{n}{k} = \frac{n!}{k!(n-k)!}\) – liczba sposobów wybrania \(k\) sukcesów z \(n\) prób
    - \(\theta^k\) – prawdopodobieństwo, że te \(k\) próby zakończą się sukcesem
    - \((1-\theta)^{n-k}\) – prawdopodobieństwo, że pozostałe próby zakończą się niepowodzeniem

    ---

    ### Wielkości opisujące rozkład dwumianowy

    Jeśli \(X \sim \text{Binom}(n, \theta)\):

    | Wielkość | Wzór |
    |---|---|
    | Średnia | \(\mu = E(X) = n\theta\) |
    | Wariancja | \(\text{Var}(X) = n\theta(1-\theta)\) |
    | Skośność | \(A = \frac{1-2\theta}{\sqrt{n\theta(1-\theta)}}\) |
    | Kurtoza nadmiarowa | \(\gamma_2 = \frac{1-6\theta(1-\theta)}{n\theta(1-\theta)}\) |

    ---

    ### Uwagi

    - Dla \(n=1\) rozkład dwumianowy **staje się rozkładem Bernoulliego**:
    \[
    \text{Binom}(1, \theta) = \text{Bern}(\theta)
    \]

    - Jeśli \(n\) jest duże i \(\theta\) jest małe, rozkład dwumianowy **przybliża się do rozkładu Poissona** (Twierdzenie Poissona).

    ---

    ### Interpretacja

    Rozkład dwumianowy odpowiada pytaniu:

    > “Ile sukcesów w \(n\) niezależnych próbach, jeśli każda próba ma prawdopodobieństwo sukcesu \(\theta\)?”

    Przykłady:

    - ile razy wypdanie orzeł w 10 rzutach monetą (\(\theta = 0.5\))
    - liczba osób, które kliknęły reklamę w grupie 1000 odbiorców (\(\theta = 0.02\))
    - liczba wadliwych produktów w partii 50 sztuk (\(\theta = 0.05\))

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkład wykładniczy

    Rozkład wykładniczy modeluje **czas oczekiwania na pierwsze zdarzenie** w procesie Poissona, gdzie liczba zdarzeń w jednostce czasu opisuje \(X \sim \text{Pois}(\lambda t)\).

    - Zmienna losowa \(T \sim \text{Exp}(\lambda)\) opisuje **czas do pierwszego zdarzenia**.
    - Parametr \(\lambda > 0\) – intensywność zdarzeń (średnia liczba zdarzeń na jednostkę czasu).

    ---

    ### Dystrybuanta (CDF)

    Dla \(t \ge 0\):

    \[
    F(t) = P(T \le t) =
    \begin{cases}
    0 & t < 0 \\
    1 - e^{-\lambda t} & t \ge 0
    \end{cases}
    \]

    Interpretacja: prawdopodobieństwo, że zdarzenie zajdzie **przed czasem \(t\)**.

    ---

    ### Funkcja gęstości prawdopodobieństwa (PDF)

    \[
    f(t) =
    \begin{cases}
    0 & t < 0 \\
    \lambda e^{-\lambda t} & t \ge 0
    \end{cases}
    \]

    - Wartość \(\lambda e^{-\lambda t}\) maleje wykładniczo z czasem.
    - **Im większe \(\lambda\)**, tym szybciej zdarzenie zajdzie.

    ---

    ### Wielkości opisujące rozkład wykładniczy

    Dla \(T \sim \text{Exp}(\lambda)\):

    | Wielkość | Wzór | Interpretacja |
    |---|---|---|
    | Średnia | \(\mu = E(T) = 1/\lambda\) | Średni czas oczekiwania |
    | Mediana | \(q_{1/2} = \frac{\ln 2}{\lambda}\) | Czas, w którym 50% zdarzeń już nastąpiło |
    | Wariancja | \(\sigma^2 = 1/\lambda^2\) | Miara rozrzutu czasów |
    | Skośność | \(A = 2\) | Rozkład jest silnie prawostronnie skośny |
    | Kurtoza nadmiarowa | \(\gamma_2 = 6\) | Rozkład “spiczasty” w porównaniu z normalnym |

    ---

    ### Interpretacja

    - Modeluje **czas między zdarzeniami w procesie Poissona**.
    - Przykłady zastosowań:
      - czas oczekiwania na autobus, gdy przyjazdy są losowe
      - czas do awarii maszyny
      - czas życia cząsteczki w reakcji chemicznej

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkład gamma

    Rozkład gamma jest **ciągłym rozkładem prawdopodobieństwa** stosowanym m.in. do modelowania **czasów oczekiwania**. Jest ogólniejszy niż rozkład wykładniczy:

    \[
    \text{Gamma}(1,r) = \text{Exp}(r)
    \]

    ---

    ### Parametry

    - \(s > 0\) – parametr **kształtu** (shape)
    - \(r > 0\) – parametr **intensywności** (rate)
    - Czasem zamiast \(r\) używa się \(1/r\) jako **parametru skali** (scale)

    Zmienna losowa \(X \sim \text{Gamma}(s,r)\) ma gęstość:

    \[
    f(x) =
    \begin{cases}
    0 & x \le 0 \\
    \frac{r^s}{\Gamma(s)} x^{s-1} e^{-rx} & x > 0
    \end{cases}
    \]

    gdzie \(\Gamma(s)\) jest funkcją Eulera:

    \[
    \Gamma(s) = \int_0^\infty x^{s-1} e^{-x} dx
    \]

    ---

    ### Dystrybuanta (CDF)

    \[
    F(x) = P(X \le x) = \int_0^x f(t) dt
    \]

    - Dystrybuanta rośnie od 0 do 1 wraz ze wzrostem \(x\)
    - Parametr \(s\) zmienia kształt rozkładu, \(r\) wpływa na skalę czasu

    ---

    ### Wielkości opisujące rozkład gamma

    Dla \(X \sim \text{Gamma}(s,r)\):

    | Wielkość | Wzór | Interpretacja |
    |---|---|---|
    | Średnia | \(\mu = s/r\) | Średni czas oczekiwania |
    | Wariancja | \(\sigma^2 = s/r^2\) | Rozrzut czasów |
    | Skośność | \(A = 2/\sqrt{s}\) | Jak bardzo rozkład jest prawostronnie skośny |
    | Kurtoza nadmiarowa | \(\gamma_2 = 6/s\) | “Spiczastość” rozkładu |

    ---

    ### Interpretacja

    - Modeluje **sumę \(s\) niezależnych zmiennych wykładniczych** z tym samym parametrem \(r\)
    - Przykłady zastosowań:
      - czas życia systemu z wieloma niezależnymi etapami
      - czas do kolejnych zdarzeń w procesie Poissona
      - modelowanie kolejek w systemach obsługi

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkład t-Studenta

    Rozkład t o \( \nu \) stopniach swobody (\( \nu > 0 \)), zwany także **rozkładem Studenta**, jest **ciągłym rozkładem prawdopodobieństwa** stosowanym m.in. do wnioskowania o średniej populacji, gdy wariancja populacji jest nieznana.

    ---

    ### Parametr

    - \( \nu > 0 \) – liczba **stopni swobody**, zwykle związana z licznością próby minus 1: \( \nu = n-1 \)

    ---

    ### Gęstość prawdopodobieństwa (PDF)

    Dla zmiennej losowej \( T \sim t_\nu \):

    \[
    f(x) = \frac{\Gamma\Big(\frac{\nu+1}{2}\Big)}{\sqrt{\pi \nu}\, \Gamma\Big(\frac{\nu}{2}\Big)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}, \quad x \in \mathbb{R}
    \]

    - \(\Gamma(\cdot)\) – funkcja gamma Eulera
    - Dla \( \nu \to \infty \), rozkład t zbiega do rozkładu normalnego \( N(0,1) \)

    ---

    ### Alternatywna definicja

    Jeśli \( Z \sim N(0,1) \) i \( V \sim \chi^2(\nu) \) są **niezależne**, to:

    \[
    T = \frac{Z}{\sqrt{V/\nu}} \sim t_\nu
    \]

    ---

    ### Dystrybuanta (CDF)

    \[
    F(x) = P(T \le x) = \int_{-\infty}^x f(t) dt
    \]

    - Dystrybuanta rośnie od 0 do 1 wraz ze wzrostem \(x\)
    - Dla małych \( \nu \) rozkład ma **grubsze ogony** niż normalny

    ---

    ### Wielkości opisujące rozkład t

    | Wielkość | Wzór / Uwagi | Interpretacja |
    |---|---|---|
    | Średnia | \(0\) dla \(\nu > 1\), nie istnieje dla \(\nu \le 1\) | Symetria wokół 0 |
    | Wariancja | \(\frac{\nu}{\nu-2}\) dla \(\nu > 2\), \(+\infty\) dla \(1 < \nu \le 2\), nie istnieje dla \(\nu \le 1\) | Rozrzut wartości |
    | Skośność | 0 dla \(\nu > 3\), nie istnieje dla \(\nu \le 3\) | Symetria rozkładu |
    | Kurtoza nadwyżkowa | \(\gamma_2 = \frac{6}{\nu-4}\) dla \(\nu > 4\), \(+\infty\) dla \(2 < \nu \le 4\), nie istnieje dla \(\nu \le 2\) | Grubość ogonów i “spiczastość” |

    ---

    ### Interpretacja

    - Modeluje **standaryzowaną różnicę średniej próby od średniej populacji** przy nieznanej wariancji
    - Stosowany w testach t i **przedziałach ufności dla średniej**:

    Dla próby \( X_1, \dots, X_n \) z nieznaną wariancją \( \sigma^2 \):

    \[
    T = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}
    \]

    gdzie:

    - \( \bar{X} \) – średnia próby
    - \( S \) – odchylenie standardowe próby
    - \( n-1 \) – liczba stopni swobody

    - Przedział ufności dla \( \mu \) na poziomie \(1-\alpha\):

    \[
    \left[ \bar{X} - t_{1-\alpha/2,\, n-1} \frac{S}{\sqrt{n}},\;\; \bar{X} + t_{1-\alpha/2,\, n-1} \frac{S}{\sqrt{n}} \right]
    \]

    gdzie \( t_{p, \nu} \) jest kwantylem rozkładu t-Studenta o \(\nu\) stopniach swobody.

    ---

    ### Właściwości

    - **Symetryczny** względem 0
    - **Grubsze ogony** dla małych \( \nu \), co zwiększa odporność na wartości odstające
    - Zbliża się do rozkładu normalnego dla dużych \( \nu \)
    """)
    return


@app.cell
def _(np, stats):
    # Rozkład normalny N(0, 1)
    print(stats.norm.pdf(2.3))        # gęstość w punkcie 2.3
    print(stats.norm.cdf(2.3))        # P(X <= 2.3)
    print(stats.norm.ppf(0.975))      # kwantyl 97.5%

    # Losowanie z rozkładu normalnego
    x1 = stats.norm.rvs(size=10)      # 10 losowań z N(0, 1)
    print(f"Średnia: {np.mean(x1):.4f}")
    print(f"Wariancja: {np.var(x1, ddof=1):.4f}")  # ddof=1 dla wariancji próbkowej
    print(f"Odch. std.: {np.std(x1, ddof=1):.4f}")
    return


@app.cell
def _(np, stats):
    # Rozkład normalny z innymi parametrami N(1, 25)
    x2 = stats.norm.rvs(loc=1, scale=5, size=10)  # loc=średnia, scale=odch.std.
    print(f"Średnia: {np.mean(x2):.4f}")
    print(f"Wariancja: {np.var(x2, ddof=1):.4f}")
    print(f"Odch. std.: {np.std(x2, ddof=1):.4f}")
    return


@app.cell
def _(stats):
    # Rozkład Poissona
    print(stats.poisson.pmf(2, mu=1))        # P(X = 2) dla Poisson(1)
    print(stats.poisson.cdf(2, mu=1))        # P(X <= 2)
    print(stats.poisson.ppf(0.75, mu=1))     # kwantyl 75%
    print(stats.poisson.rvs(mu=1, size=10))  # 10 losowań
    return


@app.cell
def _(stats):
    # Rozkład dwumianowy Binom(n=10, p=0.3)
    print(stats.binom.pmf(3, n=10, p=0.3))       # P(X = 3)
    print(stats.binom.rvs(n=10, p=0.3, size=5))  # 5 losowań
    return


@app.cell
def _(stats):
    # Rozkład jednostajny U(0, 1)
    print(stats.uniform.rvs(size=5))                  # 5 losowań z U(0, 1)
    print(stats.uniform.rvs(loc=2, scale=3, size=5))  # 5 losowań z U(2, 5)
                                                       # scale = max - min
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dla osiągnięcia powtarzalności obliczeń z czynnikiem losowym stosuje się `np.random.seed()` lub `random_state`:
    """)
    return


@app.cell
def _(np, stats):
    # Bez ustawienia ziarna - różne wyniki
    print("Losowanie 1:", stats.norm.rvs(size=5))
    print("Losowanie 2:", stats.norm.rvs(size=5))

    # Z ustawionym ziarnem - powtarzalne wyniki
    np.random.seed(2020)
    print("Z ziarnem 1:", stats.norm.rvs(size=5))

    np.random.seed(2020)
    print("Z ziarnem 2:", stats.norm.rvs(size=5))  # Te same wartości co powyżej
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatywnie, można użyć `random_state` dla większej kontroli:
    """)
    return


@app.cell
def _(np, stats):
    # Tworzenie generatora z konkretnym ziarnem
    rng = np.random.default_rng(2020)
    print("Generator 1:", stats.norm.rvs(size=5, random_state=rng))

    # Ten sam generator daje kolejne wartości
    print("Generator 2:", stats.norm.rvs(size=5, random_state=rng))


    # Nowy generator z tym samym ziarnem
    rng2 = np.random.default_rng(2020)
    print("Nowy generator:", stats.norm.rvs(size=5, random_state=rng2))  # Te same co Generator 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 1

    Czas oczekiwania na pewne zdarzenie ma rozkład Gamma(3, r). Wykonano serię
    pomiarów i uzyskano czasy 1.4, 1.8, 1.4, 1.4 i 1.5. Oblicz estymatę
    największej wiarygodności parametru r.
    """)
    return


@app.cell
def _(stats):
    # Rozwiązanie zadania 2
    # Uzupełnij kod poniżej

    times = [1.4, 1.8, 1.4, 1.4 , 1.5]
    a_hat, loc_hat, scale_hat = stats.gamma.fit(times, method="MLE", fa=3, floc=0)

    r_hat = 1 / scale_hat

    print("a_hat:", a_hat)
    print("loc_hat:", loc_hat)
    print("scale_hat:", scale_hat)
    print("r_hat (rate):", r_hat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2

    Plik `goals.csv` zawiera dane o liczbie goli strzelonych przez pewną drużynę
    piłkarską w kolejnych meczach. Zakładamy, że liczba goli ma rozkład Poissona
    o nieznanej wartości λ. Wyznacz estymator największej wiarygodności parametru λ.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    goals_df = pd.read_csv(DATA_DIR / "goals.csv")
    print(goals_df.describe())
    goals_df
    return (goals_df,)


@app.cell
def _(goals_df):
    # Estymacja MLE dla rozkładu Poissona
    # Uzupełnij kod poniżej

    data = goals_df.iloc[:, 0] 

    goals_lambda_mle = data.mean()
    print(goals_lambda_mle )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 3
    Ladislaus Josephovich Bortkiewicz (lub Władysław Bortkiewicz) w 1898 roku wykorzystał dane z Preussische Statistik o śmierciach w wyniku kopnięcia przez konia w 14 dywizjach wojska pruskiego do pokazania, że rozkład Poissona dobrze opisuje rzadkie, losowe zdarzenia w dużych populacjach. Chociaż zarzucono mu później celowe odrzucenie kilku dywizji dla lepszego dopasowania, otrzymany zbiór danych stał się klasycznym przykładem zastosowania tego rozkładu w statystyce i jednym z pierwszych empirycznych dowodów na jego przydatność w modelowaniu rzeczywistych danych.

    Zakładamy, że liczba ofiar w korpusach w danym roku ma rozkład Poissona o nieznanym parametrze λ. Korzystając z poniższych danych, wyznacz estymator największej wiarygodności parametru λ oraz porównaj teoretyczny rozkład Poissona z empirycznym rozkładem danych.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    kicks_df = pd.read_csv(DATA_DIR / "kicks.csv", header=0, index_col=0)
    kicks_df
    return (kicks_df,)


@app.cell
def _(kicks_df):
    # Estymacja MLE dla rozkładów Poissona
    # Uzupełnij kod poniżej
    lambda_mle_to_year = {}
    for i in range(kicks_df.shape[1]):
        year_stats = kicks_df.iloc[:, i]
        lambda_mle_to_year[year_stats.name] = year_stats.mean()
        print(f'Year: {year_stats.name} lambda MLE: {year_stats.mean()}')
    return (lambda_mle_to_year,)


@app.cell
def _(kicks_df, lambda_mle_to_year, np, plt, stats):
    n_years = kicks_df.shape[1]
    n_cols = 4  
    n_rows = int(np.ceil(n_years / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    for j in range(n_years):
        year = kicks_df.columns[j]
        kics_data = kicks_df.iloc[:, j]
        lambda_mle = lambda_mle_to_year[year]
    
        counts = kics_data.value_counts().sort_index()
        x_emp = counts.index
        y_emp = counts.values
    
        x = np.arange(0, x_emp.max() + 1)
        y_poisson = stats.poisson.pmf(x, mu=lambda_mle) * len(kics_data)
    
        width = 0.4
        ax = axes[j]
        ax.bar(x_emp - width/2, y_emp, width=width, alpha=0.6, label='Empiryczny')
        ax.bar(x + width/2, y_poisson, width=width, alpha=0.6, label='Poisson')
    
        ax.set_title(f'{year} - λ={lambda_mle:.2f}')
        ax.set_xlabel('Liczba ofiar')
        ax.set_ylabel('Liczba korpusów')
        ax.legend()


    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 4

    Wyznacz przedziały ufności na poziomie 0.95 i 0.99 dla średniej wysokości
    drzew ze zbioru `trees`.

    Zbiór `trees` jest dostępny przez bibliotekę `statsmodels`:
    """)
    return


@app.cell
def _():
    from statsmodels.datasets import get_rdataset

    trees = get_rdataset("trees").data
    print(trees.describe())
    return (trees,)


@app.cell
def _(stats, trees):
    # Przedziały ufności dla średniej wysokości
    # Uzupełnij kod poniżej

    # ==========================
    # Wyznaczanie przedziałów ufności dla średniej wysokości drzew
    # ==========================

    # Założenia:
    # - Wysokość drzew przyjmujemy, że pochodzi z populacji o rozkładzie normalnym
    # - Nie znamy odchylenia standardowego populacji, dlatego używamy odchylenia próby
    # - Do wyznaczenia przedziału ufności stosujemy rozkład t-Studenta zamiast rozkładu normalnego
    #   ponieważ przy nieznanej wariancji próby rozkład t uwzględnia dodatkową niepewność
    # 
    # Przedział ufności dla średniej przy nieznanym σ ma postać:
    #   X̄ ± t_(1-α/2, n-1) * (S / √n)
    # gdzie:
    #   - X̄       : średnia z próby
    #   - S        : odchylenie standardowe próby (wariancja dzielona przez n-1)
    #   - n        : liczba obserwacji w próbie
    #   - α        : poziom istotności (np. 0.05 dla 95% CI, 0.01 dla 99% CI)
    #   - t_(1-α/2, n-1) : kwantyl rzędu (1-α/2) rozkładu t-Studenta
    #                       o (n-1) stopniach swobody
    #                       Inaczej mówiąc, jest to wartość t, taka że
    #                       P(T ≤ t_(1-α/2, n-1)) = 1-α/2 dla zmiennej T ∼ t_(n-1)
    #   - W ten sposób otrzymujemy przedział, w którym z prawdopodobieństwem 1-α
    #     znajduje się prawdziwa średnia populacji μ

    trees_height = trees['Height']
    samples_mean = trees_height.mean()
    samples_std = trees_height.std()
    n = len(trees_height)
    alpha_values = [0.05, 0.01]
    for a in alpha_values:
        t_pmf = stats.t.ppf(1-a/2,df=n-1)
        x_1 = samples_mean - t_pmf *  samples_std / (n**0.5)
        x_2 = samples_mean + t_pmf *  samples_std / (n**0.5)
        print(f'Przedział unfości dla alpha={a}: [{x_1}, {x_2}]')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 5

    Ustal minimalną liczebność próby dla oszacowania średniej wzrostu noworodków
    o rozkładzie N(μ, 1.5 cm). Zakładamy maksymalny błąd szacunku d = 0.5 cm
    oraz poziom ufności 0.99.
    """)
    return


@app.cell
def _(np, stats):
    # Minimalna liczebność próby
    # Uzupełnij kod poniżej


    # ==========================
    # Wyznaczanie minimalnej liczebności próby dla średniej wzrostu noworodków
    # ==========================

    # Założenia:
    # - Wzrost noworodków jest rozkładu normalnego N(μ, σ^2) z znanym odchyleniem standardowym σ
    # - Chcemy, aby maksymalny błąd szacunku średniej (d) nie przekroczył określonej wartości
    #   (czyli nasza średnia z próby będzie ±d od prawdziwej średniej μ)
    # - Poziom ufności 1-α (np. 0.99) określa, z jakim prawdopodobieństwem przedział obejmuje μ
    #
    # Wzór na minimalną próbę:
    #   d = z_(1-α/2) * σ / √n   =>   n = (z_(1-α/2) * σ / d)^2
    #
    # Gdzie:
    #   - σ       : odchylenie standardowe populacji
    #   - d       : maksymalny dopuszczalny błąd szacunku
    #   - α       : poziom istotności (α = 1 - poziom ufności)
    #   - z_(1-α/2) : kwantyl rzędu 1-α/2 rozkładu normalnego N(0,1)
    #                 czyli wartość z, taka że P(Z ≤ z) = 1-α/2
    #   - n       : minimalna liczba obserwacji potrzebna, aby osiągnąć zadany błąd d przy poziomie ufności


    std = 1.5      
    d = 0.5         
    alpha = 0.01   
    z = stats.norm.ppf(1 - alpha/2)

    n_min = (z * std / d) ** 2  
    print(f'Minimalna liczebność to {np.ceil(n_min)}') 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 6

    Automat produkuje blaszki o nominalnej grubości 0.04 mm. Wyniki pomiarów
    grubości losowej próby 25 blaszek zebrane są w pliku `blaszki.csv`. Czy można
    twierdzić, że blaszki są cieńsze niż 0.04 mm? Przyjmujemy rozkład normalny
    grubości blaszek oraz poziom istotności α = 0.01.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    blaszki_df = pd.read_csv(DATA_DIR / "blaszki.csv")
    blaszki_df
    return


@app.cell
def _():
    # Test hipotezy jednostronnej (blaszki cieńsze niż 0.04mm)
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 7

    Spośród 97 chorych na pewną chorobę, losowo wybranym 51 pacjentom podano lek.
    Pozostałym 46 podano placebo. Po tygodniu 12 pacjentów, którym podano lek,
    oraz 5 spośród tych, którym podano placebo, poczuło się lepiej. Zweryfikuj
    hipotezę o braku wpływu podanego leku na samopoczucie pacjentów.
    """)
    return


@app.cell
def _():
    # Test niezależności / test chi-kwadrat dla tabeli kontyngencji
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
