import numpy as np
from scipy.stats import norm

# Latent Functions

def friedman(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]
    
    if x.shape[1] < 5:
        raise ValueError("Friedman function input must have at least 5 variables.")
    
    return (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * (x[:, 2] - 0.5)**2
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )


def correlated_predictors(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]

    if x.shape[1] < 10:
        raise ValueError("Correlated predictors function input must have at least 10 variables.")
    
    return (
        2 * x[:, 0] * x[:, 3] + 2 * x[:, 6] * x[:, 9]
    )


def friedman_p(x: np.ndarray, p: int):
    p_ = np.ceil(p/2).astype(int)

    if x.ndim == 1:
        x = x[None]
    
    if x.shape[1] < (p_ + 3):
        raise ValueError("friedman_p function input must have at least p/2 + 3 variables.")
    
    return(
        friedman(x[:, [p_, p_+1, p_+2, 0, 1]])
    )


def modfriedman(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]

    if x.shape[1] < 44:
        raise ValueError("modfriedman input must have at leats 44 variables.")
    
    return (
        -4 + x[:, 0]
        + np.sin(np.pi * x[:, 0] * x[:, 43])
        - x[:, 20]
        + 0.6 * x[:, 40] * x[:, 41]
        - np.exp(-2 * (x[:, 41] + 1)**2)
        - x[:, 42]**2 + 0.5*x[:, 43]
    )


#Scenarion definitions

def datagen_cc0(n: int, p:int, s:float):
    X = np.random.normal(0, 1, (n, p))
    y = np.random.normal(0, 1, n)

    return X, y, np.full(p, False)


def datagen_cc1(n: int, p: int, s: float):
    X = np.random.uniform(0, 1, (n, p))
    y = friedman(X) + np.random.normal(0, s, n)

    truth = np.full(p, False)
    truth[0:5] = True

    return X, y, truth


def datagen_cc2(n: int, p:int, s: float):
    indices = np.arange(p)
    S = 0.3 ** np.abs(indices[:, None] - indices[None, :])

    X = np.random.multivariate_normal(np.zeros(p), S, n)
    y = correlated_predictors(X) + np.random.normal(0, s, n)

    truth = np.full(p, False)
    truth[[0, 3, 6, 9]] = True

    return X, y, truth


def datagen_cm1(n: int, p: int, s: float):
    p_ = np.ceil(p/2).astype(int)

    X = np.concat(
        (
            np.random.binomial(1, 0.5, (n, p_)),
            np.random.uniform(0, 1, (n, p_))
        ),
        axis = 1
    )
    y = friedman_p(X, p) + np.random.normal(0, s, n)

    truth = np.full(p, False)
    truth[[p_, p_+1, p_+2, 0, 1]] = True

    return X, y, truth


def datagen_cm2(n: int, p: int, s: float):
    S = np.full((44, 44), 0.3)
    np.fill_diagonal(S, 1)

    X = np.concat(
        (
            np.random.binomial(1, 0.2, (n, 20)),
            np.random.binomial(1, 0.5, (n, 20)),
            np.random.multivariate_normal(np.zeros(44), S, n)
        ),
        axis = 1
    )
    y = modfriedman(X) + np.random.normal(0, s, n)

    truth = np.full(84, False)
    truth[[0, 20, 40, 41, 42, 43]] = True

    return X, y, truth


def datagen_bm1(n: int, p:int, s: float):
    X, y_, truth = datagen_cm1(n, p, 0)
    y = np.array(
        [np.random.binomial(1, norm.cdf(i))
         for i in y_]
    )

    return X, y, truth

def datagen_bm2(n: int, p:int, s: float):
    X, y_, truth = datagen_cm2(n, 0, 0)
    y = np.array(
        [np.random.binomial(1, norm.cdf(i))
         for i in y_]
    )

    return X, y, truth