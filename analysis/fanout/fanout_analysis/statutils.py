"""Statistical helpers for the paired sender-race analysis."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return (float(lo), float(hi))


def mcnemar_exact(b, c):
    """Exact McNemar via binomial on discordant pairs (b: A-only wins, c: B-only)."""
    nb = b + c
    if nb == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), nb, 0.5, alternative="two-sided").pvalue)


def hodges_lehmann(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.median(np.subtract.outer(a, b)))


def mann_whitney(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def fdr_adjust(pvals):
    if len(pvals) == 0:
        return []
    return list(multipletests(pvals, method="fdr_bh")[1])


def cochrans_q(W):
    """W: n x k 0/1 array. Returns (Q, df, p)."""
    W = np.asarray(W, float)
    n, k = W.shape
    Lj = W.sum(axis=0)
    Li = W.sum(axis=1)
    G = W.sum()
    denom = k * G - (Li ** 2).sum()
    if denom == 0:
        return (0.0, k - 1, 1.0)
    Q = (k - 1) * (k * (Lj ** 2).sum() - G ** 2) / denom
    p = float(stats.chi2.sf(Q, k - 1))
    return (float(Q), k - 1, p)


def paired_bootstrap_winrate(W, B=10000, seed=0):
    """W: DataFrame (triggers x senders) of 0/1 land. Resample triggers (rows)
    with replacement; track win-rate and rank per sender. rank 1 = highest rate."""
    rng = np.random.default_rng(seed)
    cols = list(W.columns)
    arr = W.to_numpy()
    n = arr.shape[0]
    phat = arr.mean(axis=0)
    rates = np.empty((B, len(cols)))
    ranks = np.empty((B, len(cols)))
    for i in range(B):
        idx = rng.integers(0, n, n)
        r = arr[idx].mean(axis=0)
        rates[i] = r
        ranks[i] = (-r).argsort().argsort() + 1
    out = {}
    for j, c in enumerate(cols):
        out[c] = {
            "p_hat": float(phat[j]),
            "p_lo": float(np.percentile(rates[:, j], 2.5)),
            "p_hi": float(np.percentile(rates[:, j], 97.5)),
            "rank_median": int(np.median(ranks[:, j])),
            "rank_lo": int(np.percentile(ranks[:, j], 2.5)),
            "rank_hi": int(np.percentile(ranks[:, j], 97.5)),
        }
    return out


def bradley_terry(win_counts, max_iter=1000, tol=1e-9):
    """win_counts: dict[(i,j)] = times i beat j. Returns dict item->ability (log scale, mean 0)."""
    items = sorted({x for pair in win_counts for x in pair})
    idx = {it: k for k, it in enumerate(items)}
    n = len(items)
    w = np.zeros(n)
    M = np.zeros((n, n))
    for (i, j), c in win_counts.items():
        w[idx[i]] += c
        M[idx[i], idx[j]] += c
        M[idx[j], idx[i]] += c
    p = np.ones(n)
    for _ in range(max_iter):
        p_new = np.zeros(n)
        for i in range(n):
            denom = sum(M[i, j] / (p[i] + p[j]) for j in range(n) if j != i and M[i, j] > 0)
            p_new[i] = w[i] / denom if denom > 0 else p[i]
        if (p_new > 0).all():
            p_new /= p_new.prod() ** (1.0 / n)
        if np.max(np.abs(np.log(p_new + 1e-12) - np.log(p + 1e-12))) < tol:
            p = p_new
            break
        p = p_new
    ability = np.log(p + 1e-12)
    ability -= ability.mean()
    return {it: float(ability[idx[it]]) for it in items}


def percentile_summary(series, name):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return {"name": name, "n": 0, "p10": float("nan"), "p50": float("nan"),
                "p90": float("nan"), "p99": float("nan"), "iqr": float("nan"), "max": float("nan")}
    return {"name": name, "n": int(len(s)),
            "p10": float(s.quantile(.10)), "p50": float(s.quantile(.50)),
            "p90": float(s.quantile(.90)), "p99": float(s.quantile(.99)),
            "iqr": float(s.quantile(.75) - s.quantile(.25)), "max": float(s.max())}


def grouped_summary(df, group_col, metric_col):
    rows = [{"group": g, **percentile_summary(sub[metric_col], g)}
            for g, sub in df.groupby(group_col)]
    return pd.DataFrame(rows)
