import numpy as np
import pandas as pd

from fanout_analysis import statutils as su


def test_wilson_ci_known():
    lo, hi = su.wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo0, hi0 = su.wilson_ci(0, 912)
    assert lo0 == 0.0 and 0.002 < hi0 < 0.006


def test_mcnemar_exact_symmetry():
    assert su.mcnemar_exact(10, 0) < 0.01
    assert abs(su.mcnemar_exact(7, 7) - 1.0) < 1e-9


def test_hodges_lehmann_shift():
    a = np.array([10.0, 12, 14])
    b = np.array([1.0, 3, 5])
    assert abs(su.hodges_lehmann(a, b) - 9.0) < 1e-9


def test_fdr_adjust_monotone():
    p = [0.001, 0.01, 0.04, 0.5]
    adj = su.fdr_adjust(p)
    assert all(adj[i] <= adj[i + 1] + 1e-9 for i in range(len(adj) - 1))
    assert max(adj) <= 1.0


def test_cochrans_q_all_equal_is_nonsignificant():
    W = np.tile(np.array([1, 0, 1, 0, 1]).reshape(-1, 1), (1, 3))
    Q, dfree, p = su.cochrans_q(W)
    assert abs(Q) < 1e-9 and abs(p - 1.0) < 1e-6


def test_paired_bootstrap_winrate_ranks_strong_sender_first():
    rng = np.random.default_rng(0)
    n = 200
    a = (rng.random(n) < 0.8).astype(int)
    b = ((a == 0) & (rng.random(n) < 0.9)).astype(int)
    W = pd.DataFrame({"A": a, "B": b})
    res = su.paired_bootstrap_winrate(W, B=2000, seed=1)
    assert res["A"]["rank_median"] == 1
    assert res["A"]["p_lo"] < res["A"]["p_hat"] < res["A"]["p_hi"]


def test_percentile_summary_keys():
    s = pd.Series(range(101))
    d = su.percentile_summary(s, "x")
    assert d["p50"] == 50 and d["n"] == 101 and d["p99"] >= 98


def test_bradley_terry_orders_dominant_item_first():
    wc = {("A", "B"): 80, ("B", "A"): 20, ("A", "C"): 90, ("C", "A"): 10, ("B", "C"): 60, ("C", "B"): 40}
    ab = su.bradley_terry(wc)
    assert ab["A"] > ab["B"] > ab["C"]
