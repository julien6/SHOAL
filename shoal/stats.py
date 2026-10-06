"""Statistical protocol (Phase 8): IQM + bootstrap CIs (rliable-style), Mann-Whitney with Holm
correction, Kendall tau between rankings, balanced two-way ANOVA for feature x mechanism."""
import numpy as np
from scipy import stats as st


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return x[lo:hi].mean() if hi > lo else x.mean()


def bootstrap_ci(x, fn=iqm, n=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    boots = [fn(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def mwu(a, b):
    """Two-sided Mann-Whitney U p-value (returns 1.0 for identical constant samples)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, a[0]) and np.allclose(b, b[0]) and np.isclose(a[0], b[0]):
        return 1.0
    return float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def kendall(x, y):
    t = st.kendalltau(x, y)
    return float(t.statistic), float(t.pvalue)


def two_way_anova(y):
    """y: array (A, B, n) balanced. Returns dict of F, p for A, B and AxB."""
    y = np.asarray(y, float)
    A, B, n = y.shape
    gm = y.mean()
    ma, mb, mab = y.mean((1, 2)), y.mean((0, 2)), y.mean(2)
    ss_a = B * n * ((ma - gm) ** 2).sum()
    ss_b = A * n * ((mb - gm) ** 2).sum()
    ss_ab = n * ((mab - ma[:, None] - mb[None, :] + gm) ** 2).sum()
    ss_e = ((y - mab[..., None]) ** 2).sum()
    df_a, df_b, df_ab, df_e = A - 1, B - 1, (A - 1) * (B - 1), A * B * (n - 1)
    mse = ss_e / df_e if ss_e > 0 else 1e-12
    out = {}
    for name, ss, df in [("A", ss_a, df_a), ("B", ss_b, df_b), ("AxB", ss_ab, df_ab)]:
        F = (ss / df) / mse
        out[name] = dict(F=float(F), p=float(st.f.sf(F, df, df_e)))
    return out
