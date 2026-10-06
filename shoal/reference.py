"""Analytic bioeconomic reference points of the aggregate SHOAL dynamics (Phase 7).

Aggregate model (x = X/K, per step, harvest then regrowth):
    x' = y + g(y),  y = x (1 - F),   g(y) = r y (1 - y) * phi(y)
    phi = 1 (no threshold) or (y - a)/(1 - a) (critical depensation)
F in [0, F_max] is the fishing mortality; F_max = sum_i h_i / C when every agent fishes a
distinct cell holding the average stock. Spatial effects (search, diffusion, local richness)
are ignored -> the planner value is an *approximate* upper bound.
"""
import numpy as np

from .env import EnvParams, agent_types


def g(x, p: EnvParams):
    out = p.r * x * (1 - x)
    if p.threshold:
        out = out * (x - p.a) / (1 - p.a)
    return out


def msy(p: EnvParams):
    xs = np.linspace(0, 1, 100001)
    gs = g(xs, p)
    i = np.argmax(gs)
    return dict(x_msy=xs[i], msy_per_step=gs[i] * p.n_cells)


def open_access(p: EnvParams):
    """Selfish agents fish a cell iff h_i s_c > c: stock is driven to c / h_max (Gordon 1954),
    unless capacity is too small to get there: x_cap solves r(1-x)phi(x) = F_max."""
    h, _ = agent_types(p)
    h = np.asarray(h)
    x_cost = p.cost / h.max()
    fmax = h.sum() / p.n_cells
    xs = np.linspace(1e-4, 1, 100001)
    per_capita = g(xs, p) / xs
    above = xs[per_capita >= fmax]
    x_cap = above.max() if above.size else 0.0  # largest equilibrium under full effort
    x_oa = max(x_cost, x_cap)
    collapse = p.threshold and x_oa < p.a
    return dict(x_oa=x_oa, x_cost=x_cost, x_cap=x_cap, f_max=fmax, predicts_collapse=bool(collapse))


def planner(p: EnvParams, n_x=2001, n_f=101):
    """Finite-horizon DP maximising undiscounted group catch from x0 (upper bound used to
    normalise group harvest) and the net (catch - effort cost) optimum."""
    h, need = agent_types(p)
    h = np.asarray(h)
    fmax = h.sum() / p.n_cells
    xs = np.linspace(0, 1, n_x)
    fs = np.linspace(0, fmax, n_f)
    Y = xs[:, None] * (1 - fs[None, :])
    Xn = np.clip(Y + g(Y, p), 0, 1)
    catch = xs[:, None] * fs[None, :] * p.n_cells
    # effort cost: cheapest agents first (largest h) -> approximate by mean cost per unit F
    effort = fs[None, :] * p.n_cells / h.mean()  # number of fishing actions
    res = {}
    for name, R in [("harvest", catch), ("net", catch - p.cost * effort)]:
        V = np.zeros(n_x)
        for _ in range(p.horizon):
            Q = R + np.interp(Xn, xs, V)
            V = Q.max(1)
        res[name] = float(np.interp(p.x0, xs, V))
    # steady-state optimal stock (infinite horizon, from last Q policy)
    pol_x = xs[np.argmax(Q, 1)]
    # Equality reference: planner allocates catch proportional to needs (if heterogeneous)
    alloc = np.asarray(need) / np.asarray(need).sum()
    gini_ref = np.abs(alloc[:, None] - alloc[None]).sum() / (2 * len(alloc) * alloc.sum())
    return dict(planner_harvest=res["harvest"], planner_net=res["net"],
                planner_gini_need_prop=float(gini_ref) if p.heterogeneity else 0.0)


def threshold_basin(p: EnvParams, steps=None):
    """Check: without harvest, from x < a the stock never recovers within the horizon."""
    if not p.threshold:
        return dict(recovers_below_a=True)
    steps = steps or p.horizon
    x = p.a * 0.95
    for _ in range(steps):
        x = x + g(x, p)
    return dict(recovers_below_a=bool(x >= p.a), x_end_from_095a=float(x))


def all_reference_points(p: EnvParams):
    return {**msy(p), **open_access(p), **planner(p), **threshold_basin(p)}
