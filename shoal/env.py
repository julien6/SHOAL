"""SHOAL environment: a pure-JAX grid fishery (parallel multi-agent API).

Ostrom SES anchors (see docs/ledger.md):
  Resource system / units : per-cell stock s_c in [0, 1], logistic regrowth, diffusion,
                            optional *global* critical depensation threshold A.
  Actors                  : types theta_i = (h_i capacity, n_i seasonal need).
  Governance              : access zones (hard = inadmissible, soft = monitored + fined),
                            optional costly punishment action.
  Interactions/outcomes   : catch, costs, fines -> rewards and metrics.

Every feature can be switched off; with heterogeneity, zones and threshold off we get the
Harvest-like *degenerate* configuration.
"""
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

# actions
STAY, UP, DOWN, LEFT, RIGHT, FISH, PUNISH = range(7)
N_ACTIONS = 7
_MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0]], dtype=jnp.int32)


@dataclass(frozen=True)
class EnvParams:
    grid: int = 6
    n_agents: int = 6
    horizon: int = 300
    season_len: int = 50
    r: float = 0.03             # intrinsic growth rate per step
    diffusion: float = 0.1      # fraction of a cell's stock spread to its 4 neighbours per step
    x0: float = 0.7             # initial stock (fraction of carrying capacity, per cell)
    cost: float = 0.02          # cost of one fishing action (Gordon-Schaefer effort cost)
    # --- SES feature H: actor heterogeneity (capacity + need) ---
    heterogeneity: bool = False
    h_mean: float = 0.3
    h_hi: float = 0.5
    h_lo: float = 0.1
    need_mean: float = 1.5      # seasonal subsistence need (only used when heterogeneity on)
    need_lambda: float = 0.5    # shortfall penalty per unit of unmet need
    # --- SES feature Z: access zones ---
    zones: bool = False
    zone_mode: str = "hard"     # "hard" | "soft"
    monitor_p: float = 0.5      # soft zones: detection probability
    zone_fine: float = 0.5
    # --- SES feature T: global critical depensation threshold ---
    threshold: bool = False
    a: float = 0.2              # Allee threshold (fraction of K)
    # --- punishment capability (learning configuration, Perolat 2017) ---
    punishment: bool = False
    punish_cost: float = 0.2
    punish_fine: float = 1.0
    freeze_steps: int = 5
    # --- misc ---
    obs_radius: int = 2
    collapse_level: float = 0.2   # X/K below which an episode is counted as collapsed/depleted

    @property
    def n_cells(self):
        return self.grid * self.grid

    @property
    def n_zones(self):
        return max(1, self.n_agents // 2)

    def with_(self, **kw):
        return replace(self, **kw)


def config(name: str, **kw) -> EnvParams:
    """Named environment configurations of the protocol."""
    flags = {
        "degenerate": dict(),
        "+H": dict(heterogeneity=True),
        "+Z": dict(zones=True),
        "+T": dict(threshold=True),
        "full": dict(heterogeneity=True, zones=True, threshold=True),
        "full-H": dict(zones=True, threshold=True),
        "full-Z": dict(heterogeneity=True, threshold=True),
        "full-T": dict(heterogeneity=True, zones=True),
    }[name]
    return EnvParams(**{**flags, **kw})


class EnvState(NamedTuple):
    stock: jnp.ndarray          # (G, G)
    pos: jnp.ndarray            # (N, 2) int
    frozen: jnp.ndarray         # (N,) int steps remaining
    season_harvest: jnp.ndarray  # (N,)
    t: jnp.ndarray              # () int
    # episode accumulators (for metrics)
    total_harvest: jnp.ndarray  # (N,)
    n_punish: jnp.ndarray       # () punishments issued
    n_violation: jnp.ndarray    # () out-of-zone fishing actions (soft)
    n_fish: jnp.ndarray         # (N,) fishing actions
    shortfall: jnp.ndarray      # (N,) accumulated unmet need
    t_collapse: jnp.ndarray     # () first t with X/K < collapse_level (horizon if never)


# ---------------------------------------------------------------- agent types / zones
def agent_types(p: EnvParams):
    """h_i, n_i. Heterogeneous: even agents high capacity, odd low; need proportional to capacity
    (bigger boats, bigger crews), so need-proportional (unequal) harvests are legitimate."""
    idx = jnp.arange(p.n_agents)
    if p.heterogeneity:
        h = jnp.where(idx % 2 == 0, p.h_hi, p.h_lo)
        n = p.need_mean * h / p.h_mean
    else:
        h = jnp.full((p.n_agents,), p.h_mean)
        n = jnp.full((p.n_agents,), p.need_mean)
    return h.astype(jnp.float32), n.astype(jnp.float32)


def zone_masks(p: EnvParams):
    """(N, G, G) bool: cells agent i may fish. Zones = vertical strips, agent i -> zone i//2."""
    cols = jnp.arange(p.grid)
    if not p.zones:
        return jnp.ones((p.n_agents, p.grid, p.grid), bool)
    zone_of_col = jnp.minimum(cols * p.n_zones // p.grid, p.n_zones - 1)
    agent_zone = jnp.arange(p.n_agents) // 2 % p.n_zones
    m = zone_of_col[None, :] == agent_zone[:, None]  # (N, G)
    return jnp.broadcast_to(m[:, None, :], (p.n_agents, p.grid, p.grid))


# ---------------------------------------------------------------- dynamics
def growth(stock, p: EnvParams):
    """Per-cell logistic growth; with threshold, modulated by the *global* depensation factor
    (X/K - a)/(1 - a). Summed over a uniform field this equals the aggregate law
    g(x) = r x (1 - x) (x - a)/(1 - a) (Clark's critical depensation)."""
    g = p.r * stock * (1.0 - stock)
    if p.threshold:
        x = stock.mean()
        g = g * (x - p.a) / (1.0 - p.a)
    return g


def diffuse(stock, d):
    """Mass-conserving diffusion: each cell sends d/4 to each existing 4-neighbour."""
    G = stock.shape[0]
    out = stock * d / 4.0
    up = jnp.zeros_like(stock).at[:-1, :].set(out[1:, :])     # flows from row i+1 to row i
    down = jnp.zeros_like(stock).at[1:, :].set(out[:-1, :])
    left = jnp.zeros_like(stock).at[:, :-1].set(out[:, 1:])
    right = jnp.zeros_like(stock).at[:, 1:].set(out[:, :-1])
    n_nb = (jnp.full((G, G), 4.0)
            - (jnp.arange(G) == 0)[:, None] - (jnp.arange(G) == G - 1)[:, None]
            - (jnp.arange(G) == 0)[None, :] - (jnp.arange(G) == G - 1)[None, :])
    return stock - out * n_nb + up + down + left + right


# ---------------------------------------------------------------- API
def reset(key, p: EnvParams):
    G, N = p.grid, p.n_agents
    zm = zone_masks(p)
    # random start cell inside own zone
    logits = jnp.where(zm.reshape(N, -1), 0.0, -1e9)
    cells = jax.random.categorical(key, logits, axis=-1)
    pos = jnp.stack([cells // G, cells % G], -1).astype(jnp.int32)
    z = jnp.zeros((N,), jnp.float32)
    return EnvState(
        stock=jnp.full((G, G), p.x0, jnp.float32), pos=pos, frozen=jnp.zeros((N,), jnp.int32),
        season_harvest=z, t=jnp.array(0, jnp.int32), total_harvest=z,
        n_punish=jnp.array(0.0), n_violation=jnp.array(0.0), n_fish=z, shortfall=z,
        t_collapse=jnp.array(p.horizon, jnp.int32))


def step(key, state: EnvState, actions, p: EnvParams):
    """Returns (state, rewards (N,), done (), info dict). Order: move -> fish -> punish -> regrow."""
    G, N = p.grid, p.n_agents
    h, need = agent_types(p)
    zm = zone_masks(p)
    active = state.frozen == 0
    actions = jnp.where(active, actions, STAY)
    if not p.punishment:
        actions = jnp.where(actions == PUNISH, STAY, actions)

    # 1. movement
    pos = jnp.clip(state.pos + _MOVES[actions], 0, G - 1)
    cell = pos[:, 0] * G + pos[:, 1]

    # 2. fishing (Schaefer catch h_i * s_c, shared proportionally if demand exceeds the cell)
    fishing = actions == FISH
    in_zone = zm.reshape(N, -1)[jnp.arange(N), cell]
    if p.zones and p.zone_mode == "hard":
        fishing_eff = fishing & in_zone          # inadmissible outside own zone -> no-op
    else:
        fishing_eff = fishing
    violation = fishing & ~in_zone if (p.zones and p.zone_mode == "soft") else jnp.zeros((N,), bool)
    flat = state.stock.reshape(-1)
    demand = jnp.zeros(G * G).at[cell].add(jnp.where(fishing_eff, h, 0.0))
    scale = jnp.minimum(1.0, 1.0 / jnp.maximum(demand, 1e-8))
    catch = jnp.where(fishing_eff, h * flat[cell] * scale[cell], 0.0)
    harvested = jnp.zeros(G * G).at[cell].add(catch)
    flat = jnp.maximum(flat - harvested, 0.0)
    rew = catch - p.cost * fishing.astype(jnp.float32)

    # soft zones: monitored with prob monitor_p, fined
    k1, _ = jax.random.split(key)
    caught = violation & (jax.random.uniform(k1, (N,)) < p.monitor_p)
    rew = rew - p.zone_fine * caught

    # 3. punishment: zap every other agent within Chebyshev radius 1
    punishing = actions == PUNISH
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).max(-1)
    hit = punishing[:, None] & (dist <= 1) & ~jnp.eye(N, dtype=bool)   # (punisher, target)
    hit_any = hit.any(0)
    rew = rew - p.punish_fine * hit.sum(0) - p.punish_cost * punishing
    frozen = jnp.where(hit_any, p.freeze_steps, jnp.maximum(state.frozen - 1, 0))

    # 4. regrowth (global threshold uses post-harvest stock) + diffusion
    stock = flat.reshape(G, G)
    g = growth(stock, p)
    stock = jnp.clip(diffuse(stock + g, p.diffusion), 0.0, 1.0)

    # 5. seasons / needs (season harvest is in the observation -> Markov, see ledger)
    season_h = state.season_harvest + catch
    t = state.t + 1
    season_end = (t % p.season_len) == 0
    short = jnp.maximum(need - season_h, 0.0)
    if p.heterogeneity:
        rew = rew - jnp.where(season_end, p.need_lambda * short, 0.0)
    shortfall = state.shortfall + jnp.where(season_end, short, 0.0)
    season_h = jnp.where(season_end, 0.0, season_h)

    xk = stock.mean()
    t_collapse = jnp.where((xk < p.collapse_level) & (state.t_collapse == p.horizon), t, state.t_collapse)
    new = EnvState(stock=stock, pos=pos, frozen=frozen, season_harvest=season_h, t=t,
                   total_harvest=state.total_harvest + catch,
                   n_punish=state.n_punish + punishing.sum(),
                   n_violation=state.n_violation + violation.sum(),
                   n_fish=state.n_fish + fishing_eff, shortfall=shortfall, t_collapse=t_collapse)
    done = t >= p.horizon
    info = dict(catch=catch, growth=g.sum(), fished=fishing_eff, punished=punishing, hit=hit_any)
    return new, rew.astype(jnp.float32), done, info


# ---------------------------------------------------------------- observations
def obs_dim(p: EnvParams):
    w = 2 * p.obs_radius + 1
    return 3 * w * w + 9 + p.n_agents


def observe(state: EnvState, p: EnvParams):
    """Egocentric (2R+1)^2 window x {stock, other agents, fishable(in-grid & own zone)} + scalars."""
    G, N, R = p.grid, p.n_agents, p.obs_radius
    w = 2 * R + 1
    h, need = agent_types(p)
    zm = zone_masks(p).astype(jnp.float32)
    count = jnp.zeros((G, G)).at[state.pos[:, 0], state.pos[:, 1]].add(1.0)
    pad = lambda a: jnp.pad(a, R)

    def one(i):
        y, x = state.pos[i, 0], state.pos[i, 1]
        sl = lambda a: jax.lax.dynamic_slice(pad(a), (y, x), (w, w))
        others = count.at[y, x].add(-1.0)
        win = jnp.stack([sl(state.stock), sl(others), sl(zm[i])], -1).reshape(-1)
        scal = jnp.array([
            h[i] / p.h_hi, need[i] / (p.need_mean * p.h_hi / p.h_mean),
            jnp.where(p.heterogeneity, state.season_harvest[i] / need[i], 0.0),
            (state.t % p.season_len) / p.season_len,
            state.frozen[i] / p.freeze_steps,
            y / (G - 1), x / (G - 1),
            state.stock.mean(),            # public stock assessment
            float(p.punishment)])
        return jnp.concatenate([win, scal, jax.nn.one_hot(i, N)])

    return jax.vmap(one)(jnp.arange(N)).astype(jnp.float32)


def global_state_dim(p: EnvParams):
    return 2 * p.n_cells + p.n_agents + 2


def global_state(state: EnvState, p: EnvParams):
    """Centralised-critic input (MAPPO)."""
    _, need = agent_types(p)
    count = jnp.zeros((p.grid, p.grid)).at[state.pos[:, 0], state.pos[:, 1]].add(1.0)
    return jnp.concatenate([state.stock.reshape(-1), count.reshape(-1),
                            state.season_harvest / need,
                            jnp.array([(state.t % p.season_len) / p.season_len, state.stock.mean()])]
                           ).astype(jnp.float32)


# ---------------------------------------------------------------- episode metrics
def gini(v):
    v = jnp.maximum(v, 0.0)
    n = v.shape[-1]
    diff = jnp.abs(v[..., :, None] - v[..., None, :]).sum((-1, -2))
    return diff / (2.0 * n * jnp.maximum(v.sum(-1), 1e-8))


def episode_metrics(state: EnvState, p: EnvParams):
    _, need = agent_types(p)
    th = state.total_harvest
    return dict(
        group_harvest=th.sum(),
        final_stock=state.stock.mean(),
        collapse=(state.stock.mean() < p.collapse_level).astype(jnp.float32),
        t_collapse=state.t_collapse.astype(jnp.float32),
        equality=1.0 - gini(th),
        equality_need=1.0 - gini(th / need),
        punish_rate=state.n_punish / (p.horizon * p.n_agents),
        violation_rate=state.n_violation / (p.horizon * p.n_agents),
        fish_rate=state.n_fish.mean() / p.horizon,
        shortfall=state.shortfall.sum(),
    )
