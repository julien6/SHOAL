"""Non-learning reference policies (privileged access to the state): random, greedy, quota."""
import jax
import jax.numpy as jnp

from .env import FISH, N_ACTIONS, STAY, agent_types, zone_masks, EnvParams, EnvState


def _richest_move(state: EnvState, p: EnvParams):
    """Action stepping each agent toward the richest fishable cell (distance-discounted)."""
    G, N = p.grid, p.n_agents
    zm = zone_masks(p)
    yy, xx = jnp.meshgrid(jnp.arange(G), jnp.arange(G), indexing="ij")
    dy = yy[None] - state.pos[:, 0, None, None]
    dx = xx[None] - state.pos[:, 1, None, None]
    score = jnp.where(zm, state.stock[None] - 0.02 * (jnp.abs(dy) + jnp.abs(dx)), -1e9)
    tgt = jnp.argmax(score.reshape(N, -1), -1)
    ty, tx = tgt // G - state.pos[:, 0], tgt % G - state.pos[:, 1]
    vert = jnp.where(ty < 0, 1, 2)    # UP / DOWN
    horiz = jnp.where(tx < 0, 3, 4)   # LEFT / RIGHT
    return jnp.where((ty == 0) & (tx == 0), STAY, jnp.where(jnp.abs(ty) >= jnp.abs(tx), vert, horiz))


def _threshold_policy(state: EnvState, p: EnvParams, s_min, rel=0.0):
    """Fish if the local cell is fishable, its stock exceeds s_min and is at least `rel` times the
    best fishable stock; otherwise move toward the richest cell."""
    zm = zone_masks(p)
    y, x = state.pos[:, 0], state.pos[:, 1]
    local = state.stock[y, x]
    ok = zm[jnp.arange(p.n_agents), y, x]
    best = jnp.where(zm, state.stock[None], 0.0).reshape(p.n_agents, -1).max(-1)
    mv = _richest_move(state, p)
    return jnp.where(ok & (local > s_min) & (local >= rel * best), FISH, mv)


def greedy(key, state, p):
    """Myopic exploiter: fish while catch exceeds the effort cost (open access), but move on
    when the local cell is less than 20% as rich as the best one."""
    h, _ = agent_types(p)
    return _threshold_policy(state, p, p.cost / h, rel=0.2)


def quota(key, state, p, s_target=0.5):
    """Planner-inspired: harvest only cells above the MSY stock level (x_MSY ~ 0.5 for logistic)."""
    return _threshold_policy(state, p, s_target)


def random(key, state, p):
    return jax.random.randint(key, (p.n_agents,), 0, 6)  # no punishment


SCRIPTED = dict(random=random, greedy=greedy, quota=quota)
