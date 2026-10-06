"""Evaluation of trained populations + the conditionality probe (Phase 8, RQ3)."""
import jax
import jax.numpy as jnp
import numpy as np

from . import env as E
from . import policies as P
from .ppo import env_for, policy_fn
from .reference import msy
from .rollout import run_episodes


def background_fn(name, p):
    if name == "cooperators":
        x = float(msy(p)["x_msy"])
        return lambda k, s, o: P.quota(k, s, p, x)
    if name == "defectors":
        return lambda k, s, o: P.greedy(k, s, p)
    raise ValueError(name)


def eval_population(nets, p0, mech, cfg, key, n_episodes=16, background=None, focal_mask=None):
    """nets: batched over seeds. Returns dict metric -> np.array (S,) averaged over episodes.
    background: None (all learned) or 'cooperators' / 'defectors' (scripted co-players on agents
    where focal_mask is False). Also returns the focal agents' fishing rate and harvest."""
    p = env_for(p0, mech)
    if focal_mask is None:
        focal_mask = jnp.arange(p.n_agents) % 2 == 0 if background else jnp.ones(p.n_agents, bool)
    bg = background_fn(background, p) if background else None

    def one_seed(actor, k):
        learned = policy_fn(actor, p, mech, cfg)

        def act(k, s, o):
            k1, k2 = jax.random.split(k)
            a = learned(k1, s, o)
            return jnp.where(focal_mask, a, bg(k2, s, o)) if bg else a

        def ep(k):
            k0, key = jax.random.split(k)
            s = E.reset(k0, p)

            def body(c, _):
                s, kk = c
                kk, ka, ks = jax.random.split(kk, 3)
                a = act(ka, s, E.observe(s, p))
                s, r, d, info = E.step(ks, s, a, p)
                return (s, kk), info["fished"]
            (sT, _), fished = jax.lax.scan(body, (s, key), None, p.horizon)
            m = E.episode_metrics(sT, p)
            m["focal_fish_rate"] = (fished * focal_mask).sum() / (focal_mask.sum() * p.horizon)
            m["focal_harvest"] = (sT.total_harvest * focal_mask).sum() / focal_mask.sum()
            return m
        ms = jax.vmap(ep)(jax.random.split(k, n_episodes))
        return jax.tree_util.tree_map(lambda x: x.mean(0), ms)

    S = jax.tree_util.tree_leaves(nets.actor)[0].shape[0]
    out = jax.jit(jax.vmap(one_seed))(nets.actor, jax.random.split(key, S))
    return {k: np.asarray(v) for k, v in out.items()}


def eval_scripted(name, p, key, n_runs=10, n_episodes=16):
    fn = dict(random=lambda k, s, o: P.random(k, s, p),
              greedy=lambda k, s, o: P.greedy(k, s, p),
              quota=lambda k, s, o: P.quota(k, s, p, float(msy(p)["x_msy"])))[name]
    m, _ = jax.jit(lambda k: run_episodes(k, p, fn, n_runs * n_episodes))(key)
    return {k: np.asarray(v).reshape(n_runs, n_episodes).mean(1) for k, v in m.items()}
