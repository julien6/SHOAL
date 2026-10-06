"""Vectorised episode rollouts for arbitrary (possibly mixed learned/scripted) policies."""
import jax
import jax.numpy as jnp

from . import env as E


def run_episode(key, p: E.EnvParams, act_fn):
    """act_fn(key, state, obs) -> actions (N,). Returns (episode metrics, stock trajectory)."""
    k0, key = jax.random.split(key)
    s0 = E.reset(k0, p)

    def body(carry, _):
        s, k = carry
        k, ka, ks = jax.random.split(k, 3)
        a = act_fn(ka, s, E.observe(s, p))
        s, r, d, info = E.step(ks, s, a, p)
        return (s, k), (s.stock.mean(), info["fished"], r)

    (sT, _), (traj, fished, rews) = jax.lax.scan(body, (s0, key), None, p.horizon)
    m = E.episode_metrics(sT, p)
    m["return"] = rews.sum(0).mean()
    return m, traj


def run_episodes(key, p, act_fn, n):
    return jax.vmap(lambda k: run_episode(k, p, act_fn))(jax.random.split(key, n))
