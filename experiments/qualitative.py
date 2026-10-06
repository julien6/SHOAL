"""Phase 9 qualitative analysis: rolls out the trained populations (results/runs/main__*.pkl) and
records stock trajectories, spatial harvest maps, per-agent harvests and zone use.
Writes results/qualitative.npz."""
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoal import env as E  # noqa: E402
from shoal.ppo import TrainCfg, env_for, policy_fn  # noqa: E402

RUNS = ROOT / "results" / "runs"
CFG = TrainCfg(num_updates=500)
CONFIGS = ["degenerate", "+H", "+Z", "+T", "full"]
MECHS = ["ippo", "mappo", "ia", "punish", "svo"]
N_EP = 4


def rollout(actor, p, mech, key):
    act = policy_fn(actor, p, mech, CFG)
    zm = E.zone_masks(p)

    def ep(k):
        k0, kk = jax.random.split(k)
        s = E.reset(k0, p)

        def body(c, _):
            s, kk = c
            kk, ka, ks = jax.random.split(kk, 3)
            a = act(ka, s, E.observe(s, p))
            s2, r, d, info = E.step(ks, s, a, p)
            cell = s2.pos[:, 0] * p.grid + s2.pos[:, 1]
            hmap = jnp.zeros(p.n_cells).at[cell].add(info["catch"])
            inzone = zm.reshape(p.n_agents, -1)[jnp.arange(p.n_agents), cell]
            return (s2, kk), (s2.stock.mean(), hmap, info["catch"], inzone)
        _, (traj, hmap, catch, inzone) = jax.lax.scan(body, (s, kk), None, p.horizon)
        return traj, hmap.sum(0), catch.sum(0), inzone.mean(0), catch.cumsum(0)
    return jax.vmap(ep)(jax.random.split(key, N_EP))


out = {}
for c in CONFIGS:
    for m in MECHS:
        p = env_for(E.config(c), m)
        nets = pickle.load(open(RUNS / f"main__{c}__{m}.pkl", "rb"))
        S = jax.tree_util.tree_leaves(nets.actor)[0].shape[0]
        f = jax.jit(jax.vmap(lambda a, k: rollout(a, p, m, k)))
        traj, hmap, catch, inzone, cum = f(nets.actor, jax.random.split(jax.random.PRNGKey(5), S))
        k = f"{c}|{m}"
        out[k + "|traj"] = np.asarray(traj)                    # (S, EP, T)
        out[k + "|hmap"] = np.asarray(hmap).mean((0, 1))       # (C,)
        out[k + "|catch"] = np.asarray(catch)                  # (S, EP, N)
        out[k + "|inzone"] = np.asarray(inzone).mean((0, 1))   # (N,)
        print("done", k, flush=True)
np.savez_compressed(ROOT / "results" / "qualitative.npz", **out)
