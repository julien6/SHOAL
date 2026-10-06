"""Record and animate episodes of trained populations (seed 0) and scripted policies.
Writes figures/replays/*.gif.   Usage: .venv/bin/python experiments/replay.py"""
import pickle
import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoal import env as E  # noqa: E402
from shoal.ppo import TrainCfg, env_for, policy_fn  # noqa: E402
from shoal.reference import open_access  # noqa: E402
from shoal.render import record_episode, save_animation  # noqa: E402

RUNS = ROOT / "results" / "runs"
OUT = ROOT / "figures" / "replays"
OUT.mkdir(parents=True, exist_ok=True)
CFG = TrainCfg(num_updates=500)


def trained_policy(config, mech, seed=0):
    p = env_for(E.config(config), mech)
    nets = pickle.load(open(RUNS / f"main__{config}__{mech}.pkl", "rb"))
    actor = jax.tree_util.tree_map(lambda x: x[seed], nets.actor)
    return p, policy_fn(actor, p, mech, CFG)


def record(config, mech, key=0, seed=0):
    p, act = trained_policy(config, mech, seed)
    return p, record_episode(jax.random.PRNGKey(key), p, act)


if __name__ == "__main__":
    LAB = dict(ippo="IPPO", svo="+SVO", ia="+IA", punish="+Punish", mappo="MAPPO")
    for config, mech in [("degenerate", "ippo"), ("+Z", "ippo"), ("+Z", "svo"), ("full", "ippo"), ("full", "svo")]:
        p, traj = record(config, mech)
        name = f"{config.replace('+', 'plus')}_{mech}.gif"
        save_animation(traj, p, OUT / name, title=f"{config}, {LAB[mech]}", x_oa=open_access(p)["x_oa"])
        print("wrote", OUT / name, flush=True)
