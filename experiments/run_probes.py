"""Exp 4 (RQ3 / Cl5): imposed-vs-learned probes on the populations trained in run_protocol.py.

Persistence    : fine-tune each trained population for FT updates with the *selfish* reward and
                 no punishment action (mech 'ippo'); measure the change of each metric. Control =
                 selfish IPPO fine-tuned the same way (pure continued-training drift).
Conditionality : focal agents 0..N/2-1 run the trained policy; the others are scripted
                 cooperators (quota) or defectors (greedy). Score = focal fishing rate with
                 defectors - with cooperators (>0: exploits/retaliates, <0: compensates/restrains).
"""
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoal import env as E  # noqa: E402
from shoal.evaluate import eval_population  # noqa: E402
from shoal.ppo import MECHS, TrainCfg, train_seeds  # noqa: E402
from run_protocol import CFG, ENV_KW, RUNS, SEEDS  # noqa: E402

FT = 150
PROBE_CONFIGS = ["degenerate", "full"]
PROBE_MECHS = ["ippo", "ia", "svo", "punish"]


def load(c, mech):
    return pickle.load(open(RUNS / f"main__{c}__{mech}.pkl", "rb"))


def main():
    out_f = ROOT / "results" / "probes.json"
    res = json.load(open(out_f)) if out_f.exists() else {}
    for c in PROBE_CONFIGS:
        p = E.config(c, **ENV_KW)
        for mech in MECHS:
            key = f"{c}__{mech}"
            if key in res or not (RUNS / f"main__{c}__{mech}.pkl").exists():
                continue
            nets = load(c, mech)
            r = {}
            # conditionality (all mechanisms)
            focal = jnp.arange(p.n_agents) < p.n_agents // 2
            for bg in ["cooperators", "defectors"]:
                m = eval_population(nets, p, mech, CFG, jax.random.PRNGKey(77), background=bg, focal_mask=focal)
                r[f"cond_{bg}"] = {k: v.tolist() for k, v in m.items()}
            # persistence (fine-tune selfishly)
            if mech in PROBE_MECHS:
                cfg = CFG.with_(num_updates=FT)
                ft_nets, _ = train_seeds(p, "ippo", cfg, [s + 1000 for s in SEEDS], init=nets)
                m = eval_population(ft_nets, p, "ippo", cfg, jax.random.PRNGKey(12345))
                r["after_selfish_ft"] = {k: v.tolist() for k, v in m.items()}
            res[key] = r
            json.dump(res, open(out_f, "w"))
            print("probe done", key, flush=True)


if __name__ == "__main__":
    main()
