"""Pilot retuning of the Allee threshold A (Gate 5: full config must not be trivially collapsed)."""
import json
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shoal import env as E  # noqa: E402
from shoal.evaluate import eval_population  # noqa: E402
from shoal.ppo import TrainCfg, train_seeds  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
res = {}
for a in [0.1, 0.15]:
    for name in ["full"]:
        p = E.config(name, a=a, collapse_level=a)
        cfg = TrainCfg(num_updates=500)
        nets, _ = train_seeds(p, "ippo", cfg, [0, 1, 2])
        m = eval_population(nets, p, "ippo", cfg, jax.random.PRNGKey(99))
        res[f"{name}_a{a}"] = {k: v.tolist() for k, v in m.items()}
        print(a, name, {k: [round(x, 3) for x in v.tolist()] for k, v in m.items()
                        if k in ("final_stock", "collapse", "group_harvest")}, flush=True)
json.dump(res, open(OUT / "pilot_threshold.json", "w"), indent=1)
