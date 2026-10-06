"""Phase 5.3 feasibility pilot ("kill test"): selfish IPPO, 3 seeds, degenerate vs full."""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shoal import env as E  # noqa: E402
from shoal.evaluate import eval_population  # noqa: E402
from shoal.ppo import TrainCfg, train_seeds  # noqa: E402
from shoal.reference import all_reference_points  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
U = int(sys.argv[1]) if len(sys.argv) > 1 else 500

res = {}
for name in ["degenerate", "full"]:
    p = E.config(name)
    cfg = TrainCfg(num_updates=U)
    t = time.time()
    nets, curves = train_seeds(p, "ippo", cfg, [0, 1, 2])
    jax.block_until_ready(curves)
    dt = time.time() - t
    m = eval_population(nets, p, "ippo", cfg, jax.random.PRNGKey(99))
    env_steps = 3 * U * cfg.num_envs * cfg.num_steps
    res[name] = dict(
        seconds=dt, env_steps_per_s=env_steps / dt,
        final_stock=m["final_stock"].tolist(), collapse=m["collapse"].tolist(),
        group_harvest=m["group_harvest"].tolist(), fish_rate=m["fish_rate"].tolist(),
        curve_stock=np.asarray(curves["stock"]).mean(0)[:: max(1, U // 20)].round(3).tolist(),
        reference=all_reference_points(p))
    print(name, json.dumps({k: v for k, v in res[name].items() if k != "reference"}), flush=True)

json.dump(res, open(OUT / "pilot.json", "w"), indent=1, default=float)
