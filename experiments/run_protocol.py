"""Main protocol (Phase 8/9): env configurations x learning configurations x seeds.

Exp 1/2 : 5 env configs x 5 mechanisms x 10 seeds   (RQ1, RQ2 / Cl2, Cl3)
Exp 2b  : feature ablations from full (full-H, full-Z, full-T) x {ippo, ia, punish} (Cl4)
Exp 3   : threshold sensitivity, IPPO on +T, A in {0.03 .. 0.2} (around x_oa=0.067)         (Phase 7 prediction)
Reference policies: random / greedy / quota on every env config.

Resumable: each cell is written to results/runs/<exp>__<config>__<mech>.json (+ .pkl nets).
Usage: python experiments/run_protocol.py [main|ablation|threshold|scripted|all|ext_threshold|ext_agents]
"""
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoal import env as E  # noqa: E402
from shoal.evaluate import eval_population, eval_scripted  # noqa: E402
from shoal.ppo import MECHS, TrainCfg, train_seeds  # noqa: E402
from shoal.reference import all_reference_points  # noqa: E402

RUNS = ROOT / "results" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(10))
CFG = TrainCfg(num_updates=500)
ENV_KW = json.load(open(ROOT / "results" / "frozen_env.json")) if (ROOT / "results" / "frozen_env.json").exists() else {}

MAIN_CONFIGS = ["degenerate", "+H", "+Z", "+T", "full"]
ABLATIONS = ["full-H", "full-Z", "full-T"]
ABLATION_MECHS = ["ippo", "ia", "punish"]
THRESHOLDS = [0.03, 0.05, 0.07, 0.1, 0.2]
# extension (added 2026-10-06 after the main analysis; exploratory, not pre-registered)
EXT_THRESHOLDS = [0.05, 0.07, 0.1]
EXT_MECHS = ["ia", "svo"]


def run_cell(exp, cname, mech, p, seeds=SEEDS, cfg=CFG):
    f = RUNS / f"{exp}__{cname}__{mech}.json"
    if f.exists():
        return
    t = time.time()
    nets, curves = train_seeds(p, mech, cfg, seeds)
    jax.block_until_ready(curves)
    dt = time.time() - t
    m = eval_population(nets, p, mech, cfg, jax.random.PRNGKey(12345))
    out = dict(exp=exp, config=cname, mech=mech, seeds=seeds, seconds=dt,
               env_params=p.__dict__, train_cfg=cfg.__dict__,
               metrics={k: v.tolist() for k, v in m.items()},
               curves={k: np.asarray(v).mean(0).tolist() for k, v in curves.items()})
    pickle.dump(jax.device_get(nets), open(f.with_suffix(".pkl"), "wb"))
    json.dump(out, open(f, "w"))
    print(f"[{time.strftime('%H:%M:%S')}] {exp} {cname:10s} {mech:6s} {dt:6.0f}s  "
          f"stock={np.mean(m['final_stock']):.3f} collapse={np.mean(m['collapse']):.2f} "
          f"H={np.mean(m['group_harvest']):.1f} E={np.mean(m['equality']):.3f}", flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("scripted", "all"):
        f = RUNS / "scripted.json"
        if not f.exists():
            res = {}
            for c in MAIN_CONFIGS + ABLATIONS:
                p = E.config(c, **ENV_KW)
                res[c] = dict(reference=all_reference_points(p))
                for pol in ["random", "greedy", "quota"]:
                    res[c][pol] = {k: v.tolist() for k, v in eval_scripted(pol, p, jax.random.PRNGKey(7)).items()}
            json.dump(res, open(f, "w"), default=float)
            print("scripted done", flush=True)
    if what in ("main", "all"):
        for c in MAIN_CONFIGS:
            for mech in MECHS:
                run_cell("main", c, mech, E.config(c, **ENV_KW))
    if what in ("ablation", "all"):
        for c in ABLATIONS:
            for mech in ABLATION_MECHS:
                run_cell("ablation", c, mech, E.config(c, **ENV_KW))
    if what in ("threshold", "all"):
        for a in THRESHOLDS:
            kw = {**ENV_KW, "a": a, "collapse_level": a}
            run_cell("threshold", f"+T_a{a}", "ippo", E.config("+T", **kw))
    if what in ("ext_threshold",):
        for a in EXT_THRESHOLDS:
            for mech in EXT_MECHS:
                kw = {**ENV_KW, "a": a, "collapse_level": a}
                run_cell("threshold", f"+T_a{a}", mech, E.config("+T", **kw))
    if what in ("ext_agents",):
        for mech in ["ippo", "svo", "ia"]:
            run_cell("agents8", "degenerate", mech, E.config("degenerate", n_agents=8, **ENV_KW))


if __name__ == "__main__":
    main()
