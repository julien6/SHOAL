# SHOAL — minimal proof of concept

*Social-ecological Harvesting with Ostrom-grounded Access and Limits* (provisional name).
A pure-JAX multi-agent fishery for testing how MARL social mechanisms behave when the commons has
SES structure: **H** actor heterogeneity (capacity + need), **Z** access zones, **T** a global critical
depensation threshold. All three off = Harvest-like *degenerate* configuration.

The roadmap is `roadmap.md`; its status after this PoC is in `docs/roadmap_status.md`.

## Layout
| Path | Content |
|---|---|
| `shoal/env.py` | environment (reset/step/observe, metrics), configs `degenerate, +H, +Z, +T, full, full-H/Z/T` |
| `shoal/reference.py` | analytic reference points: MSY, open-access stock, planner DP bound, threshold basin |
| `shoal/policies.py` | scripted references: random, greedy (open access), quota (MSY) |
| `shoal/ppo.py` | vectorised IPPO / MAPPO + inequity aversion, punishment, SVO; all seeds in one `jit(vmap)` |
| `shoal/evaluate.py` | evaluation + conditionality probe (scripted co-players) |
| `shoal/render.py` | episode recording + frame drawing + GIF/MP4 animation (stock heatmap, agent types, zones, fishing rings, punishment zaps, global-stock gauge with threshold) |
| `shoal/stats.py` | IQM, bootstrap CI, Mann–Whitney + Holm, Kendall τ, two-way ANOVA |
| `tests/test_env.py` | invariant tests (stock balance, threshold basin, zones, punishment, determinism, depletion) |
| `experiments/` | `pilot.py`, `run_protocol.py`, `run_probes.py`, `analyze.py`, `figure1.py` |
| `docs/` | ledger (C1), ODD+D, literature/novelty/gaps, decision log, Annex L (frozen), results, paper draft, mock review |
| `results/`, `figures/` | raw per-cell JSON + trained nets, summary tables, figures |

## Reproduce (CPU only, ~4 h on a 14-core laptop)
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest -q tests
.venv/bin/python experiments/pilot.py               # Phase 5.3 kill test (~4 min)
.venv/bin/python experiments/run_protocol.py all    # Exp 1-3 + scripted references
.venv/bin/python experiments/run_probes.py          # Exp 4 (persistence, conditionality)
.venv/bin/python experiments/analyze.py && .venv/bin/python experiments/figure1.py
```
Throughput: ≈ 25–30k env-steps/s (×6 agents) for a batch of 10 seeds on CPU; one protocol cell
(10 seeds × 500 PPO updates × 2048 env-steps ≈ 10M env-steps) ≈ 5–6 min.

## Rendering
```python
from shoal import env as E, policies as P
from shoal.render import record_episode, draw_frame, save_animation
p = E.config("full")
traj = record_episode(jax.random.PRNGKey(0), p, lambda k, s, o: P.greedy(k, s, p))
fig, _ = draw_frame(traj, t=30, p=p, title="full, greedy, t=30")   # one frame
save_animation(traj, p, "episode.gif")                             # or .mp4 (ffmpeg)
```
`experiments/replay.py` renders trained populations (seed 0) to `figures/replays/*.gif`
(degenerate/IPPO, +Z/IPPO vs +Z/SVO, full/IPPO vs full/SVO). The annotated walkthrough used in the
paper is `paper/figures/fig_walkthrough.pdf` (`python paper/make_figures.py walk`).
