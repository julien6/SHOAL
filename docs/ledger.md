# C1 — Translation ledger: Ostrom SES framework / MoHuB → SHOAL POSG

POSG ⟨N, S, {Aᵢ}, T, {Oᵢ}, {Rᵢ}, γ⟩. Status: **K** kept, **A** approximated, **L** lost.
MARL constraint responsible for a loss: **M** Markov state, **R** scalar reward, **H** short horizon,
**N** small number of agents, **F** fixed action/observation spaces. Code pointers refer to `shoal/env.py`.

## Compact table (main-body version, ≈ ¼ page)

| SES component (Ostrom 2009) | SHOAL element | Status | Constraint | Expected consequence |
|---|---|---|---|---|
| RS Resource system (size, productivity, equilibrium) | 6×6 grid, logistic regrowth r, carrying capacity K=36 | K | — | — |
| RS5 Productivity / RS7 predictability of dynamics | deterministic growth + diffusion; no environmental noise | A | — | overstates predictability; no stochastic collapse |
| RU Resource units (mobility, growth) | per-cell stock, diffusion (fish mobility) | K | — | — |
| Critical thresholds (Clark depensation) | global factor (X/K−A)/(1−A) (feature **T**) | K | — | irreversible collapse below A |
| A1 Number of actors | n = 6 | A | N | weak open-access pressure from crowd size |
| A2 Socioeconomic attributes / heterogeneity | types θᵢ=(hᵢ, nᵢ) (feature **H**) | K | — | — |
| A8 Dependence on the resource | seasonal need nᵢ, shortfall penalty | A | M, R | needs as reward penalty; season harvest in obs |
| A5 Leadership / A6 social capital / trust | — | L | F | no communication channel |
| A7 Knowledge of SES / mental models | public stock assessment X/K in obs; local 5×5 view | A | M | agents know the aggregate stock |
| GS5 Operational rules (boundary / access) | access zones, hard (admissible actions) or soft (fine) (feature **Z**) | K | — | — |
| GS6 Collective-choice rules (rule making) | — | L | F, H | institutions are exogenous, agents cannot change them |
| GS8 Monitoring and sanctioning | soft zones: detection p, fine; punishment action | A | R | sanctions are reward deductions, no reputation |
| I1 Harvesting levels | Schaefer catch hᵢ·s_c, effort cost c | K | — | — |
| I2 Information sharing / I3 deliberation | — | L | F | — |
| I4 Conflicts | punishment (zap, freeze 5 steps) | A | — | Pérolat-style |
| I7 Self-organising activities | only through learned policies | A | H | — |
| O1 Social performance (efficiency, equity) | group harvest / planner bound; 1−Gini raw and need-normalised | K | — | — |
| O2 Ecological performance | final stock / K, collapse, time to collapse | K | — | — |
| ECO / related ecosystems, markets | — | L | N | constant price (=1), no market feedback |

## MoHuB decision steps (Schlüter et al. 2017)

| Step | SES theory usually assumed | SHOAL translation | Status | Constraint |
|---|---|---|---|---|
| Perceive | bounded, local, social learning of others' outcomes | local window + others' positions + public stock | A | M |
| Evaluate | utility incl. needs, norms, other-regarding prefs | scalar reward: catch − cost − need shortfall (+ IA / SVO term) | A | R |
| Decide | heuristics, satisficing, habit, imitation | stochastic policy optimised by PPO (shared parameters) | A | — |
| Learn | experiential / social learning across years | gradient learning over ~1M steps; no imitation | A | H |
| Social: norms, identity, institutional change | rule creation, identity | — | L | F, H |

## Variables augmenting the state (Markov property)
| Variable | Why | Where |
|---|---|---|
| season harvest of agent i | seasonal need breaks Markov otherwise | obs scalar 3, global state |
| season phase t mod L | need penalty timing | obs scalar 4 |
| frozen timer | punishment effect | obs scalar 5 |
| reward traces eᵢ (IA, SVO) | shaped reward depends on history | **not** in obs — known approximation (as in Hughes 2018) |
| episode step | truncation | **not** observed; truncation bootstrapped → no end-game |

## H2 classification (translatability)
- **Directly translatable** (reward/action change): inequity aversion, SVO, costly punishment, access rules (hard), harvesting capacity.
- **Translatable with state augmentation**: subsistence needs (season harvest), sanctions with memory (frozen timer), monitoring (violation flags / fines).
- **Resistant**: collective-choice rule change, communication / deliberation, trust and reputation, identity, market feedbacks.

Coverage check (Cl1): all four first-tier components (RS, RU, A, GS) and I/O are mapped; every
SHOAL variable (`EnvParams` fields) appears above. Losses are justified by a named constraint. ✅
