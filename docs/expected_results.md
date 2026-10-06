# Annex L — Expected-results document (pre-registered)

> **Frozen 2026-10-06 before any run other than the pilot.** Predictions are unchanged since; observed
> values were appended at the end after the runs (Sect. L.8).
> Pilot facts used (selfish IPPO, 3 seeds, 500 updates): degenerate final stock ≈ 0.12 (depleted,
> x_MSY = 0.5); full config collapse 3/3 (also at A = 0.1, 0.15). Mechanism hyper-parameters untuned.
> Values are honest first estimates, not targets. Confidence is low for every mechanism.

## L.1 Assumptions
H1a, H1b, H1c, H3 assumed true; open-access stock x_oa = c/h_max (0.067 homogeneous, 0.04 heterogeneous)
< A = 0.2 ⇒ selfish collapse with threshold; planner harvest bound 60 (+T/full) and 98 (degenerate/+H/+Z),
approximate (aggregate model, ignores spatial friction).

## L.2 Conditions
| Condition | Type | Implementation | Tuning budget | Known strengths / weaknesses |
|---|---|---|---|---|
| IPPO selfish | reference learner | `shoal/ppo.py`, shared params | 0 (defaults) | depletes (pilot) |
| MAPPO selfish | reference learner | centralised critic | 0 | same rewards as IPPO |
| + IA (α=β=0.1) | reward | reimpl. of Hughes 2018 on reward traces | 0 | no punishment channel → weaker than original |
| + punishment | action | zap radius 1, freeze 5 | 0 | reactive (H1b) |
| + SVO (15°–75°) | reward | reimpl. of McKee 2020 | 0 | prosocial agents may restrain |
| random / greedy / quota | scripted | `shoal/policies.py` | — | quota ≈ planner proxy |

## L.3 Anticipated biases
| Bias | Affected | Direction | Magnitude | Mitigation |
|---|---|---|---|---|
| Re-implementations weaker than originals | IA, SVO | ↓ effect | −30 to −100 % of effect | report as such |
| Untuned mechanism weights | IA, SVO | ± | large | sensitivity left as future work |
| Short training (≈ 1M env steps / seed) | all | ↓ cooperation | medium | learning curves reported |
| Parameter sharing | all | ↓ heterogeneity of behaviour | small | agent id in obs |
| Bimodal outcomes near threshold | +T, full | ± | high | report collapse rates |

## L.4 Expected numbers (10 seeds)

**Collapse rate (final X/K < 0.2)**
| Condition | degenerate | +T | full | Rank (full) | Confidence |
|---|---|---|---|---|---|
| IPPO | 90 % [60,100] (depletion) | 100 % [80,100] | 100 % [80,100] | 4 | high |
| MAPPO | 90 % [50,100] | 100 % [70,100] | 100 % [70,100] | 4 | med |
| + IA | 60 % [20,100] | 80 % [40,100] | 90 % [50,100] | 2 | low |
| + punishment | 80 % [40,100] | 100 % [70,100] | 100 % [70,100] | 3 | low |
| + SVO | 50 % [10,100] | 70 % [30,100] | 80 % [40,100] | 1 | low |
| quota (scripted) | 0 % | 0 % | 0 % | — | high |

**Group harvest / planner bound**
| Condition | degenerate | full |
|---|---|---|
| IPPO | 0.65 [0.5, 0.8] | 0.43 [0.38, 0.5] |
| + IA | 0.7 [0.5, 0.85] | 0.5 [0.38, 0.75] |
| + punishment | 0.6 [0.4, 0.8] | 0.42 [0.3, 0.55] |
| + SVO | 0.7 [0.5, 0.85] | 0.5 [0.38, 0.75] |

**Equality (1 − Gini)** — degenerate: all learners ≥ 0.9; +H: raw ≈ 0.65–0.75 for selfish (capacity ratio 5:1),
IA raises raw equality but lowers need-normalised equality relative to IPPO (H1a).

## L.5 Expected trends
| Trend | Expected | Source |
|---|---|---|
| Ranking, degenerate (final stock) | SVO ≈ IA > punishment ≈ IPPO ≈ MAPPO | Hughes 2018, McKee 2020 |
| Ranking, full | preference-based > punishment ≥ selfish; differences compressed by collapse floor | H1b |
| Kendall τ (deg vs full, final stock) | 0.4 [−0.2, 0.8] | — |
| IA raw-equality gain in +H vs degenerate | larger raw gain, need-normalised gain ≤ 0 | H1a |
| Punishment frequency, +Z vs degenerate | lower | H1c |
| Removing T from full | collapse rate drops strongly (≥ 50 points) | threshold analysis |
| Removing H from full | raw equality rises (≥ +0.15) | structural inequality |
| Removing Z from full | punishment frequency rises | H1c |
| Threshold sweep (IPPO, +T) | collapse ≈ 0 % for A=0.03, rising to 100 % for A ≥ 0.07–0.1 | x_oa = 0.067 |
| No learner exceeds planner bound | holds | planner optimum |
| Planner Gini under +H | 0.33 > 0 → raw Gini biased | equality reference |
| Probes: persistence | IA/SVO restraint decays after selfish fine-tuning (Δstock < control); punishment-trained populations also decay | H3 |
| Probes: conditionality | selfish learners: ≈ 0; at least one mechanism differs significantly from IPPO | H3 |

## L.6 Link to claims
| Claim | Supporting | Refuting |
|---|---|---|
| Cl2 | IPPO final stock < x_MSY in ≥ 80 % seeds; IA > IPPO on stock and equality (Holm p<0.05) | IA ≤ IPPO |
| Cl3 | τ < 0.5 and ≥ 1 significant reversal | τ ≥ 0.8, no reversal |
| Cl4 | each removal significant in predicted direction | no effect |
| Cl5 | ≥ 2 mechanisms separated by a probe | no difference |

## L.7 Comparison after runs → filled in `docs/results.md`
## L.8 Addenda (dated, after freeze)
| Date | Change | Justification | Before/after results? |
|---|---|---|---|
| 2026-10-06 | none to predictions; Observed columns appended below | — | after |

## L.4/L.5 Observed (appended after the runs; predictions above unchanged)
| Item | Predicted | Observed | Status |
|---|---|---|---|
| Collapse, degenerate (IPPO/MAPPO/IA/punish/SVO) | 90/90/60/80/50 % | 100/100/100/100/100 % (depletion, X_T/K<0.2) | inside intervals; mechanisms over-predicted |
| Collapse, +T | 100/100/80/100/70 % | 100 % for all | inside intervals |
| Collapse, full | 100/100/90/100/80 % | 100 % for all | inside intervals |
| Quota collapse | 0 % | 0 % | confirmed |
| Harvest/planner, degenerate (IPPO/IA/punish/SVO) | 0.65/0.70/0.60/0.70 | 0.68/0.66/0.68/0.72 | inside |
| Harvest/planner, full | 0.43/0.50/0.42/0.50 | 0.43/0.42/0.43/0.43 | inside |
| Ranking degenerate (stock) | SVO≈IA > punish≈IPPO≈MAPPO | SVO > punish ≈ IPPO > MAPPO > IA | partially |
| Kendall τ deg vs full (stock) | 0.4 [−0.2, 0.8] | 0.2 | inside |
| IA raw-equality gain in +H | larger than degenerate | −0.025 (degenerate +0.000) | refuted |
| IA need-equality gain in +H | ≤ 0 | −0.032 | confirmed |
| Punishment frequency +Z < degenerate | lower | 3.5e-6 vs 2.1e-5, n.s. (punishment unused) | untestable |
| −T: collapse drops ≥ 50 pts | yes | 0 pts (metric saturates); stock +0.09 | refuted on metric |
| −H: raw equality +0.15 | yes | +0.32 | confirmed |
| −Z: punishment frequency rises | yes | no change | refuted / untestable |
| Threshold sweep | 0 % at 0.03 → 100 % at ≥ 0.07–0.1 | 0, 0, 52, 100, 100 % | confirmed |
| No learner above planner | holds | 0 / 420 runs | confirmed |
| Planner Gini under +H | 0.33 | 0.33 | confirmed (by construction) |
| Persistence: IA/SVO decay | yes | SVO yes (p_Holm=0.001), IA no restraint to lose | partially |
| Conditionality: ≥ 1 mechanism ≠ IPPO | yes | none | refuted |
