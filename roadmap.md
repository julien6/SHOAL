# Recipe: SHOAL — from project to AAMAS 2028 submission

> **Adapted from** `aamas_paper_recipe.md` for the project *"Apprendre à partager une ressource :
> cognition sociale des agents dans un socio-écosystème simulé"*.
> Tick the boxes as you go and keep this file in the repository: it is the paper's logbook.
> **Gates** are checkpoints: move on only if the criterion is met; otherwise pivot or reduce scope.
>
> 🔁 **Main adaptation.** The original recipe assumes the contribution is an *algorithm* that must beat
> baselines. Here the contribution is an **environment + translation method + evaluation suite**
> (the project explicitly rules out a new algorithm). Consequently, throughout this file:
> - "baselines" means (a) **existing environments** SHOAL is compared to, and (b) **learning
>   configurations** (selfish IPPO/MAPPO, social mechanisms, reference policies) run *inside* SHOAL;
> - claims are about **fidelity, backward compatibility, discriminative power and interpretability of
>   outcomes**, not about "our method wins";
> - deductive validation relies on **bioeconomic reference points** (open-access equilibrium,
>   maximum sustainable yield, planner optimum, collapse threshold), not on convergence theorems.
>
> 📌 **Paper appendix rule (unchanged).** Only the main body (references included, appendices excluded)
> is reviewed. Standing test: *can a reviewer who never opens the appendices understand, judge and be
> convinced?* For SHOAL this means in particular: the SES→Markov-game translation table, the resource
> dynamics equations and Figure 1 **must** be in the main body.
>
> ⚠️ `SHOAL` is a **provisional** acronym (see Phase 4). `{{…}}` fields remain to be filled.

> 🧪 **PoC status (2026-10-06):** a minimal proof of concept of Phases 0–11 is implemented — see
> `README.md`, `docs/roadmap_status.md` (phase-by-phase status, next steps) and `docs/results.md`
> (verdicts). Ticked boxes below are those completed by the PoC; reduced-scope items are noted there.

---

## Project identity card

| Field | Value |
|---|---|
| Working title | SHOAL: A Social-Ecological Testbed for Learning to Share a Renewable Resource *(provisional)* |
| Contribution acronym | SHOAL *(provisional — uniqueness check pending, Phase 4)* |
| Target venue / track | AAMAS 2028 – main track (fallback: ALA workshop / JAAMAS / TMLR) |
| Area / sub-community | Primary: *Learning and Adaptation* (sequential social dilemmas, MARL). Secondary: *Modeling and Simulation of Societies*; *Coordination, Organizations, Institutions, Norms and Ethics* — **check the 2028 area list** |
| Abstract deadline | {{AAMAS 2028 CfP — not yet published; expected ≈ late Sep 2027}} |
| Paper deadline | {{expected ≈ early Oct 2027 (AAMAS 2027's was 1 Oct 2026)}} |
| Conference | {{May 2028, location TBA (IFAAMAS bid call prefers the Americas)}} |
| Author response phase | {{dates}} |
| Authors and roles | REDACTED – lead / supervision; {{M2 student – state of the art}}; {{M1 student – implementation}}; {{co-supervisors}} |
| Repository (private) | {{link}} |
| Anonymized repository | {{Anonymous GitHub link}} |
| Initial informal document | `projet-1-reformule.docx` (project description, 13 references) |

---

## Contents

0. [Scoping and backward planning](#phase-0--scoping-and-backward-planning)
1. [Mapping the community](#phase-1--mapping-the-community)
2. [Review of the field](#phase-2--review-of-the-field)
3. [Confronting the topic with the literature](#phase-3--confronting-the-topic-with-the-literature)
4. [Hypotheses, contributions and name](#phase-4--hypotheses-contributions-and-name)
5. [Designing the contribution](#phase-5--designing-the-contribution)
6. [Claims](#phase-6--claims)
7. [Deductive validation](#phase-7--deductive-validation)
8. [Experimental protocol](#phase-8--experimental-protocol)
9. [Execution, results and discussion](#phase-9--execution-results-and-discussion)
10. [Writing](#phase-10--writing)
11. [Adversarial review](#phase-11--adversarial-review)
12. [Submission](#phase-12--submission)
13. [After submission](#phase-13--after-submission)
- [Annexes](#annexes-templates-to-fill-in)

---

## Phase 0 — Scoping and backward planning

**Goal**: set the course before investing time and compute. The project starts as an M2 (state of the
art) + M1 (implementation) student project; the paper requires going **beyond** the student scope
(theoretical reference points, full factorial protocol, imposed-vs-learned analysis).

### Actions
- [ ] Read the AAMAS 2028 CfP as soon as it is out (expected ≈ summer 2027) and note:
  - [ ] format (ACM sigconf), page limit (8 pages in recent editions — **verify**), whether references count
  - [ ] double-blind rules (the environment's public repo must not de-anonymize: release after notification)
  - [ ] abstract deadline distinct from paper deadline
  - [ ] supplementary material / appendices / code policy
  - [ ] LLM-use policy
  - [ ] author response phase
  - [ ] whether prior non-archival workshop versions (e.g. ALA 2027) are allowed
- [ ] Decide on a **preliminary version at ALA 2027** (workshop at AAMAS 2027, Hanoi, May 2027): early feedback from the sequential-social-dilemma community, if non-archival
- [ ] Clarify authorship and roles with the students upfront (log it in Annex A)
- [ ] Build the backward plan (below)
- [x] Estimate the compute budget (below)

### Compute budget (first estimate)

`seeds × env configurations × learning configurations × run duration`

| Factor | Minimal credible | Full |
|---|---|---|
| Env configurations (3 binary SES features) | 5 (degenerate, +H, +Z, +T, full) | 8 (full 2³ factorial) |
| Learning configurations | 5 (IPPO, MAPPO, +inequity aversion, +punishment, +SVO diversity) | 7 (+ social influence, + norm learning from sanctions) |
| Seeds | 10 | 15 |
| Runs | 250 | 840 |
| Run duration | {{to measure in the pilot, Phase 5.3}} | |

- [x] Measure steps/s of the PettingZoo implementation in the pilot; if too slow, decide (Annex A) between vectorization (SuperSuit), reduced grid / horizon, or a JAX port (JaxMARL-style)

### Backward plan (paper deadline D ≈ early Oct 2027 — to be confirmed)

| Milestone | Target date |
|---|---|
| Mapping and review completed (reuse of the M2 state of the art) | Dec 2026 |
| Novelty test vs. Melting Pot / Harvest / CPR / GovSim done | Dec 2026 |
| Gaps, hypotheses, claims v0 frozen | Jan 2027 |
| Feasibility pilot conclusive (degenerate config reproduces depletion) | Feb 2027 |
| Optional ALA 2027 submission (preliminary) | {{ALA deadline, ≈ Feb 2027}} |
| Implementation complete + invariant tests | Apr 2027 |
| Protocol frozen + expected-results document frozen (before any run) | May 2027 |
| Final experiments completed | ~D-21 (mid Sep 2027) |
| Full draft written | ~D-14 |
| Mock reviews returned | ~D-10 |
| Abstract and keywords submitted | {{abstract date}} |
| Submission | D-1 at the latest |

### ✅ Gate 0
- [x] The compute budget covers the **minimal credible** protocol (250 runs). Otherwise: drop the social-influence mechanism first (most expensive: needs a model of other agents), then reduce the factorial to one-at-a-time + full.

---

## Phase 1 — Mapping the community

**Goal**: SHOAL sits at the boundary of **two communities that rarely cite each other**: MARL / sequential
social dilemmas (DeepMind-centred cluster) and social-ecological systems (SES) modelling (ecological
economics, JASSS, *Ecology and Society*). The paper must speak MARL first (reviewers) while being credible
to SES modellers.

### Actions — building the graph
- [ ] Seed papers (start from the 13 project references, then expand): Leibo et al. 2017; Pérolat et al. 2017; Hughes et al. 2018; Jaques et al. 2019; McKee et al. 2020; Köster et al. 2022; Vinitsky et al. 2023; Ostrom 2009; Schlüter et al. 2017; Müller et al. 2013; An 2012; Bourceret et al. 2021; Terry et al. 2021
- [ ] Add the **environment/benchmark** cluster: Melting Pot (Leibo et al. 2021; Agapiou et al. 2022 — Melting Pot 2.0), Sequential Social Dilemma Games codebase (Vinitsky et al.), GovSim (Piatti et al. 2024 — LLM agents in a fishery), AI Economist (Zheng et al.) {{verify references}}
- [ ] Add the **SES + learning** cluster: experiential-learning / RL agents in fisheries ABMs with thresholds (e.g. Lindkvist & Norberg, *Ecological Economics* 2014 {{verify}}), small-scale fisheries ABMs
- [ ] Add the **bioeconomics** foundations: Gordon 1954 / Schaefer (open access), Clark, *Mathematical Bioeconomics* (critical depensation)
- [ ] Add **institutions**: Ostrom 1990 (design principles), McGinnis & Ostrom 2014 (updated SES framework)
- [ ] Extract authors / references (DBLP, Semantic Scholar, OpenAlex); build the co-citation graph (VOSviewer / Connected Papers)
- [ ] Identify bridges between the MARL and SES clusters (few: Köster, Leibo, Hughes on the MARL side; Schlüter on the SES side)
- [ ] Cross-reference with recent AAMAS area chairs for *Learning and Adaptation* and *Modeling and Simulation of Societies*, ALA / MABS organizers

### What to extract
- [ ] Canonical vocabulary (Annex B): *sequential social dilemma*, *common-pool resource (CPR)*, *appropriation*, *mixed-motive*, *intrinsic motivation*, *social value orientation*, *sanctioning*; SES side: *resource system / resource units / actors / governance system*, *first/second-tier variables*
- [ ] Formalisms: partially observable Markov game (POSG) on the MARL side; ODD+D / MoHuB on the SES side
- [ ] Standard metrics: utilitarian efficiency, equality (1 − Gini), sustainability, peace (Leibo 2017 / Pérolat 2017 / Hughes 2018)
- [ ] Standard protocols: Melting Pot focal/background population evaluation; Gorsane et al. 2022 evaluation guidelines
- [ ] Open debates: "is a reward-shaped preference social cognition?"; realism vs. tractability in ABMs

> ⚠️ **Ethics and effectiveness.** The graph serves to speak the community's language and to choose
> area and keywords, **not** to target reviewers.

### Deliverables
- [ ] Annotated graph ({{file link}})
- [ ] Must-cite list (Annex B)
- [ ] Two-sided glossary MARL ↔ SES (Annex B) — it also feeds the translation ledger (Phase 5)

---

## Phase 2 — Review of the field

**Goal**: reuse and deepen the M2 state of the art, organized around its three questions:
(Q1) how environmental models represent human decisions; (Q2) how learning agents behave around a shared
resource; (Q3) the role of rules, sanctions and institutions. Add a fourth: (Q4) **which environments and
benchmarks exist for CPR dilemmas, and what do they abstract away?**

### Actions
- [ ] Reproducible queries (Scholar / Semantic Scholar): `"common-pool resource" AND "reinforcement learning"`, `"sequential social dilemma"`, `"social-ecological system" AND "agent-based" AND learning`, `fishery AND "multi-agent reinforcement learning"`
- [ ] Forward / backward snowballing from Pérolat 2017, Hughes 2018, Schlüter 2017, Bourceret 2021
- [ ] Skim 80–120 papers, read 30–40 in depth with reading sheets (Annex C)
- [ ] Extract "limitations / future work" sections of Pérolat 2017, Hughes 2018, Leibo 2017, Melting Pot papers (self-declared gaps, e.g. homogeneous agents, simplistic regrowth)
- [x] Taxonomy along 4 axes:
  1. **Where the social mechanism lives**: reward (inequity aversion, SVO, influence) / action space (punishment) / observation (sanction signals) / environment rules (institutions)
  2. **Agent heterogeneity**: none / preferences / capabilities / needs
  3. **Resource dynamics**: local density regrowth / logistic / with depensation threshold
  4. **Grounding**: ad hoc / SES framework / empirical calibration
- [x] Environment comparison table (feeds Phase 7)

### Deliverables
- [x] Taxonomy + comparison table
- [ ] Synthesis "state of the art + open questions" (3–5 pages), ending with the project's synthesis question: *which behavioural hypotheses of SES models can be translated into learning mechanisms, and which resist translation?*

---

## Phase 3 — Confronting the topic with the literature

### 3.1 Reformulating the problem
- [ ] Assess:
  - **Alignment**: MARL for cooperation and sustainability is active (Melting Pot contest, norm-learning papers, LLM-agent commons such as GovSim); SES modellers call for better behavioural grounding (An 2012; Schlüter 2017). {{add 2025–2027 evidence}}
  - **Originality**: not any single feature (see novelty test) but (i) the explicit, documented SES→Markov-game translation with its losses, (ii) the combination of capability/need heterogeneity, access institutions and a global depensation threshold with analytic reference points, (iii) a protocol separating imposed preferences from learned behaviour
  - **Impact**: MARL researchers get a testbed where conclusions on social mechanisms can be checked against SES-relevant structure; SES modellers get a documented bridge to learning agents
- [ ] Reformulate in community vocabulary

> **Problem in one sentence**: Conclusions about which social mechanisms help learning agents sustain a
> common-pool resource are drawn in environments that omit structural features that SES research deems
> decisive (actor heterogeneity, access rules, irreversible collapse), and the abstraction choices behind
> these environments are never made explicit.
>
> **Problem in one paragraph**: {{…}}

- [ ] **Novelty test** — actively search for the paper that would kill the idea. Known threats to address explicitly:
  - **Melting Pot `commons_harvest` substrates** (open / closed / partnership): the *closed* variant already has restricted-access areas; regrowth depends on local apple density, so local depletion is already irreversible → SHOAL must **not** claim access zones or local collapse as new per se
  - **Harvest / CPR gridworlds** (Hughes 2018; Pérolat 2017): punishment and depletion already present
  - **McKee 2020**: heterogeneity already present — but of *preferences* (SVO), not of capabilities / needs
  - **GovSim** (2024): fishery for LLM agents — different agent class, but same "sustainability" framing
  - {{SES ABMs with RL agents and thresholds (Lindkvist & Norberg 2014?)}}
  - Result: {{none / paper X → how we differ}}

### 3.2 Decomposition into gaps

| Gap | Statement | Evidence of the gap |
|---|---|---|
| G1 — Structural gap | CPR environments used in MARL omit, or include only in isolation, the SES features that drive outcomes in Ostrom's framework: capability/need heterogeneity of actors, access institutions, global irreversible collapse | Pérolat 2017; Hughes 2018; Leibo 2021 (Melting Pot); Ostrom 2009 |
| G2 — Translation gap | SES models mostly rely on ad hoc decision rules; there is no systematic account of which behavioural theories can be translated into learning mechanisms under MARL constraints (Markov state, scalar reward, short horizons) | An 2012; Schlüter 2017; Müller 2013; Bourceret 2021 |
| G3 — Evaluation gap | Social mechanisms are each evaluated in their own environment with their own metrics, and the effect of a preference **imposed** in the reward is conflated with **learned** social behaviour | Hughes 2018; Jaques 2019; McKee 2020; Vinitsky 2023 |

- [ ] Fill in the related work × gaps matrix (Annex D)

### ✅ Gate 3
- [ ] Novelty confirmed: no existing environment covers G1 **and** G2 **and** G3
- [ ] At least two reproducible reference points: (a) known qualitative results to reproduce in SHOAL's degenerate configuration (depletion under selfish learning; inequity aversion improves cooperation), and (b) ≥ 2 social mechanisms with public implementations (e.g. SSD codebase)
- Otherwise: reposition as a "translation study" paper (G2 + G3) on top of Melting Pot substrates, or target ALA / JASSS.

---

## Phase 4 — Hypotheses, contributions and name

| Gap | Hypothesis | Sub-contribution |
|---|---|---|
| G1 | **H1a** (heterogeneity): when agents differ in harvesting capacity and subsistence need, payoff-based inequity aversion no longer improves equality as in homogeneous settings, because it penalizes structurally unequal but legitimate outcomes. **H1b** (threshold): with critical depensation, mechanisms acting *before* depletion (preference-based) outperform reactive mechanisms (punishment), which need to observe over-harvesting first. **H1c** (access): access zones act as an institutional substitute for punishment: punishment frequency and its marginal benefit drop when zones are active | **C2** — SHOAL environment (PettingZoo), with three parametric SES features that can be switched off to recover a Harvest-like *degenerate* configuration |
| G2 | **H2**: SES behavioural hypotheses (MoHuB categories) split into three classes — directly translatable (reward or action modifications), translatable at the cost of state augmentation (memory, need satisfaction, reputation), and resistant (institutional change, communication, identity) under Markov / scalar-reward / short-horizon constraints | **C1** — Translation ledger: a documented mapping from Ostrom's SES variables and MoHuB decision steps to POSG elements, each entry stating what is kept, approximated or lost; environment described with ODD+D |
| G3 | **H3**: the effects of mechanisms on sustainability and equality differ when measured with SES-aware metrics and reference points, and imposed preferences can be distinguished from learned conditional behaviour by persistence and conditionality probes | **C3** — Evaluation suite: metrics (reused from [3, 4] + normalized by analytic reference points), statistical protocol, persistence and conditionality probes, reference results for IPPO/MAPPO and social mechanisms |

> **Overall hypothesis H**: Grounding a CPR environment in Ostrom's SES framework, through an explicit
> translation ledger, changes the conclusions MARL research draws about social mechanisms — some
> mechanisms that help in homogeneous open-access gridworlds become ineffective or unfair once actor
> heterogeneity, access rules and irreversible collapse are present.

### Name and acronym
- [ ] Candidates: **SHOAL** (fish school; *Social-ecological Harvesting with Ostrom-grounded Access and Limits*), OSTRA (*Ostrom-STRuctured Appropriation*), {{other}}
- [ ] Pronounceable / not taken (Google Scholar, GitHub, arXiv, PyPI) / evokes the core idea
- [ ] Freeze the name before the ALA submission

### Deliverable
- [ ] Traceability matrix v1 (Annex E)

---

## Phase 5 — Designing the contribution

### 5.1 Informal ideation
- [ ] One page: grid fishery, 6–8 agents, renewable stock; toy example showing that the same selfish policy is sustainable without threshold and catastrophic with threshold

### 5.2 Component description

| Component | Role | Inputs / outputs | SES anchor (Ostrom 2009) |
|---|---|---|---|
| Resource system | Grid of cells with local stock; regrowth law with optional depensation threshold | stock field → stock field | Resource system (RS), resource units (RU) |
| Actors | Agent types θᵢ = (harvesting capacity hᵢ, subsistence need nᵢ) | type → harvest rate, need term in reward | Actors (A): number, heterogeneity, dependence on the resource |
| Governance | Access zones (cells → set of allowed agents); hard variant (forbidden move) vs soft variant (allowed but sanctionable) | position, type → admissible actions / sanction | Governance system (GS): boundary rules, monitoring, sanctions |
| Interactions / outcomes | Harvest, punishment action, observations | → rewards, metrics | Interactions (I) → Outcomes (O) |
| Evaluation suite | Metrics, reference points, probes | trajectories → scores | Outcomes: ecological and social performance |

### 5.3 Feasibility pilot ("kill test")
- [x] Degenerate config (homogeneous, no zones, no threshold) + selfish IPPO, 3 seeds: resource depleted? **yes** (X/K≈0.12 vs x_MSY 0.5)
- [x] Full config: does the threshold produce collapse in some seeds and not others (non-trivial regime)? **all seeds collapse at A=0.2 (= open-access prediction); seed-level transition at A≈x_oa=0.07 (52 %), see decision log**
- [x] Throughput (steps/s) and run duration measured → update Gate 0
- If depletion is not reproduced or the full config is trivially collapsed in every run: retune dynamics before anything else.

### 5.4 Formalization
- [x] Preliminaries: POSG ⟨N, S, {Aᵢ}, T, {Oᵢ}, {Rᵢ}, γ⟩ (standard notation of Leibo 2017 / Pérolat 2017)
- [x] Resource dynamics: per-cell or aggregate stock with logistic growth and **critical depensation**, e.g. g(x) = r·x·(1 − x/K)·(x/A − 1) with Allee threshold A < K {{choose and justify; compare with local-density regrowth of Harvest}}
- [x] Heterogeneity: hᵢ in the transition; nᵢ in the reward (e.g. concave utility or shortfall penalty). **Any need defined over a season breaks the Markov property unless the running harvest is put in the state → ledger entry**
- [x] Access zones as an admissible-action function (hard) or a sanction function (soft)
- [x] Reference points defined formally (Phase 7)
- [ ] All notation used in the main body is defined in the main body

### 5.5 Translation ledger (C1) — core artefact
- [x] For each Ostrom first-tier component and the relevant second-tier variables, and for each MoHuB step (perceive / evaluate / decide): **kept / approximated / lost**, with the MARL constraint responsible (Markov state, scalar reward, horizon, number of agents) and the expected consequence
- [x] Main-body version: compact table (≈ ¼ page); full version in appendix
- [x] ODD+D description of SHOAL (appendix, referenced from the main body)

### 5.6 Schematization
- [x] Figure 1: left, Ostrom's four components; right, SHOAL's grid with types, zones, stock; arrows = ledger mapping. Understandable without the text.

### 5.7 Implementation plan
- [x] Lightweight SRS (1 page): parallel PettingZoo API, configurable features, deterministic seeding, logging of all metrics
- [ ] Building blocks: PettingZoo [13] (+ SuperSuit), training via {{BenchMARL / EPyMARL / CleanRL-style IPPO-MAPPO}}; social mechanisms reused from public code when available
- [x] Implementation of the environment, mechanisms, metrics, probes
- [x] **Invariant tests**: stock balance (Δstock = regrowth − harvest), threshold behaviour (no recovery below A within the horizon at default parameters), zone admissibility, PettingZoo `api_test` / `parallel_api_test`, seed determinism, degenerate config = Harvest-like behaviour

> 💡 Favour lightweight design and invariant tests over exhaustive UML.

### ✅ Gate 5
- [ ] Pilot shows the expected signal (depletion in degenerate config; non-trivial regime in full config)
- [x] Implementation passes invariant tests

---

## Phase 6 — Claims

### Actions
- [x] Numbered claims, each linked to a gap and measurable (Annex F, prefilled)
- [x] **Freeze claims and protocol before the final experiments** — date: 2026-10-06 – commit: 6ccb452
- [x] For each claim, plan what will be written if it is not supported (Annex F, last column)
- [x] Translate each claim into quantitative expectations (Annex L)

Summary of claims (details in Annex F):
- **Cl1** (G2, fidelity): every SES first-tier component is either represented in SHOAL or listed as lost in the ledger, with justification.
- **Cl2** (G1, backward compatibility): in the degenerate configuration, SHOAL reproduces the known qualitative results (selfish learners deplete; inequity aversion improves sustainability and equality).
- **Cl3** (G1, discriminative power): activating the SES features changes the ranking of social mechanisms on sustainability and/or equality.
- **Cl4** (G1, necessity of each feature): each feature, removed from the full configuration, significantly changes at least one metric in the direction predicted by H1a–H1c.
- **Cl5** (G3, imposed vs learned): persistence and conditionality probes separate at least two mechanisms.

> Well-formed example: *"Cl3: the Kendall rank correlation between mechanism rankings (on collapse
> rate) in the degenerate and full configurations is below 0.5, with at least one pairwise reversal
> significant after Holm correction (10 seeds)."*

---

## Phase 7 — Deductive validation

**Adaptation**: no convergence theorem is expected. Deductive validation consists of **analytic reference
points** of the resource model and **structural properties** of the environment, which (i) make metrics
interpretable and (ii) yield testable predictions.

### Actions
- [x] Derive, for the aggregate dynamics, the **reference points**:
  - collapse threshold A and its basin (no recovery below A)
  - maximum sustainable yield (MSY) and the stock level that achieves it
  - **open-access (Gordon–Schaefer-type) equilibrium**: where selfish harvesting is expected to drive the stock
  - **planner optimum**: maximal discounted group harvest (dynamic programming on the aggregate model, or a computed quota policy) → upper bound used to normalize group harvest
  - equality reference: Gini of the planner policy under heterogeneous needs (equality ≠ 0 Gini when needs differ)
- [x] Structural properties: Markov property of the state (list augmented variables), per-step complexity O(|grid| + n), observation size, episode length vs. regrowth time-scale
- [x] Make explicit the gap between the aggregate analytic model and the spatial grid (approximation)
- [ ] Each result placed right after the definition of the object; derivations in appendix; statements and intuition in the main body

### Environment comparison table (replaces the "guarantees vs baselines" table)

| Property | SHOAL | Harvest (SSD) | Commons Harvest (Melting Pot) | CPR (Pérolat 2017) | GovSim |
|---|---|---|---|---|---|
| Agent class | RL | RL | RL | RL | LLM |
| Capability / need heterogeneity | ● | ○ | {{verify}} | ○ | {{verify}} |
| Access institutions | ● (hard / soft) | ○ | ◐ (closed variant) | ○ | {{verify}} |
| Irreversible collapse | ● global, tunable threshold | ◐ local | ◐ local | ◐ local | {{verify}} |
| Analytic reference points (MSY, open access, planner) | ● | ○ | ○ | ○ | {{verify}} |
| Explicit SES grounding (Ostrom / ODD+D) | ● | ○ | ○ | ○ | ○ |
| API | PettingZoo | custom | dm_env / custom | custom | custom |
| Step complexity | {{…}} | {{…}} | {{…}} | {{…}} | {{…}} |

### Predictions to verify empirically

| Prediction | Source | Verified by | Result |
|---|---|---|---|
| Selfish IPPO drives the stock toward the open-access level; with threshold → collapse in most seeds | open-access equilibrium | Exp. 1 | |
| No learning configuration exceeds the planner bound on group harvest | planner optimum | all | |
| Collapse rate increases sharply as A approaches the open-access stock level | threshold analysis | Exp. 3 (sensitivity) | |
| Under heterogeneous needs, the planner's Gini > 0 → raw Gini is a biased fairness measure | equality reference | Exp. 2 | |

---

## Phase 8 — Experimental protocol

Dedicated section of the paper. Template in Annex G (prefilled).

### Checklist
- [x] RQs linked to claims:
  - **RQ1** (Cl2) Does SHOAL's degenerate configuration reproduce known results?
  - **RQ2** (Cl3, Cl4) How do the SES features change the effects and ranking of social mechanisms?
  - **RQ3** (Cl5) Which mechanisms produce learned, conditional behaviour rather than an imposed preference?
  - (Cl1 is established by the ledger, not by experiment)
- [x] Environment configurations: degenerate, +heterogeneity, +zones, +threshold, full (+ full 2³ factorial if budget allows); 6 and 8 agents
- [x] Learning configurations, **one social mechanism at a time** (as in the project):
  - selfish IPPO; selfish MAPPO (centralized critic)
  - + inequity aversion (Hughes 2018) — preference in reward
  - + costly punishment action, selfish reward (Pérolat 2017) — capability in action space
  - + SVO diversity (McKee 2020) — heterogeneous preferences in reward
  - optional: + social influence (Jaques 2019); + norm learning from public sanctions (Vinitsky 2023)
- [x] Reference policies (non-learning): random; greedy; planner / quota policy (Phase 7)
- [x] **Same hyperparameter tuning budget for all learning configurations**
- [x] Metrics: group harvest (normalized by planner bound); sustainability (final stock / K, collapse rate, time to collapse); equality (1 − Gini on raw harvest **and** on need-normalized harvest — justify the latter); punishment frequency; zone violations (soft variant)
- [x] Probes for imposed vs learned (RQ3):
  - **Persistence**: fine-tune without the social term (or without the punishment action) and measure metric decay
  - **Conditionality**: evaluate focal agents with scripted background co-players (cooperators / defectors), Melting Pot-style, and measure whether behaviour responds to co-players
- [x] Statistical protocol: ≥ 10 seeds; IQM + bootstrap CIs (`rliable`); Welch / Mann–Whitney with Holm correction; two-way analysis (feature × mechanism) for interactions; Gorsane et al. 2022 guidelines
- [x] Reproducibility: hardware, versions, seeds, total compute, full hyperparameters (appendix), anonymized repo
- [x] Threats to validity (Annex G)
- [ ] Main body / appendix split: main body = RQs, configs, mechanisms, metrics, probes, stats, compute summary; appendix = hyperparameters, mechanism implementation details, full ledger, ODD+D, extra configurations

### Expected-results document
- [x] Write Annex L **before running any configuration other than the pilot**, freeze with a dated commit
- [x] Predictions are honest estimates, not targets (include cases where a mechanism is expected to *fail*, e.g. inequity aversion under heterogeneous needs)

### ✅ Gate 8
- [ ] A colleague outside the project can reproduce the experiments from the written protocol alone
- [x] Annex L complete and frozen

---

## Phase 9 — Execution, results and discussion

### Actions
- [x] Run tracking (W&B / MLflow / Hydra), raw results saved
- [x] Results organized **by RQ**
- [x] For each RQ: figure/table → factual observation → explicit verdict on the claim
- [x] Confront results with the reference points (Phase 7)
- [x] Feature ablations analysed (which SES feature matters for which mechanism)
- [ ] Qualitative analysis: spatial harvesting patterns, zone use, punishment targets, collapse trajectories
- [x] **Discussion of imposed vs learned** (required by the project): state clearly that a fairness term in the reward shows the *consequences of a preference*, not the emergence of social cognition; report what the probes say
- [x] Discussion of translation losses: which ledger entries plausibly change the conclusions
- [ ] Each claim decided by a figure/table in the main body

### Expected vs observed comparison
- [x] Fill in the "Observed" columns of Annex L; classify each result (confirmed / explained by an anticipated bias / unexpected); investigate unexpected deviations before interpreting; calibration indicators (coverage, Kendall's τ, mean absolute deviation); never revise predictions after the fact

### Verdict tracking

| Claim | RQ | Experiment | Expected (Annex L) | Observed | Verdict | Comment |
|---|---|---|---|---|---|---|
| Cl1 | — | translation ledger | full coverage | all components mapped, losses justified | supported | docs/ledger.md |
| Cl2 | RQ1 | degenerate config | depletion; IA improves S and E | depletion 10/10; IA no effect | partially | IA reimpl. untuned, no punishment |
| Cl3 | RQ2 | degenerate vs full | ranking change | τ=0.2; reversals sig. in one config only; ANOVA interaction p<1e-19 | partially | collapse floor in full |
| Cl4 | RQ2 | feature ablations | each feature matters | H ✔ (+0.32 raw equality); T ✘ on pre-reg. metric (stock +0.09 exploratory); Z ✘ (punishment unused) | partially | |
| Cl5 | RQ3 | probes | ≥ 2 mechanisms separated | persistence separates SVO vs IPPO; conditionality none | weakly supported | |

---

## Phase 10 — Writing

### Page budget (8 pages — verify for 2028)

| Section | Pages | Content | Status |
|---|---|---|---|
| Introduction | ~1.25 | SES & tragedy of the commons; MARL social mechanisms; problem; gaps G1–G3; idea; contributions C1–C3; key findings | [ ] |
| Related work | ~0.75 | by gap: CPR environments (G1); behaviour in SES models (G2); social mechanisms & their evaluation (G3); positioning vs Melting Pot | [ ] |
| Preliminaries | ~0.5 | POSG; Ostrom's SES framework (4 components, tiers) | [ ] |
| From SES to Markov game (C1) | ~0.75 | translation constraints; ledger table; what is lost | [ ] |
| SHOAL (C2) | ~1.25 | Figure 1; dynamics; heterogeneity; zones; reference points (Phase 7) | [ ] |
| Evaluation suite & protocol (C3) | ~1 | metrics, probes, configurations, mechanisms, statistics | [ ] |
| Results & discussion | ~1.5 | by RQ; imposed vs learned; translation losses | [ ] |
| Limitations & conclusion | ~0.3 | two paragraphs | [ ] |
| References | outside limit* | | [ ] |
| Appendices | unlimited* — **not reviewed** | full ledger, ODD+D, derivations, hyperparameters, extra results | [ ] |

### Main body vs appendix for SHOAL

| Must be in the main body | Can go in an appendix |
|---|---|
| Gaps, contributions, positioning vs Melting Pot / Harvest | Extended related work (SES side) |
| Compact ledger table + the constraints that drive losses | Full ledger, ODD+D description |
| Dynamics equations, threshold, reference points (statements + intuition) | Derivations of MSY / open access / planner |
| Figure 1 | Additional environment screenshots |
| Configurations, mechanisms, metrics, probes, statistics | Hyperparameters, mechanism implementation details |
| Results deciding Cl2–Cl5 | Extra configurations, 6 vs 8 agents, sensitivity to A, per-seed curves |
| Imposed-vs-learned caveat, limitations | Secondary failure cases |

Rules (unchanged): no indispensable content only in appendix; pointers phrased as complements; each appendix referenced; appendices anonymized.

### Introduction structure
1. Commons and SES (fishers, aquifers); MARL agents around a limited resource → tragedy of the commons
2. Social mechanisms proposed in MARL; but tested in environments far from SES structure
3. Why hard: realism vs learnability (many actors, local rules, institutions vs small Markov state, scalar reward, short episodes)
4. Gaps G1–G3
5. Idea: an environment whose structure is taken from Ostrom's framework, with explicit translation losses
6. Contributions C1–C3 with section pointers
7. Key findings (ranking changes, which feature matters, what the probes reveal)

### Writing rules
- [ ] Title / abstract with exact keywords: *multi-agent reinforcement learning*, *sequential social dilemmas*, *common-pool resources*, *social-ecological systems*, *environment / benchmark*
- [ ] Notation consistent with Leibo 2017 / Pérolat 2017
- [ ] Balanced references: MARL (AAMAS / NeurIPS / ICML) + SES (*Ecological Economics*, *Ecology and Society*, *EMS*) + bioeconomics
- [ ] Each intro claim points to evidence (Annex E)
- [ ] Self-citations in the third person
- [ ] Figure 1 on page 1 or top of page 2
- [ ] Never call reward-shaped agents "socially aware" without qualification

### Conclusion (two paragraphs)
- [ ] ¶1: problem, SHOAL, main findings
- [ ] ¶2: limitations (ledger losses, small populations, no communication, no institutional change by agents); future work (agents that *create* rules, larger populations, empirical calibration on a real fishery, LLM agents as in GovSim)

---

## Phase 11 — Adversarial review

- [ ] 2–3 mock reviews (Annex H), main body + references only; ideally one MARL reviewer and one SES modeller
- [x] Read-without-appendix test passed
- [x] "Reviewer 2" pass (Annex I, prefilled)
- [x] Clarity pass: abstract + Figure 1 + end of intro sufficient?
- [x] Consistency pass: G/H/C/Cl numbering, ledger vs environment code vs text
- [ ] Language proofreading

---

## Phase 12 — Submission

See Annex J.
- [ ] Abstract, title, area, keywords by the intermediate deadline
- [ ] Conflicts declared
- [ ] Final PDF compliant and anonymized
- [ ] Supplementary material (code archive anonymized, configs, full ledger)
- [ ] Submitted ≥ 24 h before the deadline

---

## Phase 13 — After submission

- [ ] Draft answers to anticipated objections (Annex I)
- [ ] Complementary experiments ready (extra seeds, 2³ factorial if not done, social-influence mechanism)
- [ ] Rebuttal (Annex K)
- [ ] Public release of SHOAL **after** notification (PyPI + docs)
- [ ] If rejected: incorporate reviews → {{NeurIPS Datasets & Benchmarks / TMLR / JAAMAS / JASSS}}

---

# Annexes: templates to fill in

## Annex A — Decision log

| Date | Decision | Rationale | Alternatives discarded |
|---|---|---|---|
| {{…}} | Contribution = environment + ledger + evaluation suite, not an algorithm | Algorithmic terrain saturated (project document) | New social mechanism |
| {{…}} | PettingZoo for the environment | Standard API, required by the project | Melting Pot substrate (dm_lab2d), JaxMARL |
| {{…}} | Target AAMAS 2028 main track, optional ALA 2027 preliminary | Timeline of the student project | AAMAS 2027 (too early) |
| {{…}} | Authorship and roles of students | {{…}} | {{…}} |
| {{…}} | Hard vs soft access zones | {{…}} | {{…}} |

## Annex B — Mapping and glossary

### Must-cite papers

| Reference | Type | Venue | Why unavoidable |
|---|---|---|---|
| Leibo et al. 2017 | foundational | AAMAS | Sequential social dilemmas, metrics |
| Pérolat et al. 2017 | foundational | NeurIPS | First CPR MARL model, punishment |
| Hughes et al. 2018 | foundational | NeurIPS | Inequity aversion; metrics (U, E, S, P) |
| Jaques et al. 2019 | foundational | ICML | Social influence |
| McKee et al. 2020 | recent | AAMAS | SVO diversity — closest on heterogeneity |
| Köster et al. 2022 | recent | PNAS | Compliance & enforcement learning |
| Vinitsky et al. 2023 | recent | Collective Intelligence | Norms from public sanctions |
| Leibo et al. 2021; Agapiou et al. 2022 | benchmark | ICML / arXiv {{verify}} | Melting Pot — main novelty threat |
| Ostrom 2009 | foundational | Science | SES framework (structure of SHOAL) |
| Schlüter et al. 2017 | foundational | Ecol. Econ. | MoHuB (ledger axes) |
| Müller et al. 2013 | foundational | EMS | ODD+D |
| An 2012 | survey | Ecol. Model. | Decisions in coupled human–natural ABMs |
| Bourceret et al. 2021 | survey | Ecol. & Soc. | Governance in SES ABMs |
| Terry et al. 2021 | tool | NeurIPS | PettingZoo |
| Agarwal et al. 2021; Gorsane et al. 2022 | methodology | NeurIPS | Statistical evaluation |
| Gordon 1954; Clark (bioeconomics) | foundational | — | Open access, depensation, reference points |
| Piatti et al. 2024 (GovSim) | recent | NeurIPS {{verify}} | Fishery sustainability with LLM agents |

### Key groups and researchers

| Group / researcher | Institution | Topic | Link to our topic |
|---|---|---|---|
| Leibo, Hughes, Köster, McKee | Google DeepMind | SSD, Melting Pot, norms | Core MARL community |
| Vinitsky | {{…}} | Norms, SSD codebase | Mechanism implementations |
| Schlüter | Stockholm Resilience Centre | MoHuB, SES behaviour | Translation ledger |
| Bourceret, Amblard, Mathias | INRAE | Governance in SES ABMs | Governance component |
| {{…}} | | | |

### Glossary (MARL ↔ SES)

| Community term | Definition | SES counterpart / avoid |
|---|---|---|
| Common-pool resource (CPR) | Subtractable, hard-to-exclude resource | same; avoid "public good" |
| Sequential social dilemma | Temporally extended mixed-motive Markov game | "social dilemma" alone |
| Appropriation | Harvesting from a CPR | "consumption" |
| Inequity aversion | Reward penalty for payoff differences | other-regarding preference |
| Sanction / punishment | Costly action reducing another's payoff | graduated sanctions (Ostrom) |
| Resource system / units / actors / governance | Ostrom first-tier components | — |
| Depensation (Allee effect) | Growth rate falls at low stock; critical if negative below threshold | "tipping point" (only informally) |

## Annex C — Reading sheet (one per paper)

```markdown
### {{Authors, Year, Title}} — {{Venue}}
- **Problem**:
- **Formalism** (POSG / ABM / other):
- **Agent heterogeneity** (none / preferences / capabilities / needs):
- **Resource dynamics**:
- **Institutions / sanctions**:
- **Social mechanism and where it lives** (reward / action / observation / rules):
- **Metrics**:
- **Key results**:
- **Stated limitations**:
- **Unstated limitations**:
- **Code available**: yes / no – link
- **Gaps covered**: G1 ● / G2 ◐ / G3 ○
- **Use in SHOAL**: reference result to reproduce / mechanism to implement / ledger entry / discussion
```

## Annex D — Related work × gaps matrix

Legend: ● fully, ◐ partially, ○ not at all. **To verify while reading.**

| Work | G1 structure | G2 translation | G3 evaluation | Code | Role |
|---|---|---|---|---|---|
| Leibo 2017 (SSD) | ○ | ○ | ◐ | ◐ | metrics |
| Pérolat 2017 (CPR) | ◐ | ○ | ○ | {{…}} | reference result, punishment |
| Hughes 2018 (IA) | ○ | ○ | ◐ | ● (SSD repo) | mechanism, reference result |
| McKee 2020 (SVO) | ◐ | ○ | ◐ | {{…}} | mechanism |
| Melting Pot (2021/22) | ◐ | ○ | ● | ● | main novelty threat, eval protocol |
| Vinitsky 2023 | ○ | ○ | ◐ | {{…}} | optional mechanism |
| Schlüter 2017 (MoHuB) | ○ | ◐ | ○ | — | ledger axes |
| Bourceret 2021 | ○ | ◐ | ○ | — | governance mapping |
| GovSim 2024 | {{…}} | ○ | ◐ | ● | discussion |
| **SHOAL (ours)** | ● | ● | ● | yes | — |

## Annex E — Traceability matrix

| Gap | Hypothesis | Contribution | Claim(s) | Deductive evidence | Inductive evidence | Main-body section | Appendix complements |
|---|---|---|---|---|---|---|---|
| G1 | H1a–c | C2 | Cl2, Cl3, Cl4 | reference points (open access, planner, threshold) | Exp. 1–3 | §SHOAL, §Results RQ1–RQ2 | derivations, extra configs |
| G2 | H2 | C1 | Cl1 | ledger (descriptive) | — | §From SES to Markov game | full ledger, ODD+D |
| G3 | H3 | C3 | Cl5 | — | probes (Exp. 4) | §Evaluation suite, §Results RQ3 | probe details |

## Annex F — Claim specification

| Claim | Statement | Gap | Metric | Compared to | Condition | Success threshold | If not supported |
|---|---|---|---|---|---|---|---|
| Cl1 | All four Ostrom first-tier components represented or explicitly lost; every SHOAL variable mapped | G2 | ledger coverage | — | — | 100 % coverage; each loss justified | Not "unsupported" by nature: fix the ledger |
| Cl2 | Degenerate config reproduces: selfish IPPO depletes; IA improves S and E | G1 | S, E, collapse | Hughes 2018 / Pérolat 2017 directions | 10 seeds | same direction, Holm-corrected p < 0.05 | Investigate dynamics; if still not, report and weaken backward-compatibility claim |
| Cl3 | Mechanism ranking differs between degenerate and full configs | G1 | ranking on S and E | degenerate config | 10 seeds | Kendall τ < 0.5 and ≥ 1 significant pairwise reversal | Report robustness of SSD conclusions (valid but weaker finding; consider ALA / JAAMAS) |
| Cl4 | Each feature removed from full config changes ≥ 1 metric as predicted | G1 | Gini (H), collapse (T), punishment freq. (Z) | full config | 10 seeds | significant effect, predicted direction | Drop or redesign the feature; report negative result |
| Cl5 | Probes separate ≥ 2 mechanisms (persistence and/or conditionality) | G3 | persistence, conditionality scores | across mechanisms | 10 seeds | significant difference | Present probes as a methodological proposal with negative result |

## Annex G — Experimental protocol (prefilled)

```markdown
### Research questions
- RQ1 (Cl2): Does the degenerate configuration reproduce known SSD results?
- RQ2 (Cl3, Cl4): How do heterogeneity, access zones and the collapse threshold change mechanism effects and rankings?
- RQ3 (Cl5): Which mechanisms yield learned, conditional behaviour rather than an imposed preference?

### Environment configurations
| Config | Heterogeneity | Zones | Threshold | Justification |
| Degenerate | off | off | off | Harvest-like, backward compatibility |
| +H / +Z / +T | one on | | | isolate each feature |
| Full | on | on | on | SES-grounded setting |
Agents: 6 and 8. Grid: {{…}}. Horizon: {{…}}.

### Learning configurations
| Config | Mechanism location | Implementation | Tuning |
| IPPO selfish | — | {{…}} | equal budget |
| MAPPO selfish | — | {{…}} | equal budget |
| + inequity aversion | reward | {{SSD repo / reimpl.}} | equal budget |
| + punishment | action | {{…}} | equal budget |
| + SVO diversity | reward (heterogeneous) | {{…}} | equal budget |
| (+ social influence / norm learning) | reward / observation | {{…}} | equal budget |
Reference policies: random, greedy, planner/quota.

### Metrics
| Metric | Definition | Standard / specific | Claim(s) |
| Group harvest (normalized) | Σ harvest / planner bound | standard (normalization specific) | Cl2–4 |
| Sustainability | final stock / K; collapse rate; time to collapse | standard + specific | Cl2–4 |
| Equality | 1 − Gini (raw); 1 − Gini (need-normalized) | standard + specific | Cl2–4 |
| Punishment frequency | punishments / step | standard | Cl4 |
| Persistence / conditionality | see Phase 8 | specific | Cl5 |

### Statistical protocol
- Seeds: ≥ 10
- Aggregation: IQM, bootstrap 95% CIs (rliable)
- Tests: Welch / Mann–Whitney; two-way analysis for feature × mechanism
- Correction: Holm

### Reproducibility
- Hardware: {{…}}  - Software: {{…}}  - Compute: {{…}}  - Anonymized repo: {{…}}

### Threats to validity
- Internal: reimplementation of mechanisms; unequal tuning; PettingZoo throughput limiting training length
- External: one resource type (fishery), 6–8 agents, gridworld abstraction; no empirical calibration
- Construct: raw Gini vs fairness under heterogeneous needs; "social cognition" vs reward shaping; aggregate reference points vs spatial dynamics
```

## Annex H — Mock review form

*(unchanged from the generic recipe — add one SES-specific line:)*
**Is the SES grounding credible to a social-ecological modeller, or decorative?**

## Annex I — Likely objections ("Reviewer 2")

| Objection | Likelihood | Where defused | Answer prepared |
|---|---|---|---|
| "Melting Pot Commons Harvest already does this (zones, depletion)" | high | Related work, comparison table | combination + global threshold + reference points + ledger + probes; degenerate config shows continuity |
| "Just another gridworld; environment papers lack insight" | high | Results RQ2 | ranking changes (Cl3) are the insight |
| "Ostrom grounding is cosmetic" | high | ledger table in main body | each variable mapped, losses stated |
| "Fairness in reward ≠ social cognition" | high | Discussion, RQ3 | we say so explicitly; probes |
| Too few agents (6–8) | medium | Limitations | consistent with SSD literature; scaling as future work |
| Missing mechanism (social influence, norms) | medium | Protocol | budget choice; complementary runs ready |
| Unequal tuning / reimplemented baselines | medium | Protocol, Annex L biases | equal budget; check against reported values |
| Too few seeds / no stats | low | Protocol | ≥ 10 seeds, rliable, Holm |
| Need-normalized Gini not justified | medium | Metrics | planner Gini reference (Phase 7) |
| Essential content in appendix | medium | Phase 10 split | read-without-appendix test |

## Annex J — Submission checklist

*(unchanged from the generic recipe)*, plus:
- [ ] Environment repository anonymized (no group/lab name in package, docstrings, URLs, licence headers)
- [x] Compact ledger table and Figure 1 in the main body

## Annex K — Rebuttal template

*(unchanged from the generic recipe)*

## Annex L — Expected-results document (pre-registered predictions)

> Freeze before running any configuration other than the pilot. Date: {{date}} – commit: {{hash}}.
> The values below are **first estimates (low confidence)** to be revised once, before freezing, after the pilot.

### L.1 Assumptions behind the predictions
- Hypotheses assumed true: H1a, H1b, H1c, H3
- Theoretical results: open-access equilibrium, planner bound, threshold analysis (Phase 7)
- Other sources: Hughes 2018 and Pérolat 2017 qualitative results; pilot results

### L.2 Condition descriptions

| Condition | Type | Implementation | Tuning budget | Gaps | Known strengths / weaknesses |
|---|---|---|---|---|---|
| IPPO selfish | reference learner | {{…}} | {{…}} | — | simple; known to deplete CPRs |
| MAPPO selfish | reference learner | {{…}} | {{…}} | — | centralized critic does not change selfish rewards |
| + inequity aversion | mechanism (reward) | {{…}} | {{…}} | G3 | strong in homogeneous settings; H1a predicts weakness under heterogeneity |
| + punishment | mechanism (action) | {{…}} | {{…}} | G3 | reactive; H1b predicts weakness with threshold |
| + SVO diversity | mechanism (reward) | {{…}} | {{…}} | G3 | robust populations; sensitive to SVO distribution |
| Planner / quota | reference policy | analytic | — | — | upper bound; not learned |

### L.3 Anticipated biases

| Bias | Affected | Direction | Magnitude | Mitigation |
|---|---|---|---|---|
| Mechanism reimplementations underperform originals | IA, influence | ↓ | {{−10 to −30 % of effect}} | reproduce in degenerate config first |
| Familiarity / tuning effort | all | ± | {{…}} | equal budget, logged |
| Training truncated by PettingZoo throughput | all, esp. full config | ↓ | {{…}} | measure in pilot; vectorize |
| Heterogeneous agents learn at different speeds | +H configs | ↑ Gini | {{…}} | report per-type learning curves |
| Seed variance near the threshold (bimodal outcomes) | +T configs | ± large | high | report collapse rate, not only means |

### L.4 Expected numerical results (first estimates)

**Full configuration — collapse rate (% of seeds)**

| Condition | Ideal | Realistic | Interval (80 %) | Rank | Rationale | Confidence | Observed | Status |
|---|---|---|---|---|---|---|---|---|
| IPPO selfish | 90 % | 85 % | [60, 100] | 5 | open-access equilibrium below A | med | | |
| MAPPO selfish | 85 % | 80 % | [50, 100] | 4 | same rewards | med | | |
| + punishment | 60 % | 70 % | [40, 100] | 3 | H1b: reactive | low | | |
| + inequity aversion | 40 % | 50 % | [20, 80] | 2 | preventive; weakened by H1a? | low | | |
| + SVO diversity | 40 % | 50 % | [20, 80] | 2 | preventive | low | | |
| Planner / quota | 0 % | 0 % | [0, 0] | 1 | by construction | high | | |

**Full configuration — group harvest (normalized by planner bound)**

| Condition | Realistic | Interval (80 %) | Confidence | Observed | Status |
|---|---|---|---|---|---|
| IPPO selfish | 0.30 | [0.15, 0.50] | low | | |
| + inequity aversion | 0.55 | [0.35, 0.75] | low | | |
| + punishment | 0.40 | [0.20, 0.65] | low | | |
| + SVO diversity | 0.55 | [0.35, 0.75] | low | | |

*(Repeat for degenerate config and each single-feature config; and for equality, raw vs need-normalized.)*

### L.5 Expected trends and orderings

| Trend | Expected | Source | Observed | Status |
|---|---|---|---|---|
| Ranking, degenerate config (sustainability) | IA ≈ SVO > punishment > selfish | Hughes 2018, Pérolat 2017 | | |
| Ranking, full config | preference-based > punishment > selfish, gaps larger | H1b | | |
| IA equality gain, +H vs degenerate | smaller (raw Gini), possibly negative on need-normalized Gini | H1a | | |
| Punishment frequency, +Z vs degenerate | lower | H1c | | |
| 8 vs 6 agents | higher collapse rate | open access with more appropriators | | |

### L.6 Link to claims

| Claim | Supporting values | Refuting values | Observed | Verdict |
|---|---|---|---|---|
| Cl2 | selfish collapse/depletion; IA > selfish on S and E | IA ≤ selfish | | |
| Cl3 | τ < 0.5, ≥ 1 significant reversal | τ ≥ 0.8, no reversal | | |
| Cl4 | each feature significant in predicted direction | no significant effect | | |
| Cl5 | ≥ 2 mechanisms separated by probes | no difference | | |

### L.7 Comparison after runs
- Share inside realistic intervals: {{…}} — Kendall's τ expected vs observed: {{…}} — MAD: {{…}}

| Unexpected deviation | Investigation | Conclusion | Consequence |
|---|---|---|---|
| {{…}} | {{…}} | {{…}} | {{…}} |

### L.8 Addenda (dated, after freeze)

| Date | Change | Justification | Before / after results? |
|---|---|---|---|
| {{…}} | {{…}} | {{…}} | {{…}} |
