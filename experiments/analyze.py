"""Phase 9 analysis: tables by RQ, claim verdicts, Annex L comparison, figures.
Writes results/summary.json, results/tables.md and figures/*.png."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoal import stats as S  # noqa: E402

RUNS = ROOT / "results" / "runs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
MECHS = ["ippo", "mappo", "ia", "punish", "svo"]
CONFIGS = ["degenerate", "+H", "+Z", "+T", "full"]
METRICS = ["norm_harvest", "final_stock", "collapse", "equality", "equality_need", "punish_rate", "fish_rate"]
ALPHA = 0.05


def load(exp, c, m):
    f = RUNS / f"{exp}__{c}__{m}.json"
    return json.load(open(f)) if f.exists() else None


scripted = json.load(open(RUNS / "scripted.json"))
planner = {c: v["reference"]["planner_harvest"] for c, v in scripted.items()}


def metrics(exp, c, m):
    r = load(exp, c, m)
    if r is None:
        return None
    d = {k: np.asarray(v) for k, v in r["metrics"].items()}
    d["norm_harvest"] = d["group_harvest"] / planner[c.split("_")[0] if c in planner else "+T"]
    return d


def fmt(x):
    lo, hi = S.bootstrap_ci(x)
    return f"{S.iqm(x):.2f} [{lo:.2f}, {hi:.2f}]"


out, md = {}, []

# ------------------------------------------------------------------ reference points
md.append("## Reference points (Phase 7)\n")
md.append("| Config | x_MSY | MSY/step | x_oa | predicts collapse | planner harvest | planner Gini (need-prop.) | random H | greedy H | quota H | greedy stock | quota stock |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c, v in scripted.items():
    rp = v["reference"]
    md.append(f"| {c} | {rp['x_msy']:.2f} | {rp['msy_per_step']:.2f} | {rp['x_oa']:.3f} | {rp['predicts_collapse']} | "
              f"{rp['planner_harvest']:.1f} | {rp['planner_gini_need_prop']:.2f} | "
              f"{np.mean(v['random']['group_harvest']):.1f} | {np.mean(v['greedy']['group_harvest']):.1f} | "
              f"{np.mean(v['quota']['group_harvest']):.1f} | {np.mean(v['greedy']['final_stock']):.2f} | {np.mean(v['quota']['final_stock']):.2f} |")

# ------------------------------------------------------------------ main table
data = {(c, m): metrics("main", c, m) for c in CONFIGS for m in MECHS}
md.append("\n## Main results: IQM [95% bootstrap CI] over 10 seeds (16 eval episodes each)\n")
for c in CONFIGS:
    md.append(f"\n### {c}\n")
    md.append("| Mechanism | " + " | ".join(METRICS) + " |")
    md.append("|---" * (len(METRICS) + 1) + "|")
    for m in MECHS:
        d = data[(c, m)]
        if d is None:
            continue
        md.append(f"| {m} | " + " | ".join(fmt(d[k]) if k != "collapse" else f"{d[k].mean()*100:.0f} %" for k in METRICS) + " |")

# planner bound check
exceed = [(c, m) for (c, m), d in data.items() if d is not None and (d["norm_harvest"] > 1).any()]
out["planner_bound_violations"] = exceed

# ------------------------------------------------------------------ RQ1 / Cl2
deg = {m: data[("degenerate", m)] for m in MECHS}
x_msy = scripted["degenerate"]["reference"]["x_msy"]
cl2 = {}
cl2["ippo_frac_seeds_below_xmsy"] = float((deg["ippo"]["final_stock"] < x_msy).mean())
ps, names = [], []
for met in ["final_stock", "equality"]:
    for m in ["ia", "svo", "punish", "mappo"]:
        ps.append(S.mwu(deg[m][met], deg["ippo"][met]))
        names.append(f"{m}_vs_ippo_{met}")
adj = S.holm(ps)
cl2["tests"] = {n: dict(p_holm=float(a), diff_mean=float(deg[n.split('_')[0]][n.split('_vs_ippo_')[1]].mean()
                                                       - deg['ippo'][n.split('_vs_ippo_')[1]].mean()))
                for n, a in zip(names, adj)}
ia_s = cl2["tests"]["ia_vs_ippo_final_stock"]
ia_e = cl2["tests"]["ia_vs_ippo_equality"]
cl2["verdict"] = ("supported" if cl2["ippo_frac_seeds_below_xmsy"] >= 0.8 and ia_s["diff_mean"] > 0 and ia_s["p_holm"] < ALPHA
                  and ia_e["diff_mean"] > 0 and ia_e["p_holm"] < ALPHA else
                  "partially supported" if cl2["ippo_frac_seeds_below_xmsy"] >= 0.8 else "not supported")
out["Cl2"] = cl2

# ------------------------------------------------------------------ RQ2 / Cl3 ranking change
def ranking(c, met):
    return [data[(c, m)][met].mean() for m in MECHS]


cl3 = {}
for met in ["final_stock", "norm_harvest", "equality_need"]:
    tau = {c: S.kendall(ranking("degenerate", met), ranking(c, met))[0] for c in CONFIGS[1:]}
    # significant reversals: sign of difference flips and difference significant (Holm over pairs) in both configs
    pairs = list(itertools.combinations(MECHS, 2))
    pd = S.holm([S.mwu(data[("degenerate", a)][met], data[("degenerate", b)][met]) for a, b in pairs])
    pf = S.holm([S.mwu(data[("full", a)][met], data[("full", b)][met]) for a, b in pairs])
    rev = []
    for (a, b), p1, p2 in zip(pairs, pd, pf):
        s1 = np.sign(data[("degenerate", a)][met].mean() - data[("degenerate", b)][met].mean())
        s2 = np.sign(data[("full", a)][met].mean() - data[("full", b)][met].mean())
        if s1 != s2 and s1 != 0 and s2 != 0 and (p1 < ALPHA or p2 < ALPHA):
            rev.append(dict(pair=f"{a}-{b}", p_deg=float(p1), p_full=float(p2), both_sig=bool(p1 < ALPHA and p2 < ALPHA)))
    cl3[met] = dict(tau_vs_degenerate=tau, reversals_deg_vs_full=rev)
# two-way ANOVA config (deg, full) x mechanism
for met in ["final_stock", "equality", "norm_harvest"]:
    y = np.array([[data[(c, m)][met] for m in MECHS] for c in ["degenerate", "full"]])
    cl3[f"anova_{met}"] = S.two_way_anova(y)
t = cl3["final_stock"]["tau_vs_degenerate"]["full"]
cl3["verdict"] = "supported" if (t < 0.5 and any(r["both_sig"] for r in cl3["final_stock"]["reversals_deg_vs_full"])) else \
    ("partially supported" if t < 0.5 or cl3["final_stock"]["reversals_deg_vs_full"] or cl3["norm_harvest"]["reversals_deg_vs_full"] else "not supported")
out["Cl3"] = cl3

# ------------------------------------------------------------------ RQ2 / Cl4 ablations
cl4 = {}
pred = {"full-H": ("equality", +1), "full-T": ("collapse", -1), "full-Z": ("punish_rate", +1)}
ps, keys = [], []
for abl, (met, sign) in pred.items():
    for m in ["ippo", "ia", "punish"]:
        if met == "punish_rate" and m != "punish":
            continue
        a, f = metrics("ablation", abl, m), data[("full", m)]
        if a is None:
            continue
        ps.append(S.mwu(a[met], f[met]))
        keys.append((abl, m, met, sign, float(a[met].mean() - f[met].mean())))
adj = S.holm(ps) if ps else []
for (abl, m, met, sign, diff), p in zip(keys, adj):
    cl4[f"{abl}/{m}/{met}"] = dict(diff=diff, p_holm=float(p), predicted_sign=sign,
                                    ok=bool(p < ALPHA and np.sign(diff) == sign))
by_feat = {abl: any(v["ok"] for k, v in cl4.items() if k.startswith(abl)) for abl in pred}
cl4["per_feature"] = by_feat
cl4["verdict"] = "supported" if all(by_feat.values()) else ("partially supported" if any(by_feat.values()) else "not supported")
out["Cl4"] = cl4

# H1c: punishment frequency +Z vs degenerate
out["H1c_punish_rate_Z_vs_deg"] = dict(
    deg=float(data[("degenerate", "punish")]["punish_rate"].mean()), plusZ=float(data[("+Z", "punish")]["punish_rate"].mean()),
    p=S.mwu(data[("degenerate", "punish")]["punish_rate"], data[("+Z", "punish")]["punish_rate"]))
# H1a: IA equality gain, +H vs degenerate (raw and need-normalised)
out["H1a_IA_equality_gain"] = {c: dict(raw=float(data[(c, "ia")]["equality"].mean() - data[(c, "ippo")]["equality"].mean()),
                                       need=float(data[(c, "ia")]["equality_need"].mean() - data[(c, "ippo")]["equality_need"].mean()))
                               for c in CONFIGS}

# ------------------------------------------------------------------ Exp 3 threshold sweep
thr = {}
for f in sorted(RUNS.glob("threshold__*.json")):
    r = json.load(open(f))
    a = r["env_params"]["a"]
    thr[a] = dict(collapse=float(np.mean(r["metrics"]["collapse"])), final_stock=float(np.mean(r["metrics"]["final_stock"])))
out["threshold_sweep"] = thr

# ------------------------------------------------------------------ RQ3 / Cl5 probes
pf = ROOT / "results" / "probes.json"
cl5 = {}
if pf.exists():
    pr = json.load(open(pf))
    for c in ["degenerate", "full"]:
        cond, pers = {}, {}
        for m in MECHS:
            k = f"{c}__{m}"
            if k not in pr:
                continue
            fd = np.asarray(pr[k]["cond_defectors"]["focal_fish_rate"])
            fc = np.asarray(pr[k]["cond_cooperators"]["focal_fish_rate"])
            cond[m] = fd - fc
            if "after_selfish_ft" in pr[k]:
                pers[m] = np.asarray(pr[k]["after_selfish_ft"]["final_stock"]) - data[(c, m)]["final_stock"]
        mlist = list(cond)
        pairs = list(itertools.combinations(mlist, 2))
        pc = S.holm([S.mwu(cond[a], cond[b]) for a, b in pairs])
        plist = list(pers)
        ppairs = [(m, "ippo") for m in plist if m != "ippo"]
        pp = S.holm([S.mwu(pers[a], pers[b]) for a, b in ppairs]) if ppairs else []
        cl5[c] = dict(
            conditionality={m: dict(mean=float(v.mean()), iqm=float(S.iqm(v))) for m, v in cond.items()},
            conditionality_sig_pairs=[f"{a}-{b} (p={p:.3g})" for (a, b), p in zip(pairs, pc) if p < ALPHA],
            persistence_dstock={m: float(v.mean()) for m, v in pers.items()},
            persistence_sig_vs_ippo=[f"{a} (p={p:.3g})" for (a, _), p in zip(ppairs, pp) if p < ALPHA])
    sep = any(cl5[c]["conditionality_sig_pairs"] or cl5[c]["persistence_sig_vs_ippo"] for c in cl5)
    cl5["verdict"] = "supported" if sep else "not supported"
out["Cl5"] = cl5

json.dump(out, open(ROOT / "results" / "summary.json", "w"), indent=1, default=float)
open(ROOT / "results" / "tables.md", "w").write("\n".join(md) + "\n")

# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = dict(ippo="#4C72B0", mappo="#8172B2", ia="#55A868", punish="#C44E52", svo="#DD8452")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, met, lab in zip(axes, ["final_stock", "norm_harvest", "equality_need"],
                        ["final stock X_T/K", "group harvest / planner bound", "1 − Gini (need-normalised)"]):
    w = 0.16
    for j, m in enumerate(MECHS):
        vals = [data[(c, m)][met] for c in CONFIGS]
        ax.bar(np.arange(len(CONFIGS)) + (j - 2) * w, [S.iqm(v) for v in vals], w, color=COL[m], label=m)
        for i, v in enumerate(vals):
            ax.scatter(np.full(len(v), i + (j - 2) * w), v, s=3, color="k", alpha=0.4, zorder=3)
    ax.set_xticks(range(len(CONFIGS)), CONFIGS)
    ax.set_title(lab, fontsize=10)
    if met == "final_stock":
        ax.axhline(scripted["full"]["reference"]["x_msy"], ls=":", c="gray", lw=1)
        ax.axhline(0.2, ls="--", c="r", lw=1)
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "fig_main_results.png", dpi=150)

if thr:
    fig, ax = plt.subplots(figsize=(4, 3))
    a = sorted(thr)
    ax.plot(a, [thr[x]["collapse"] for x in a], "o-", label="collapse rate (IPPO)")
    ax.plot(a, [thr[x]["final_stock"] for x in a], "s--", label="final stock")
    ax.axvline(scripted["+T"]["reference"]["x_oa"], c="gray", ls=":", label="open-access x_oa")
    ax.set_xlabel("Allee threshold A")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_threshold_sweep.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for ax, c in zip(axes, ["degenerate", "full"]):
    for m in MECHS:
        r = load("main", c, m)
        ax.plot(np.asarray(r["curves"]["stock"]), color=COL[m], label=m, lw=1)
    ax.set_title(f"{c}: mean stock during training rollouts", fontsize=9)
    ax.set_xlabel("PPO update")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_learning_curves.png", dpi=150)

print(json.dumps({k: out[k].get("verdict") if isinstance(out[k], dict) else out[k] for k in out}, indent=1, default=str))
