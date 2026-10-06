"""Exploratory (post-hoc, NOT pre-registered) tests, reported separately in docs/results.md."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoal import stats as S  # noqa: E402

R = ROOT / "results" / "runs"
g = lambda e, c, m, k: np.asarray(json.load(open(R / f"{e}__{c}__{m}.json"))["metrics"][k])
out = {}
out["svo_vs_ippo_collapse_+Z"] = dict(svo=g("main", "+Z", "svo", "collapse").tolist(), ippo=g("main", "+Z", "ippo", "collapse").tolist(),
                                     p=S.mwu(g("main", "+Z", "svo", "collapse"), g("main", "+Z", "ippo", "collapse")))
out["svo_final_stock_+Z_vs_deg"] = dict(p=S.mwu(g("main", "+Z", "svo", "final_stock"), g("main", "degenerate", "svo", "final_stock")),
                                       diff=float(g("main", "+Z", "svo", "final_stock").mean() - g("main", "degenerate", "svo", "final_stock").mean()))
out["ia_vs_ippo_equality_need_+H"] = dict(p=S.mwu(g("main", "+H", "ia", "equality_need"), g("main", "+H", "ippo", "equality_need")),
                                         diff=float(g("main", "+H", "ia", "equality_need").mean() - g("main", "+H", "ippo", "equality_need").mean()))
out["ia_vs_ippo_equality_raw_+H"] = dict(p=S.mwu(g("main", "+H", "ia", "equality"), g("main", "+H", "ippo", "equality")),
                                        diff=float(g("main", "+H", "ia", "equality").mean() - g("main", "+H", "ippo", "equality").mean()))
for m in ["ippo", "ia", "punish"]:
    a, f = g("ablation", "full-T", m, "final_stock"), g("main", "full", m, "final_stock")
    out[f"fullT_vs_full_final_stock_{m}"] = dict(diff=float(a.mean() - f.mean()), p=S.mwu(a, f))
    a, f = g("ablation", "full-T", m, "norm_harvest") if False else g("ablation", "full-T", m, "group_harvest"), g("main", "full", m, "group_harvest")
    out[f"fullT_vs_full_group_harvest_{m}"] = dict(diff=float(a.mean() - f.mean()), p=S.mwu(a, f))
out["punish_rate_all"] = {c: float(g("main", c, "punish", "punish_rate").mean()) for c in ["degenerate", "+H", "+Z", "+T", "full"]}
json.dump(out, open(ROOT / "results" / "exploratory.json", "w"), indent=1)
print(json.dumps(out, indent=1))
