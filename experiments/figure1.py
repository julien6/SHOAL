"""Figure 1: Ostrom's SES first-tier components (left) mapped to SHOAL (right) through the ledger."""
import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoal import env as E  # noqa: E402
from shoal import policies as P  # noqa: E402

p = E.config("full")
s = E.reset(jax.random.PRNGKey(3), p)
k = jax.random.PRNGKey(4)
for _ in range(25):  # a few greedy steps for a non-uniform stock field
    k, k1 = jax.random.split(k)
    s, *_ = E.step(k1, s, P.greedy(k1, s, p), p)
h, need = E.agent_types(p)

fig = plt.figure(figsize=(10, 4.4))
axl = fig.add_axes([0.0, 0.0, 0.42, 1.0])
axl.set_xlim(0, 10); axl.set_ylim(0, 10); axl.axis("off")
boxes = {
    "RS": (0.3, 7.2, "Resource system (RS)\nproductivity, thresholds"),
    "RU": (0.3, 4.9, "Resource units (RU)\nmobility, regrowth"),
    "A": (0.3, 2.6, "Actors (A)\nheterogeneity, dependence"),
    "GS": (0.3, 0.3, "Governance (GS)\naccess rules, sanctions"),
}
for key, (x, y, txt) in boxes.items():
    axl.add_patch(FancyBboxPatch((x, y), 5.6, 1.8, boxstyle="round,pad=0.1", fc="#EAF2F8", ec="#2E86C1"))
    axl.text(x + 2.8, y + 0.9, txt, ha="center", va="center", fontsize=9)
axl.text(3.1, 9.6, "Ostrom (2009) SES framework", ha="center", fontsize=10, weight="bold")
labels = {"RS": "logistic r, global Allee\nthreshold A (feature T)",
          "RU": "per-cell stock s_c,\ndiffusion",
          "A": "types θ=(h_i, n_i)\n(feature H)",
          "GS": "zones (feature Z),\npunishment action"}
for key, (x, y, _) in boxes.items():
    axl.annotate("", xy=(10.2, y + 0.9), xytext=(6.1, y + 0.9),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1.2), annotation_clip=False)
    axl.text(8.1, y + 1.35, labels[key], ha="center", va="bottom", fontsize=7, color="#333")

axr = fig.add_axes([0.5, 0.08, 0.36, 0.84])
G = p.grid
im = axr.imshow(np.asarray(s.stock), cmap="YlGnBu", vmin=0, vmax=1)
for z in range(1, p.n_zones):
    axr.axvline(z * G / p.n_zones - 0.5, color="crimson", lw=2, ls="--")
pos = np.asarray(s.pos)
for i in range(p.n_agents):
    hi = float(h[i]) > p.h_mean
    axr.scatter(pos[i, 1] + (i % 2) * 0.2 - 0.1, pos[i, 0], s=260 if hi else 90,
                marker="^" if hi else "o", c="#E67E22" if hi else "#8E44AD", edgecolor="k", zorder=3)
    axr.text(pos[i, 1] + (i % 2) * 0.2 - 0.1, pos[i, 0], str(i), ha="center", va="center", fontsize=7, color="w", zorder=4)
axr.set_xticks([]); axr.set_yticks([])
axr.set_title(f"SHOAL (full config): X/K = {float(s.stock.mean()):.2f}, A = {p.a}", fontsize=10)
cb = fig.colorbar(im, ax=axr, fraction=0.046)
cb.set_label("cell stock s_c / k")
axr.text(0, G - 0.1, "▲ high capacity/need   ● low   - - zone boundary", fontsize=7, va="top")
fig.savefig(ROOT / "figures" / "fig1_ses_to_shoal.png", dpi=170, bbox_inches="tight")
print("ok")
