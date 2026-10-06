"""Rendering for SHOAL: episode recording (JAX), frame drawing and animation (matplotlib).

    traj = record_episode(key, p, act_fn)          # stacked states / actions / catches, time-major
    fig, ax = draw_frame(traj, t, p)                # one annotated-ready frame
    save_animation(traj, p, "replay.gif")           # GIF (Pillow) or MP4 (ffmpeg)

Rendering happens on recorded states after the rollout, so training speed is unaffected.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from . import env as E  # noqa: E402

# palette (validated categorical slots; see paper/make_figures.py)
C_HI, C_LO, C_FISH, C_ZONE, C_PUN = "#eb6834", "#4a3aa7", "#eda100", "#e34948", "#0b0b0b"
INK, MUTED = "#0b0b0b", "#52514e"


class Trajectory(NamedTuple):
    stock: np.ndarray    # (T+1, G, G)
    pos: np.ndarray      # (T+1, N, 2)
    frozen: np.ndarray   # (T+1, N)
    season_h: np.ndarray  # (T+1, N)
    actions: np.ndarray  # (T, N)
    catch: np.ndarray    # (T, N)
    fished: np.ndarray   # (T, N)
    punished: np.ndarray  # (T, N)


def record_episode(key, p: E.EnvParams, act_fn, steps=None):
    """Roll out one episode and keep every state. act_fn(key, state, obs) -> actions (N,)."""
    steps = steps or p.horizon
    k0, key = jax.random.split(key)
    s0 = E.reset(k0, p)

    def body(c, _):
        s, k = c
        k, ka, ks = jax.random.split(k, 3)
        a = act_fn(ka, s, E.observe(s, p))
        a = jnp.where(s.frozen == 0, a, E.STAY)
        s2, r, d, info = E.step(ks, s, a, p)
        out = (s2.stock, s2.pos, s2.frozen, s2.season_harvest, a, info["catch"], info["fished"], info["punished"])
        return (s2, k), out

    _, outs = jax.jit(lambda s, k: jax.lax.scan(body, (s, k), None, steps))(s0, key)
    stock, pos, frozen, sh, a, catch, fished, pun = (np.asarray(x) for x in outs)
    cat = lambda first, rest: np.concatenate([np.asarray(first)[None], rest], 0)
    return Trajectory(cat(s0.stock, stock), cat(s0.pos, pos), cat(s0.frozen, frozen),
                      cat(s0.season_harvest, sh), a, catch, fished, pun)


def draw_grid(ax, traj: Trajectory, t: int, p: E.EnvParams, show_ids=True, title=None, scale=1.0):
    """Draw the grid at time t (t indexes states; actions of step t-1 are shown as fishing/zaps)."""
    G = p.grid
    h, _ = E.agent_types(p)
    ax.imshow(traj.stock[t], cmap="Blues", vmin=0, vmax=1, extent=(-0.5, G - 0.5, G - 0.5, -0.5))
    for k in range(G + 1):
        ax.axhline(k - 0.5, color="white", lw=0.4)
        ax.axvline(k - 0.5, color="white", lw=0.4)
    if p.zones:
        for z in range(1, p.n_zones):
            ax.axvline(z * G / p.n_zones - 0.5, color=C_ZONE, lw=1.6, ls=(0, (3, 2)))
    pos = traj.pos[t]
    fished = traj.fished[t - 1] if t > 0 else np.zeros(p.n_agents, bool)
    pun = traj.punished[t - 1] if t > 0 else np.zeros(p.n_agents, bool)
    # small offsets so co-located agents stay visible
    offs = np.array([(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2), (0.0, -0.28), (0.0, 0.28), (-0.28, 0), (0.28, 0)])
    for i in range(p.n_agents):
        y, x = pos[i]
        dx, dy = offs[i % len(offs)]
        cx, cy = x + dx, y + dy
        hi = float(h[i]) > p.h_mean if p.heterogeneity else None
        col = C_HI if hi else (C_LO if hi is False else "#2a78d6")
        mk = "^" if hi else "o"
        if traj.frozen[t][i] > 0:
            col = "#9b9a96"
        if fished[i]:
            ax.add_patch(Circle((cx, cy), 0.26 * max(scale, 0.75), fill=False, ec=C_FISH, lw=1.8 * max(scale, 0.7), zorder=4))
        if pun[i]:
            ax.add_patch(Rectangle((x - 1.5, y - 1.5), 3, 3, fill=False, ec=C_PUN, lw=1.3, ls=(0, (1, 1.2)), zorder=4))
        ax.scatter([cx], [cy], s=(95 if mk == "^" else 70) * scale, marker=mk, c=col, edgecolors="white", linewidths=0.9 * scale ** 0.5, zorder=5)
        if show_ids:
            ax.text(cx, cy + (0.03 if mk == "^" else 0), str(i), ha="center", va="center", fontsize=5.5, color="white", zorder=6)
    ax.set_xlim(-0.5, G - 0.5); ax.set_ylim(G - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8, pad=3)


def draw_gauge(ax, traj: Trajectory, t: int, p: E.EnvParams, x_oa=None, label=True):
    """Vertical stock gauge x = X/K with the threshold A (if on) and the open-access stock."""
    x = traj.stock[t].mean()
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#f4f3ef", ec=MUTED, lw=0.6))
    ax.add_patch(Rectangle((0, 0), 1, x, fc="#2a78d6", ec="none"))
    if p.threshold:
        ax.add_patch(Rectangle((0, 0), 1, p.a, fc="none", ec=C_ZONE, hatch="////", lw=0, alpha=0.6, zorder=3))
        ax.axhline(p.a, color=C_ZONE, lw=1.2, zorder=4)
    if x_oa is not None:
        ax.axhline(x_oa, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.tick_params(labelsize=5.5, length=2, width=0.5)
    ax.set_yticks([0, 0.5, 1])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if label:
        ax.set_title(f"$x$={x:.2f}", fontsize=6.5, pad=2)


def draw_frame(traj, t, p, title=None, x_oa=None, figsize=(3.2, 2.6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.05, 0.72, 0.82])
    gx = fig.add_axes([0.82, 0.08, 0.07, 0.72])
    draw_grid(ax, traj, t, p, title=title)
    draw_gauge(gx, traj, t, p, x_oa=x_oa)
    return fig, (ax, gx)


def save_animation(traj, p, path, title="", every=2, fps=12, x_oa=None):
    """Write a GIF (Pillow) or MP4 (ffmpeg) of the recorded episode."""
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(4.0, 3.2), dpi=110)
    ax = fig.add_axes([0.02, 0.05, 0.72, 0.82])
    gx = fig.add_axes([0.82, 0.08, 0.07, 0.72])
    frames = list(range(0, traj.stock.shape[0], every))

    def update(t):
        ax.clear(); gx.clear()
        draw_grid(ax, traj, t, p, title=f"{title}  t={t}")
        draw_gauge(gx, traj, t, p, x_oa=x_oa)
    anim = FuncAnimation(fig, update, frames=frames)
    writer = "ffmpeg" if str(path).endswith(".mp4") else "pillow"
    anim.save(path, writer=writer, fps=fps)
    plt.close(fig)
