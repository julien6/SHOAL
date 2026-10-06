"""Vectorised IPPO / MAPPO with social mechanisms, written for CPU (JaxMARL-style, single jit).

* Parameter sharing across agents (agent id + type in the observation) -- standard in JaxMARL
  IPPO; each agent still optimises its *own* (possibly shaped) reward.
* MAPPO = same actor, centralised critic fed with the global state + the agent's own obs.
* Mechanisms (one at a time):
    ippo / mappo : selfish reward
    ia           : inequity aversion on temporally smoothed rewards (Hughes et al. 2018)
    punish       : costly punishment action enabled, selfish reward (Perolat et al. 2017)
    svo          : heterogeneous social value orientation (McKee et al. 2020)
* Episodes are truncated (not terminated) at the horizon: the final value is bootstrapped,
  so agents have no end-game incentive (the episode horizon is not observed).
The whole training run, for all seeds, is one `jax.jit(jax.vmap(train))`.
"""
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import env as E

MECHS = ("ippo", "mappo", "ia", "punish", "svo")


@dataclass(frozen=True)
class TrainCfg:
    num_envs: int = 16
    num_steps: int = 128
    num_updates: int = 300
    epochs: int = 4
    minibatches: int = 4
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    # mechanism hyper-parameters (fixed, not tuned -- same budget for every configuration)
    trace_decay: float = 0.95      # gamma*lambda of the reward traces (Hughes 2018)
    ia_alpha: float = 0.1          # disadvantageous inequity (envy)
    ia_beta: float = 0.1           # advantageous inequity (guilt)
    svo_w: float = 0.05            # weight of the SVO angle mismatch
    svo_low_deg: float = 15.0      # SVO angles spread evenly in [low, high], mean 45 deg
    svo_high_deg: float = 75.0

    def with_(self, **kw):
        return replace(self, **kw)


# ---------------------------------------------------------------- tiny MLPs (no framework)
def _dense_init(key, n_in, n_out, scale):
    w = jax.nn.initializers.orthogonal(scale)(key, (n_in, n_out))
    return dict(w=w, b=jnp.zeros(n_out))


def mlp_init(key, sizes, out_scale):
    keys = jax.random.split(key, len(sizes) - 1)
    return [_dense_init(k, a, b, np.sqrt(2) if i < len(sizes) - 2 else out_scale)
            for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:]))]


def mlp_apply(params, x):
    for l in params[:-1]:
        x = jnp.tanh(x @ l["w"] + l["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


# ---------------------------------------------------------------- mechanisms
def env_for(p: E.EnvParams, mech: str) -> E.EnvParams:
    return p.with_(punishment=(mech == "punish"))


def svo_angles(p: E.EnvParams, cfg: TrainCfg):
    return jnp.deg2rad(jnp.linspace(cfg.svo_low_deg, cfg.svo_high_deg, p.n_agents))


def extra_obs(p, mech, cfg):
    """(N, 1) mechanism feature appended to the observation (own SVO angle, else 0)."""
    if mech == "svo":
        return (svo_angles(p, cfg) / (jnp.pi / 2))[:, None]
    return jnp.zeros((p.n_agents, 1))


def shape_reward(r, traces, mech, p, cfg):
    """r: (..., N) extrinsic; traces: (..., N) smoothed rewards *including* r."""
    N = p.n_agents
    if mech == "ia":
        diff = traces[..., None, :] - traces[..., :, None]  # [i, j] = e_j - e_i
        envy = jnp.maximum(diff, 0).sum(-1) / (N - 1)
        guilt = jnp.maximum(-diff, 0).sum(-1) / (N - 1)
        return r - cfg.ia_alpha * envy - cfg.ia_beta * guilt
    if mech == "svo":
        others = (traces.sum(-1, keepdims=True) - traces) / (N - 1)
        theta_r = jnp.arctan2(others, traces)
        return r - cfg.svo_w * jnp.abs(svo_angles(p, cfg) - theta_r)
    return r


# ---------------------------------------------------------------- networks
class Nets(NamedTuple):
    actor: list
    critic: list


def init_nets(key, p, mech, cfg):
    d_obs = E.obs_dim(p) + 1
    d_crit = d_obs + (E.global_state_dim(p) if mech == "mappo" else 0)
    ka, kc = jax.random.split(key)
    return Nets(mlp_init(ka, [d_obs, cfg.hidden, cfg.hidden, E.N_ACTIONS], 0.01),
                mlp_init(kc, [d_crit, cfg.hidden, cfg.hidden, 1], 1.0))


def critic_in(obs, gs, mech):
    """obs (..., N, D); gs (..., G) global state."""
    if mech != "mappo":
        return obs
    gs = jnp.broadcast_to(gs[..., None, :], obs.shape[:-1] + gs.shape[-1:])
    return jnp.concatenate([obs, gs], -1)


def policy_fn(actor_params, p, mech, cfg):
    """act_fn for rollout.run_episode (stochastic policy)."""
    ex = extra_obs(p, mech, cfg)

    def act(key, state, obs):
        logits = mlp_apply(actor_params, jnp.concatenate([obs, ex], -1))
        return jax.random.categorical(key, logits)
    return act


# ---------------------------------------------------------------- training
class Transition(NamedTuple):
    obs: jnp.ndarray
    cin: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def make_train(p0: E.EnvParams, mech: str, cfg: TrainCfg):
    """Returns train(key, init_nets_or_None) -> (nets, curves). vmap/jit it over seeds."""
    assert mech in MECHS
    p = env_for(p0, mech)
    N, NE = p.n_agents, cfg.num_envs
    ex = extra_obs(p, mech, cfg)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))

    v_reset = jax.vmap(lambda k: E.reset(k, p))
    v_step = jax.vmap(lambda k, s, a: E.step(k, s, a, p))
    v_obs = jax.vmap(lambda s: jnp.concatenate([E.observe(s, p), ex], -1))
    v_gs = jax.vmap(lambda s: E.global_state(s, p))

    def train(key, nets=None):
        key, kn, kr = jax.random.split(key, 3)
        if nets is None:
            nets = init_nets(kn, p, mech, cfg)
        opt = tx.init(nets)
        states = v_reset(jax.random.split(kr, NE))
        traces = jnp.zeros((NE, N))

        def env_step(carry, _):
            nets, states, traces, key = carry
            key, ka, ks, kr = jax.random.split(key, 4)
            obs = v_obs(states)
            cin = critic_in(obs, v_gs(states), mech)
            logits = mlp_apply(nets.actor, obs)
            a = jax.random.categorical(ka, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1)[..., 0]
            v = mlp_apply(nets.critic, cin)[..., 0]
            nstates, r, done, info = v_step(jax.random.split(ks, NE), states, a)
            # mechanism shaping on reward traces
            traces = cfg.trace_decay * traces + r
            rs = shape_reward(r, traces, mech, p, cfg)
            # truncation bootstrap with the pre-reset final value
            v_fin = mlp_apply(nets.critic, critic_in(v_obs(nstates), v_gs(nstates), mech))[..., 0]
            rs = rs + cfg.gamma * v_fin * done[:, None]
            # auto-reset
            rstates = v_reset(jax.random.split(kr, NE))
            sel = lambda x, y: jnp.where(done.reshape((NE,) + (1,) * (x.ndim - 1)), y, x)
            final_stock = nstates.stock.mean((1, 2))
            nstates = jax.tree_util.tree_map(sel, nstates, rstates)
            traces = jnp.where(done[:, None], 0.0, traces)
            tr = Transition(obs, cin, a, logp, v, rs, jnp.broadcast_to(done[:, None], (NE, N)))
            log = dict(stock=final_stock.mean(), ext_r=r.mean(),
                       ep_end_stock=(final_stock * done).sum(), n_done=done.sum())
            return (nets, nstates, traces, key), (tr, log)

        def update(carry, _):
            nets, opt, states, traces, key = carry
            (nets, states, traces, key), (trs, logs) = jax.lax.scan(
                env_step, (nets, states, traces, key), None, cfg.num_steps)
            last_cin = critic_in(v_obs(states), v_gs(states), mech)
            last_v = mlp_apply(nets.critic, last_cin)[..., 0]

            def gae(c, t):
                adv, nv = c
                delta = t.reward + cfg.gamma * nv * (1 - t.done) - t.value
                adv = delta + cfg.gamma * cfg.gae_lambda * (1 - t.done) * adv
                return (adv, t.value), adv
            _, adv = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), trs, reverse=True)
            ret = adv + trs.value

            flat = lambda x: x.reshape((-1,) + x.shape[3:])
            batch = jax.tree_util.tree_map(flat, (trs, adv, ret))
            B = cfg.num_steps * NE * N
            mb = B // cfg.minibatches

            def loss_fn(nets, tr, adv, ret):
                logits = mlp_apply(nets.actor, tr.obs)
                lp_all = jax.nn.log_softmax(logits)
                logp = jnp.take_along_axis(lp_all, tr.action[..., None], -1)[..., 0]
                ratio = jnp.exp(logp - tr.logp)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                v = mlp_apply(nets.critic, tr.cin)[..., 0]
                v_clip = tr.value + jnp.clip(v - tr.value, -cfg.clip, cfg.clip)
                vl = 0.5 * jnp.maximum((v - ret) ** 2, (v_clip - ret) ** 2).mean()
                ent = -(jnp.exp(lp_all) * lp_all).sum(-1).mean()
                return pg + cfg.vf_coef * vl - cfg.ent_coef * ent, (pg, vl, ent)

            def epoch(c, _):
                nets, opt, key = c
                key, kp = jax.random.split(key)
                perm = jax.random.permutation(kp, B)
                shuf = jax.tree_util.tree_map(lambda x: x[perm].reshape((cfg.minibatches, mb) + x.shape[1:]), batch)

                def mbstep(c, b):
                    nets, opt = c
                    grads, aux = jax.grad(loss_fn, has_aux=True)(nets, *b)
                    upd, opt = tx.update(grads, opt, nets)
                    return (optax.apply_updates(nets, upd), opt), aux
                (nets, opt), aux = jax.lax.scan(mbstep, (nets, opt), shuf)
                return (nets, opt, key), aux
            (nets, opt, key), aux = jax.lax.scan(epoch, (nets, opt, key), None, cfg.epochs)
            curves = dict(stock=logs["stock"].mean(), ext_r=logs["ext_r"].mean(),
                          ep_end_stock=logs["ep_end_stock"].sum() / jnp.maximum(logs["n_done"].sum(), 1),
                          entropy=aux[2].mean())
            return (nets, opt, states, traces, key), curves

        (nets, *_), curves = jax.lax.scan(update, (nets, opt, states, traces, key), None, cfg.num_updates)
        return nets, curves

    return train


def train_seeds(p, mech, cfg, seeds, init=None):
    """Train len(seeds) independent runs in one vmapped jit. init: batched Nets or None."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    tr = make_train(p, mech, cfg)
    if init is None:
        return jax.jit(jax.vmap(lambda k: tr(k)))(keys)
    return jax.jit(jax.vmap(tr))(keys, init)
