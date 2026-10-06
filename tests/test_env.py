"""Invariant tests (roadmap Phase 5.7)."""
import jax
import jax.numpy as jnp
import numpy as np

from shoal import env as E
from shoal import policies as P
from shoal.reference import all_reference_points, msy
from shoal.rollout import run_episodes

KEY = jax.random.PRNGKey(0)


def _rand_actions(k, p):
    return jax.random.randint(k, (p.n_agents,), 0, E.N_ACTIONS)


def test_stock_balance():
    """Delta stock = regrowth - harvest (diffusion conserves mass)."""
    for name in ["degenerate", "full"]:
        p = E.config(name, punishment=True)
        s = E.reset(KEY, p)
        k = KEY
        for _ in range(50):
            k, ka, ks = jax.random.split(k, 3)
            a = _rand_actions(ka, p)
            s2, r, d, info = E.step(ks, s, a, p)
            lhs = s2.stock.sum() - s.stock.sum()
            rhs = info["growth"] - info["catch"].sum()
            assert abs(float(lhs - rhs)) < 1e-4, (name, float(lhs), float(rhs))
            s = s2


def test_diffusion_conserves_mass():
    st = jax.random.uniform(KEY, (6, 6))
    assert abs(float(E.diffuse(st, 0.3).sum() - st.sum())) < 1e-5


def test_threshold_no_recovery():
    """Without harvest, from below A the stock never recovers; above A it does."""
    p = E.config("+T")
    s = E.reset(KEY, p)
    for x0, expect_up in [(0.9 * p.a, False), (1.2 * p.a, True)]:
        st = s._replace(stock=jnp.full_like(s.stock, x0))
        idle = jnp.zeros(p.n_agents, jnp.int32)
        for _ in range(p.horizon):
            st, *_ = E.step(KEY, st, idle, p)
        assert (float(st.stock.mean()) > x0) == expect_up
    assert not all_reference_points(p)["recovers_below_a"]


def test_hard_zone_admissibility():
    p = E.config("+Z")
    zm = E.zone_masks(p)
    s = E.reset(KEY, p)
    # all agents start in their own zone
    assert bool(zm[jnp.arange(p.n_agents), s.pos[:, 0], s.pos[:, 1]].all())
    # put agent 0 outside its zone and fish: no catch
    s = s._replace(pos=s.pos.at[0].set(jnp.array([0, p.grid - 1])))
    a = jnp.full((p.n_agents,), E.STAY).at[0].set(E.FISH)
    _, r, _, info = E.step(KEY, s, a, p)
    assert float(info["catch"][0]) == 0.0


def test_soft_zone_violation_counted():
    p = E.config("+Z", zone_mode="soft", monitor_p=1.0)
    s = E.reset(KEY, p)
    s = s._replace(pos=s.pos.at[0].set(jnp.array([0, p.grid - 1])))
    a = jnp.full((p.n_agents,), E.STAY).at[0].set(E.FISH)
    s2, r, _, info = E.step(KEY, s, a, p)
    assert float(info["catch"][0]) > 0 and float(s2.n_violation) == 1.0
    assert float(r[0]) < float(info["catch"][0]) - p.cost  # fined


def test_punishment():
    p = E.config("degenerate", punishment=True)
    s = E.reset(KEY, p)._replace(pos=jnp.zeros((6, 2), jnp.int32))
    a = jnp.full((6,), E.STAY).at[0].set(E.PUNISH)
    s2, r, _, _ = E.step(KEY, s, a, p)
    assert abs(float(r[0]) + p.punish_cost) < 1e-6 and bool((s2.frozen[1:] == p.freeze_steps).all())
    # disabled punishment is a no-op
    p0 = E.config("degenerate")
    s3, r3, _, _ = E.step(KEY, s, a, p0)
    assert float(r3[0]) == 0.0 and int(s3.frozen.sum()) == 0


def test_seed_determinism():
    p = E.config("full", zone_mode="soft", punishment=True)
    out = [run_episodes(KEY, p, lambda k, s, o: _rand_actions(k, p), 2)[0] for _ in range(2)]
    for k in out[0]:
        np.testing.assert_array_equal(np.asarray(out[0][k]), np.asarray(out[1][k]))


def test_observation_shapes():
    p = E.config("full")
    s = E.reset(KEY, p)
    assert E.observe(s, p).shape == (p.n_agents, E.obs_dim(p))
    assert E.global_state(s, p).shape == (E.global_state_dim(p),)


def test_degenerate_greedy_depletes_quota_sustains():
    """Harvest-like behaviour: open-access (greedy) depletes; quota sustains; threshold collapses."""
    p = E.config("degenerate")
    g = run_episodes(KEY, p, lambda k, s, o: P.greedy(k, s, p), 4)[0]
    q = run_episodes(KEY, p, lambda k, s, o: P.quota(k, s, p, msy(p)["x_msy"]), 4)[0]
    assert float(g["final_stock"].mean()) < 0.6 * msy(p)["x_msy"]      # over-exploited
    assert float(q["final_stock"].mean()) > 0.35
    pt = E.config("+T")
    gt = run_episodes(KEY, pt, lambda k, s, o: P.greedy(k, s, pt), 4)[0]
    rt = run_episodes(KEY, pt, lambda k, s, o: P.random(k, s, pt), 4)[0]
    assert float(gt["collapse"].mean()) == 1.0     # open access below A -> collapse
    assert float(rt["collapse"].mean()) == 0.0     # non-trivial regime


def test_render_smoke(tmp_path):
    """Recording keeps T+1 states; a frame can be drawn and saved."""
    from shoal.render import draw_frame, record_episode
    p = E.config("full", horizon=20)
    tr = record_episode(KEY, p, lambda k, s, o: P.greedy(k, s, p))
    assert tr.stock.shape == (21, p.grid, p.grid) and tr.actions.shape == (20, p.n_agents)
    np.testing.assert_allclose(tr.stock[0], p.x0)
    fig, _ = draw_frame(tr, 10, p, title="t=10")
    fig.savefig(tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 0
