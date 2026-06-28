"""Phase 3 tests: env contract, collision/goal termination, PPO learns on a toy task."""
import numpy as np

from vantage.rl.avoidance_env import AvoidanceEnv
from vantage.rl.ppo import PPOAgent


def test_env_api_shapes():
    env = AvoidanceEnv(seed=0)
    o, _ = env.reset(seed=0)
    assert o.shape == (env.obs_dim,)
    o2, r, term, trunc, info = env.step(np.zeros(env.act_dim))
    assert o2.shape == (env.obs_dim,)
    assert isinstance(r, float) and "clearance" in info


def test_goal_termination_straight_shot():
    # no obstacles -> driving toward goal should terminate with 'goal'
    env = AvoidanceEnv(n_obstacles=0, seed=1)
    env.reset(seed=1)
    event = "timeout"
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step([1.0, 0.0])
        if term or trunc:
            event = info.get("event"); break
    assert event == "goal"


def test_ppo_improves_on_toy():
    # a few updates on an obstacle-free task must raise mean return
    env = AvoidanceEnv(n_obstacles=0, seed=2)
    agent = PPOAgent(env.obs_dim, env.act_dim, seed=2, lr=4e-4)

    def mean_return(n=10):
        tot = []
        for s in range(n):
            o, _ = env.reset(seed=100 + s); ep = 0.0
            for _ in range(env.max_steps):
                a, _ = agent.act(o, deterministic=True)
                o, r, term, trunc, _ = env.step(a); ep += r
                if term or trunc:
                    break
            tot.append(ep)
        return np.mean(tot)

    before = mean_return()
    for u in range(20):
        o, _ = env.reset(seed=u * 13)
        O, RA, LP, RW, VL, DN = [], [], [], [], [], []
        for _ in range(1500):
            a, (raw, logp) = agent.act(o)
            v = agent.value_of(o)
            o2, r, term, trunc, _ = env.step(a)
            O.append(o); RA.append(raw); LP.append(logp); RW.append(r); VL.append(v); DN.append(float(term))
            o = o2
            if term or trunc:
                o, _ = env.reset(seed=u * 13 + len(O))
        adv, ret = agent.compute_gae(RW, VL, DN, agent.value_of(o))
        agent.update(dict(obs=O, raw=RA, logp=LP, adv=adv, ret=ret), epochs=6, minibatch=512)
    after = mean_return()
    assert after > before + 2.0, f"PPO did not improve: {before:.2f} -> {after:.2f}"
