"""Phase 3 — train the reactive avoidance policy with from-scratch PPO.

Collects fixed-length rollouts across randomized worlds, runs PPO updates, and
logs an episodic-return learning curve. Supports --resume to continue from a
checkpoint (so long runs can be split across sessions). CPU-only.
"""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vantage.rl.avoidance_env import AvoidanceEnv
from vantage.rl.ppo import PPOAgent
from vantage.utils.report import write_csv

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
CKPT = os.path.join(RESULTS, "phase3_policy.npz")
CURVE = os.path.join(RESULTS, "phase3_curve.csv")


def collect(agent, env, steps, seed_base):
    obs_b, raw_b, logp_b, rew_b, val_b, done_b = [], [], [], [], [], []
    ep_returns, ep_succ = [], []
    o, _ = env.reset(seed=seed_base)
    ep_ret, ep_n = 0.0, 0
    for i in range(steps):
        a, extra = agent.act(o)
        raw, logp = extra
        v = agent.value_of(o)
        o2, r, term, trunc, info = env.step(a)
        obs_b.append(o); raw_b.append(raw); logp_b.append(logp)
        rew_b.append(r); val_b.append(v); done_b.append(float(term))
        ep_ret += r
        o = o2
        if term or trunc:
            ep_returns.append(ep_ret)
            ep_succ.append(1.0 if info.get("event") == "goal" else 0.0)
            ep_ret = 0.0; ep_n += 1
            o, _ = env.reset(seed=seed_base + 1000 + ep_n)
    last_v = agent.value_of(o)
    adv, ret = agent.compute_gae(rew_b, val_b, done_b, last_v)
    batch = dict(obs=obs_b, raw=raw_b, logp=logp_b, adv=adv, ret=ret)
    return batch, ep_returns, ep_succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    env = AvoidanceEnv(n_obstacles=6, seed=0)
    agent = PPOAgent(env.obs_dim, env.act_dim, seed=0, lr=4e-4, ent_coef=0.004)
    history = []
    if args.resume and os.path.exists(CKPT):
        agent.load(CKPT)
        if os.path.exists(CURVE):
            import csv
            with open(CURVE) as f:
                history = [(int(r["update"]), float(r["mean_return"]), float(r["success_rate"]))
                           for r in csv.DictReader(f)]
        print(f"resumed from {CKPT} ({len(history)} updates logged)")

    start_u = history[-1][0] + 1 if history else 0
    t0 = time.time()
    for u in range(start_u, start_u + args.updates):
        batch, rets, succ = collect(agent, env, args.steps, seed_base=u * 7919)
        agent.update(batch, epochs=8, minibatch=512)
        mr = float(np.mean(rets)) if rets else float("nan")
        sr = float(np.mean(succ)) if succ else 0.0
        history.append((u, mr, sr))
        if u % 5 == 0 or u == start_u + args.updates - 1:
            print(f"update {u:3d} | mean_return {mr:7.2f} | success {sr:4.2f} "
                  f"| eps {len(rets):3d} | std {np.exp(agent.log_std).mean():.2f} "
                  f"| {time.time()-t0:5.1f}s")
    agent.save(CKPT)
    write_csv(CURVE, [{"update": u, "mean_return": round(mr, 3), "success_rate": round(sr, 3)}
                      for (u, mr, sr) in history], ["update", "mean_return", "success_rate"])

    # learning curve plot
    h = np.array(history)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(h[:, 0], h[:, 1], color="tab:blue", label="mean episodic return")
    ax1.set_xlabel("PPO update"); ax1.set_ylabel("mean return", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(h[:, 0], h[:, 2], color="tab:green", alpha=0.7, label="success rate")
    ax2.set_ylabel("success rate", color="tab:green"); ax2.set_ylim(0, 1.05)
    ax1.set_title("VANTAGE Phase 3 — PPO avoidance learning curve")
    ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "phase3_curve.png"), dpi=120)
    print("saved checkpoint + curve. final success %.2f" % history[-1][2])


if __name__ == "__main__":
    main()
