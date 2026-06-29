"""Optional GPU trainer for the avoidance policy (PyTorch, CUDA-aware).

Same environment and PPO algorithm as the NumPy trainer, but vectorized in
PyTorch so it uses your GPU automatically when available (e.g. an RTX 4050).
The NumPy trainer (scripts/phase3_train.py) remains the dependency-light path
used in CI; this one is for faster local experiments.

    pip install "vantage[gpu]"        # installs torch
    python scripts/phase3_train_torch.py --updates 300

It prints the selected device so you can confirm the GPU is being used
(watch `nvidia-smi` to see utilization climb).
"""
from __future__ import annotations

import argparse
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch not installed. Run:  pip install \"vantage[gpu]\"")

from vantage.rl.avoidance_env import AvoidanceEnv

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"[phase3-torch] device = {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    env = AvoidanceEnv(n_obstacles=6, seed=0)
    net = ActorCritic(env.obs_dim, env.act_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=4e-4)
    gamma, lam, clip = 0.99, 0.95, 0.2

    for u in range(args.updates):
        O, A, LP, R, V, D = [], [], [], [], [], []
        o, _ = env.reset(seed=u * 7919); ep_ret, rets, succ = 0.0, [], []
        for _ in range(args.steps):
            ot = torch.as_tensor(o, dtype=torch.float32, device=dev)
            with torch.no_grad():
                mean, val = net(ot)
                std = net.log_std.exp()
                raw = mean + std * torch.randn_like(std)
                logp = (-0.5 * ((raw - mean) / std) ** 2 - net.log_std
                        - 0.5 * np.log(2 * np.pi)).sum()
            a = torch.tanh(raw).cpu().numpy()
            o2, r, term, trunc, info = env.step(a)
            O.append(o); A.append(raw.cpu().numpy()); LP.append(float(logp))
            R.append(r); V.append(float(val)); D.append(float(term))
            ep_ret += r; o = o2
            if term or trunc:
                rets.append(ep_ret); succ.append(info.get("event") == "goal")
                ep_ret = 0.0; o, _ = env.reset(seed=u * 7919 + len(rets))
        # GAE
        adv = np.zeros(len(R)); gae = 0.0
        with torch.no_grad():
            last_v = float(net(torch.as_tensor(o, dtype=torch.float32, device=dev))[1])
        for t in reversed(range(len(R))):
            nv = last_v if t == len(R) - 1 else V[t + 1]
            nt = 1.0 - D[t]
            delta = R[t] + gamma * nv * nt - V[t]
            gae = delta + gamma * lam * nt * gae; adv[t] = gae
        ret = adv + np.array(V)
        Ot = torch.as_tensor(np.array(O), dtype=torch.float32, device=dev)
        At = torch.as_tensor(np.array(A), dtype=torch.float32, device=dev)
        LPt = torch.as_tensor(np.array(LP), dtype=torch.float32, device=dev)
        advt = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32, device=dev)
        rett = torch.as_tensor(ret, dtype=torch.float32, device=dev)
        for _ in range(8):
            mean, val = net(Ot)
            std = net.log_std.exp()
            logp = (-0.5 * ((At - mean) / std) ** 2 - net.log_std - 0.5 * np.log(2 * np.pi)).sum(1)
            ratio = (logp - LPt).exp()
            l1 = ratio * advt
            l2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advt
            loss = -torch.min(l1, l2).mean() + 0.5 * (val - rett).pow(2).mean() \
                   - 0.004 * net.log_std.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if u % 10 == 0:
            sr = np.mean(succ) if succ else 0.0
            print(f"update {u:3d} | return {np.mean(rets):7.2f} | success {sr:4.2f}")
    os.makedirs(RESULTS, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(RESULTS, "phase3_policy_torch.pt"))
    print("saved torch checkpoint")


if __name__ == "__main__":
    main()
