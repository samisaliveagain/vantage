"""Evaluate a GPU-trained (PyTorch) avoidance policy and write fresh metrics.

Loads results/phase3_policy_torch.pt, runs the deterministic policy over N
randomized worlds, and reports success/collision/steps. Writes a plain-text
file (results/phase3_torch_eval.txt) so progress is readable even when the
training log is held open.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch

# work from any directory: add scripts/ to the path so we can import the
# sibling module; the 'vantage' package comes from the venv install.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from vantage.rl.avoidance_env import AvoidanceEnv
from phase3_train_torch import ActorCritic

RESULTS = os.path.join(_HERE, "..", "results")
CKPT = os.path.join(RESULTS, "phase3_policy_torch.pt")


def main(n=100):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AvoidanceEnv(n_obstacles=6, seed=0)
    net = ActorCritic(env.obs_dim, env.act_dim).to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval()

    succ, coll, steps = [], [], []
    for k in range(n):
        o, _ = env.reset(seed=5000 + k)
        ev = "timeout"
        for t in range(env.max_steps):
            with torch.no_grad():
                mean, _ = net(torch.as_tensor(o, dtype=torch.float32, device=dev))
            a = torch.tanh(mean).cpu().numpy()
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                ev = info.get("event", "timeout"); break
        succ.append(ev == "goal"); coll.append(ev == "collision"); steps.append(t + 1)

    lines = [
        "VANTAGE Phase 3 — GPU-trained (PyTorch) policy evaluation",
        f"device         : {dev} ({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'})",
        f"episodes       : {n}",
        f"success_rate   : {np.mean(succ):.3f}",
        f"collision_rate : {np.mean(coll):.3f}",
        f"mean_steps     : {np.mean(steps):.1f}",
    ]
    out = os.path.join(RESULTS, "phase3_torch_eval.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
