"""Minimal PPO implemented from scratch in NumPy (no deep-learning framework).

Two small tanh-MLPs (policy mean + value), a learnable diagonal log-std, GAE,
the clipped surrogate objective, an entropy bonus, and a hand-written Adam.
Kept intentionally compact and dependency-free so training runs on CPU and the
whole repo stays pip-installable for CI.
"""
from __future__ import annotations

import numpy as np


class MLP:
    """Tanh-hidden MLP with manual forward/backward and Adam."""

    def __init__(self, sizes, out_scale=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = out_scale if i == len(sizes) - 2 else np.sqrt(2.0 / sizes[i])
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1])) * scale)
            self.b.append(np.zeros(sizes[i + 1]))
        self._init_adam()

    def _init_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.adam_t = 0

    def forward(self, x):
        self.cache = [x]
        h = x
        for i in range(len(self.W) - 1):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z)
            self.cache.append(h)
        out = h @ self.W[-1] + self.b[-1]
        return out

    def backward(self, d_out):
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        h = self.cache[-1]
        gW[-1] = h.T @ d_out
        gb[-1] = d_out.sum(0)
        d = d_out @ self.W[-1].T
        for i in range(len(self.W) - 2, -1, -1):
            d = d * (1 - self.cache[i + 1] ** 2)   # tanh'
            gW[i] = self.cache[i].T @ d
            gb[i] = d.sum(0)
            d = d @ self.W[i].T
        return gW, gb

    def adam_step(self, gW, gb, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, clip=0.5):
        # global-norm grad clipping
        total = np.sqrt(sum((g ** 2).sum() for g in gW) + sum((g ** 2).sum() for g in gb))
        scale = clip / (total + 1e-8) if total > clip else 1.0
        self.adam_t += 1
        t = self.adam_t
        for i in range(len(self.W)):
            for m, v, g, p in ((self.mW, self.vW, gW, self.W), (self.mb, self.vb, gb, self.b)):
                gi = g[i] * scale
                m[i] = b1 * m[i] + (1 - b1) * gi
                v[i] = b2 * v[i] + (1 - b2) * gi ** 2
                mhat = m[i] / (1 - b1 ** t)
                vhat = v[i] / (1 - b2 ** t)
                p[i] -= lr * mhat / (np.sqrt(vhat) + eps)


class PPOAgent:
    def __init__(self, obs_dim, act_dim, hidden=64, seed=0,
                 lr=3e-4, clip=0.2, gamma=0.99, lam=0.95, ent_coef=0.005):
        self.policy = MLP([obs_dim, hidden, hidden, act_dim], seed=seed)
        self.value = MLP([obs_dim, hidden, hidden, 1], out_scale=1.0, seed=seed + 1)
        self.log_std = np.full(act_dim, -0.5)
        self.m_ls = np.zeros(act_dim); self.v_ls = np.zeros(act_dim); self.t_ls = 0
        self.clip = clip; self.gamma = gamma; self.lam = lam
        self.ent_coef = ent_coef; self.lr = lr
        self.act_dim = act_dim

    # -------------------------------------------------------------- sampling
    def act(self, obs, deterministic=False):
        mean = self.policy.forward(obs[None])[0]
        if deterministic:
            return np.tanh(mean), None
        std = np.exp(self.log_std)
        raw = mean + std * np.random.standard_normal(self.act_dim)
        logp = self._logp(mean, raw)
        return np.tanh(raw), (raw, logp)

    def _logp(self, mean, raw):
        std = np.exp(self.log_std)
        return float((-0.5 * ((raw - mean) / std) ** 2 - self.log_std
                      - 0.5 * np.log(2 * np.pi)).sum())

    def value_of(self, obs):
        return float(self.value.forward(obs[None])[0, 0])

    # -------------------------------------------------------------------- GAE
    def compute_gae(self, rewards, values, dones, last_value):
        adv = np.zeros(len(rewards))
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            adv[t] = gae
        ret = adv + np.asarray(values)
        return adv, ret

    # ----------------------------------------------------------------- update
    def update(self, batch, epochs=10, minibatch=256):
        obs = np.asarray(batch["obs"])
        raw = np.asarray(batch["raw"])
        old_logp = np.asarray(batch["logp"])
        adv = np.asarray(batch["adv"])
        ret = np.asarray(batch["ret"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n, minibatch):
                mb = idx[s:s + minibatch]
                o, r, ol, a, rt = obs[mb], raw[mb], old_logp[mb], adv[mb], ret[mb]
                mean = self.policy.forward(o)
                std = np.exp(self.log_std)
                logp = (-0.5 * ((r - mean) / std) ** 2 - self.log_std
                        - 0.5 * np.log(2 * np.pi)).sum(1)
                ratio = np.exp(logp - ol)
                clip_ratio = np.clip(ratio, 1 - self.clip, 1 + self.clip)
                use_unclipped = (ratio * a <= clip_ratio * a)
                eff_ratio = np.where(use_unclipped, ratio, 0.0)
                # d(-L_clip)/d mean  (maximize surrogate -> minimize negative)
                coef = -((eff_ratio * a)[:, None]) / (std ** 2)[None, :]   # (n,act)
                d_mean = coef * (r - mean)
                # entropy bonus pushes log_std up (handled below); mean unaffected
                self.policy.adam_step(*self.policy.backward(d_mean / len(mb)), lr=self.lr)

                # log_std gradient (surrogate + entropy)
                d_logp_ls = ((r - mean) ** 2 / std ** 2 - 1.0)
                g_ls = -(eff_ratio * a)[:, None] * d_logp_ls
                g_ls = g_ls.mean(0) - self.ent_coef  # entropy = sum(log_std)+const
                self._adam_logstd(g_ls)

                # value regression
                v = self.value.forward(o)[:, 0]
                dv = (v - rt)[:, None] / len(mb)
                self.value.adam_step(*self.value.backward(dv), lr=self.lr)
        self.log_std = np.clip(self.log_std, -2.0, 0.5)

    def _adam_logstd(self, g, lr=None, b1=0.9, b2=0.999, eps=1e-8):
        lr = lr or self.lr
        self.t_ls += 1
        self.m_ls = b1 * self.m_ls + (1 - b1) * g
        self.v_ls = b2 * self.v_ls + (1 - b2) * g ** 2
        mhat = self.m_ls / (1 - b1 ** self.t_ls)
        vhat = self.v_ls / (1 - b2 ** self.t_ls)
        self.log_std -= lr * mhat / (np.sqrt(vhat) + eps)

    # ------------------------------------------------------------- save/load
    def save(self, path):
        d = {}
        for i, (w, b) in enumerate(zip(self.policy.W, self.policy.b)):
            d[f"pW{i}"] = w; d[f"pb{i}"] = b
        for i, (w, b) in enumerate(zip(self.value.W, self.value.b)):
            d[f"vW{i}"] = w; d[f"vb{i}"] = b
        d["log_std"] = self.log_std
        np.savez(path, **d)

    def load(self, path):
        d = np.load(path)
        for i in range(len(self.policy.W)):
            self.policy.W[i] = d[f"pW{i}"]; self.policy.b[i] = d[f"pb{i}"]
        for i in range(len(self.value.W)):
            self.value.W[i] = d[f"vW{i}"]; self.value.b[i] = d[f"vb{i}"]
        self.log_std = d["log_std"]
