"""PPO with the frozen reward transformer: r = norm(±Σ log p(s_{t+1} | s_{t-T+1..t})) + β·norm(RND).
Policy, PPO and budget come from ppo.py unchanged; only the reward differs."""
import json
import numpy as np
import torch, torch.nn as nn
from ppo import DEV, Policy, parser, policy_addresses, train
from rm import RewardModel


class RunningNorm:
    def __init__(self): self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def __call__(self, x):
        for v in x:  # Welford
            self.n += 1; d = v - self.mean; self.mean += d / self.n; self.m2 += d * (v - self.mean)
        return (x - self.mean) / (np.sqrt(self.m2 / max(self.n - 1, 1)) + 1e-8)


class RND(nn.Module):
    def __init__(self, addrs):
        super().__init__()
        self.target, self.pred = Policy(addrs), Policy(addrs)  # reuse the one-hot encoder; heads unused
        for p in self.target.parameters(): p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=1e-4)

    def feat(self, net, x):
        return net.body(net.emb(x[:, net.addrs].long() + net.offs))

    def bonus(self, x):  # x [N, 8192] -> per-env error; one predictor step on the same batch
        err = (self.feat(self.pred, x) - self.feat(self.target, x).detach()).pow(2).mean(-1)
        self.opt.zero_grad(); err.mean().backward(); self.opt.step()
        return err.detach().cpu().numpy()


class LearnedReward:
    def __init__(self, ckpt, n_envs, sign=1.0, beta=0.5):
        ck = torch.load(ckpt, weights_only=False)
        self.m = RewardModel(ck["addrs"], T=ck["T"]).to(DEV).eval()
        self.m.load_state_dict(ck["state"])
        for p in self.m.parameters(): p.requires_grad_(False)
        self.ck_sum = float(sum(p.sum() for p in self.m.parameters()))
        self.addrs = torch.as_tensor(ck["addrs"], device=DEV)
        self.T, self.sign, self.beta, self.hist = ck["T"], sign, beta, None
        self.norm, self.rnorm = RunningNorm(), RunningNorm()
        self.rnd = RND(policy_addresses()).to(DEV) if beta > 0 else None
        self.raw = []

    def placement(self, obs, done, infos):
        """D-012: per-env MEAN log p over the button steps of one placement, all windows in one batch."""
        wins, owner = [], []
        for i, inf in enumerate(infos):
            seq = torch.cat([self.hist[i], torch.as_tensor(inf["subs"], device=DEV)[:, self.addrs]])  # [T+L, K]
            wins.append(seq.unfold(0, self.T + 1, 1).permute(0, 2, 1))  # [L, T+1, K]
            owner += [i] * (len(seq) - self.T)
            self.hist[i] = seq[-self.T:]
        w, owner = torch.cat(wins), torch.as_tensor(owner, device=DEV)
        with torch.autocast("cuda", torch.bfloat16):
            lp = self.m.logp_next(w[:, :self.T], w[:, self.T]).sum(-1).float()
        n = len(infos)
        return (torch.zeros(n, device=DEV).index_add_(0, owner, lp) / torch.bincount(owner, minlength=n)).cpu().numpy()

    def __call__(self, prev, obs, done, infos):
        if "subs" in infos[0]:  # placement actions (D-012)
            if self.hist is None:
                self.hist = torch.as_tensor(prev, device=DEV)[:, self.addrs][:, None].repeat(1, self.T, 1)
            lp = self.placement(obs, done, infos)
            self.raw.append(lp[~done].mean() if (~done).any() else np.nan)
            r = self.norm(self.sign * lp)
            if self.rnd is not None:
                r = r + self.beta * self.rnorm(self.rnd.bonus(torch.as_tensor(obs, device=DEV)))
            r[done] = 0.0
            nxt = torch.as_tensor(obs, device=DEV)[:, self.addrs]
            d = torch.as_tensor(done, device=DEV)
            self.hist[d] = nxt[d][:, None].repeat(1, self.T, 1)  # new episode: pad history with first state
            return r.astype(np.float32)
        prev = torch.as_tensor(prev, device=DEV)[:, self.addrs]
        nxt_full = torch.as_tensor(obs, device=DEV)
        nxt = nxt_full[:, self.addrs]
        if self.hist is None:
            self.hist = prev[:, None].repeat(1, self.T, 1)
        with torch.autocast("cuda", torch.bfloat16):
            lp = self.m.logp_next(self.hist, nxt).sum(-1).cpu().numpy()
        self.raw.append(lp[~done].mean() if (~done).any() else np.nan)
        r = self.norm(self.sign * lp)
        if self.rnd is not None:
            r = r + self.beta * self.rnorm(self.rnd.bonus(nxt_full))
        r[done] = 0.0  # auto-reset: obs is the next episode's first state, not the true successor
        self.hist = torch.cat([self.hist[:, 1:], nxt[:, None]], 1)
        d = torch.as_tensor(done, device=DEV)
        self.hist[d] = nxt[d][:, None].repeat(1, self.T, 1)  # new episode: pad history with first state
        return r.astype(np.float32)

    def hook(self):
        out = {"raw_logp": float(np.nanmean(self.raw))} if self.raw else {}
        self.raw = []; return out


if __name__ == "__main__":
    p = parser()
    p.add_argument("--rm", default="runs/rm/model.pt")
    p.add_argument("--curiosity", action="store_true")
    p.add_argument("--beta", type=float, default=0.5)
    args = p.parse_args()
    R = LearnedReward(args.rm, args.envs, sign=-1.0 if args.curiosity else 1.0, beta=args.beta)
    s0 = R.ck_sum
    train(args, reward_fn=R, run_hook=R.hook)
    assert abs(float(sum(q.sum() for q in R.m.parameters())) - s0) < 1e-3, "reward model changed"
    print("reward model frozen: checksum unchanged")
