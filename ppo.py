"""PPO on Tetris RAM. Reward is pluggable; policy/PPO are identical across reward types."""
import argparse, json, os, subprocess, time
import numpy as np
import torch, torch.nn as nn
from env import VecEnv

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def policy_addresses():
    """Policy input: WRAM bytes that changed in no-op/random rollouts, minus the volatile mask (fixed for all runs)."""
    return np.load("data/policy_addresses.npy")


class Policy(nn.Module):
    def __init__(self, addrs, hidden=256, n_act=8):
        super().__init__()
        self.register_buffer("addrs", torch.as_tensor(addrs, dtype=torch.long))
        self.register_buffer("offs", torch.arange(len(addrs)) * 256)
        self.emb = nn.EmbeddingBag(len(addrs) * 256, hidden, mode="sum")  # == one-hot bytes -> Linear
        self.body = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi, self.v = nn.Linear(hidden, n_act), nn.Linear(hidden, 1)

    def forward(self, ram):  # ram: [B, 8192] uint8
        x = ram[:, self.addrs].long() + self.offs
        h = self.body(self.emb(x))
        return torch.distributions.Categorical(logits=self.pi(h)), self.v(h).squeeze(-1)


def hand_reward(infos):
    return np.array([1.0 * i["dlines"] + 0.01 for i in infos], dtype=np.float32)


def evaluate(policy, episodes=24, n_envs=12, seed=999):
    """Sampled-action eval; policy=None -> uniform random."""
    env, done_eps = VecEnv(n_envs, seed=seed), []
    obs = env.reset()
    while len(done_eps) < episodes:
        if policy is None:
            a = np.random.randint(8, size=n_envs)
        else:
            with torch.no_grad():
                a = policy(torch.as_tensor(obs, device=DEV))[0].sample().cpu().numpy()
        obs, done, infos = env.step(a)
        done_eps += [{"lines": i["lines"], "len": i["final_t"]} for d, i in zip(done, infos) if d]
    env.close()
    return {k: float(np.mean([e[k] for e in done_eps])) for k in ("lines", "len")} | {"episodes": len(done_eps)}


def train(args, reward_fn=None, run_hook=None):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = f"runs/{args.name}"; os.makedirs(out, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(vars(args) | {"commit": commit}, open(f"{out}/config.json", "w"), indent=1)
    log = open(f"{out}/log.jsonl", "w")

    env = VecEnv(args.envs, seed=args.seed)
    pol = Policy(policy_addresses()).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr, eps=1e-5)
    N, S = args.envs, args.rollout
    obs = env.reset()
    ep_lines, ep_lens, step, t0 = [], [], 0, time.time()
    reward_fn = reward_fn or (lambda prev, obs, done, infos: hand_reward(infos))

    while step < args.steps:
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        for _ in range(S):
            o = torch.as_tensor(obs, device=DEV)
            with torch.no_grad():
                dist, v = pol(o)
                a = dist.sample()
            nobs, done, infos = env.step(a.cpu().numpy())
            r = reward_fn(obs, nobs, done, infos)
            for k, x in zip(buf, (o, a, dist.log_prob(a), v, torch.as_tensor(r, device=DEV), torch.as_tensor(done, device=DEV, dtype=torch.float32))):
                buf[k].append(x)
            for d, i in zip(done, infos):
                if d: ep_lines.append(i["lines"]); ep_lens.append(i["final_t"])
            obs = nobs
        step += N * S
        b = {k: torch.stack(x) for k, x in buf.items()}  # [S, N, ...]
        with torch.no_grad():
            nv = pol(torch.as_tensor(obs, device=DEV))[1]
        adv, last = torch.zeros_like(b["rew"]), 0
        for t in reversed(range(S)):  # GAE
            nxt = nv if t == S - 1 else b["val"][t + 1]
            nonterm = 1 - b["done"][t]
            delta = b["rew"][t] + args.gamma * nxt * nonterm - b["val"][t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + b["val"]
        flat = {k: v.reshape(N * S, *v.shape[2:]) for k, v in b.items()} | {"adv": adv.flatten(), "ret": ret.flatten()}
        for _ in range(args.epochs):
            for idx in torch.randperm(N * S, device=DEV).split(N * S // args.minibatches):
                dist, v = pol(flat["obs"][idx])
                lp = dist.log_prob(flat["act"][idx])
                ratio = (lp - flat["logp"][idx]).exp()
                a_ = flat["adv"][idx]; a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pl = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
                loss = pl + 0.5 * (v - flat["ret"][idx]).pow(2).mean() - args.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        rec = {"step": step, "sps": int(step / (time.time() - t0)), "rew_mean": float(b["rew"].mean()),
               "ep_lines": float(np.mean(ep_lines[-50:])) if ep_lines else None,
               "ep_len": float(np.mean(ep_lens[-50:])) if ep_lens else None,
               "entropy": float(dist.entropy().mean()),
               "act_hist": np.bincount(flat["act"].cpu().numpy(), minlength=8).tolist()}
        if run_hook: rec |= run_hook()
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step // (N * S)) % 20 == 0: print(rec, flush=True)
    env.close()
    torch.save(pol.state_dict(), f"{out}/policy.pt")
    res = evaluate(pol)
    json.dump(res, open(f"{out}/eval.json", "w")); print("eval", res)
    return pol


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="hand")
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
