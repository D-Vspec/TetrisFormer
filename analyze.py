"""PoC P6: results table + per-address reward attribution (mean log p per byte) on expert / random / agent traces."""
import glob, json, os, sys
import numpy as np
import torch, torch.nn.functional as F
from env import TetrisEnv
from ppo import DEV, Policy, policy_addresses
from rm import RewardModel, load

RUNS = sys.argv[2:] or ["hand_final", "pilot_a", "ours_nomask", "curiosity"]  # pilot_a = "ours" (D-009)
OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/analysis.json"  # D-010: `analyze.py runs/analysis_3m.json <runs...>`


def agent_traces(run, episodes=8):
    pol = Policy(policy_addresses()).to(DEV); pol.load_state_dict(torch.load(f"runs/{run}/policy.pt")); pol.eval()
    out = []
    for ep in range(episodes):
        env = TetrisEnv(seed=ep); obs = env.reset(timer_div=200 + ep); tr, done = [obs], False
        while not done:
            with torch.no_grad():
                a = pol(torch.as_tensor(obs[None], device=DEV))[0].sample().item()
            obs, done, _ = env.step(a); tr.append(obs)
        out.append(np.stack(tr))
    return out


@torch.no_grad()
def per_address_logp(m, eps, T, n=256, seed=0):
    """Mean log p per address over n random windows' last-step predictions."""
    rng, acc, cnt, sums = np.random.default_rng(seed), 0, 0, []
    eps = [e[:-1] for e in eps if len(e) > T + 2]  # drop the terminal transition (excluded from online reward too)
    for _ in range(n // 32):
        w = []
        for i in rng.integers(len(eps), size=32):
            s = rng.integers(len(eps[i]) - T); w.append(eps[i][s:s + T + 1])
        w = torch.as_tensor(np.stack(w), device=DEV)
        with torch.autocast("cuda", torch.bfloat16):
            lp = m.logp_next(w[:, :T], w[:, T]).float()
            acc = acc + lp.sum(0); cnt += len(w); sums += lp.sum(1).tolist()
    return (acc / cnt).cpu().numpy(), float(np.median(sums))


if __name__ == "__main__":
    ck = torch.load("runs/rm/model.pt", weights_only=False)
    m = RewardModel(ck["addrs"], T=ck["T"]).to(DEV).eval(); m.load_state_dict(ck["state"])
    addrs, T = ck["addrs"], ck["T"]
    rand_eval = json.load(open("runs/random_eval.json")) if os.path.exists("runs/random_eval.json") else None
    rows = [("random", rand_eval)] + [(r, json.load(open(f"runs/{r}/eval.json"))) for r in RUNS if os.path.exists(f"runs/{r}/eval.json")]
    sets = {"expert_val": load("expert", addrs, list(range(0, 100, 10))), "random": load("random", addrs)}
    for r, _ in rows[1:]:
        sets[r] = [e[:, addrs] for e in agent_traces(r)]
    res = {k: per_address_logp(m, v, T) for k, v in sets.items()}
    lp = {k: v[0] for k, v in res.items()}
    report = {"table": {r: e for r, e in rows}, "mean_logp_per_step": {k: float(v.sum()) for k, v in lp.items()},
              "median_logp_per_step": {k: v[1] for k, v in res.items()}, "top_addresses": {}}
    for k, v in lp.items():
        if k == "expert_val": continue
        gap = lp["expert_val"] - v  # where this policy's states are least expert-like
        top = np.argsort(-gap)[:10]
        report["top_addresses"][k] = [(hex(0xC000 + int(addrs[i])), round(float(gap[i]), 3)) for i in top]
    json.dump(report, open(OUT, "w"), indent=1)
    print(json.dumps(report, indent=1))
