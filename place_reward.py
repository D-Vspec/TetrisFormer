"""D-012 alignment check: the training reward (raw mean log p per button step per placement, no RND/norm),
averaged over all placements of 8 games, for expert / hand / ours / random placements. Same timer_divs as analyze.py."""
import sys
import numpy as np, torch
from env import TetrisEnv
from expert import place, plan, screen, rotations, norm
from ppo import DEV, Policy, policy_addresses
from rl import LearnedReward
R = LearnedReward("runs/rm/model.pt", 1, beta=0.0)
def run(kind, episodes=8):
    pol = None
    if kind not in ("expert", "random"):
        pol = Policy(policy_addresses(), n_act=40).to(DEV); pol.load_state_dict(torch.load(f"runs/{kind}/policy.pt")); pol.eval()
    rng, lps, lines = np.random.default_rng(0), [], []
    for ep in range(episodes):
        env = TetrisEnv(seed=ep); obs = env.reset(timer_div=200 + ep); done = False
        R.hist = torch.as_tensor(obs[None], device=DEV)[:, R.addrs][:, None].repeat(1, R.T, 1)
        while not done:
            if kind == "expert":
                b, f = screen(env); best = plan(b, f) if f.any() else None
                a = 0 if best is None else rotations(norm(np.argwhere(f))).index(best[0]) * 10 + best[1]
            elif kind == "random": a = int(rng.integers(40))
            else:
                with torch.no_grad(): a = pol(torch.as_tensor(obs[None], device=DEV))[0].sample().item()
            obs, done, info, subs = place(env, a)
            if not done: lps.append(float(R.placement(obs[None], np.array([done]), [{"subs": subs}])[0]))
        lines.append(info["lines"])
    print(f"{kind:22s} reward/placement mean {np.mean(lps):7.2f} median {np.median(lps):7.2f}  lines/game {np.mean(lines):5.2f}  placements {len(lps)}", flush=True)
for k in sys.argv[1:]: run(k)
