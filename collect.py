"""Collect full-WRAM traces: scripted expert (D-005) and uniform-random play. One compressed npz per episode."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from env import TetrisEnv
from expert import rollout


def random_rollout(seed):
    env, rng = TetrisEnv(seed), np.random.default_rng(seed)
    trace, done = [env.reset(timer_div=seed % 256)], False
    while not done:
        obs, done, info = env.step(int(rng.integers(8)))
        trace.append(obs)
    return np.stack(trace), info


def job(arg):
    kind, i = arg
    tr, info = rollout(i % 256) if kind == "expert" else random_rollout(10_000 + i)
    np.savez_compressed(f"data/traces/{kind}_{i:04d}.npz", ram=tr)
    return kind, i, len(tr), info["lines"]


if __name__ == "__main__":
    n_exp, n_rand = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs("data/traces", exist_ok=True)
    jobs = [("expert", i) for i in range(n_exp)] + [("random", i) for i in range(n_rand)]
    with Pool(4) as p:
        res = p.map(job, jobs)
    meta = [dict(zip(("kind", "episode", "steps", "lines"), r)) for r in res]
    json.dump(meta, open("data/traces/meta.json", "w"), indent=0)
    for k in ("expert", "random"):
        m = [r for r in meta if r["kind"] == k]
        print(k, len(m), "eps", sum(r["steps"] for r in m), "steps", np.mean([r["lines"] for r in m]), "lines/ep")
