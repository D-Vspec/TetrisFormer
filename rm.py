"""Reward transformer (Option B): byte tokens = value emb + address emb + time emb over K addresses x T steps.
Block-causal: tokens of step t see all tokens of steps <= t and predict the same address at step t+1."""
import argparse, glob, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class RewardModel(nn.Module):
    def __init__(self, addrs, T=8, d=128, layers=4, heads=4):
        super().__init__()
        K = len(addrs)
        self.T, self.K = T, K
        self.register_buffer("addrs", torch.as_tensor(addrs, dtype=torch.long))
        self.val, self.adr, self.tim = nn.Embedding(256, d), nn.Embedding(K, d), nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, 256)
        step = torch.arange(T * K) // K
        self.register_buffer("mask", step[None, :] > step[:, None])  # True = blocked

    def forward(self, x):  # x: [B, t<=T, K] uint8 values at self.addrs -> logits [B, t, K, 256]
        B, t, K = x.shape
        h = self.val(x.long()) + self.adr.weight[None, None] + self.tim.weight[:t, None]
        h = self.tf(h.reshape(B, t * K, -1), mask=self.mask[: t * K, : t * K])
        return self.head(self.norm(h)).reshape(B, t, K, 256)

    @torch.no_grad()
    def logp_next(self, hist, nxt):
        """hist [B, T, K], nxt [B, K] -> per-byte log p(nxt | hist) [B, K]."""
        lp = F.log_softmax(self(hist)[:, -1].float(), -1)
        return lp.gather(-1, nxt.long()[..., None]).squeeze(-1)


def load(kind, addrs, episodes=None):
    fs = sorted(glob.glob(f"data/traces/{kind}_*.npz"))
    fs = fs if episodes is None else [fs[i] for i in episodes]
    return [np.load(f)["ram"][:, addrs] for f in fs]


def windows(eps, n, T, rng):
    """n random windows of T+1 consecutive steps."""
    lens = np.array([len(e) for e in eps]); ok = np.flatnonzero(lens > T + 1)
    out = np.empty((n, T + 1, eps[0].shape[1]), np.uint8)
    for j, i in enumerate(rng.choice(ok, n, p=lens[ok] / lens[ok].sum())):
        s = rng.integers(len(eps[i]) - T); out[j] = eps[i][s:s + T + 1]
    return torch.as_tensor(out, device=DEV)


def nll(model, w):  # per-byte NLL (nats) over all predicted positions
    logits = model(w[:, :-1])
    return F.cross_entropy(logits.reshape(-1, 256).float(), w[:, 1:].reshape(-1).long(), reduction="none").view(w.shape[0], -1)


@torch.no_grad()
def evaluate(model, eps, T, n=512, seed=0):
    rng, out = np.random.default_rng(seed), []
    for _ in range(n // 64):
        out.append(nll(model, windows(eps, 64, T, rng)).mean().item())
    return float(np.mean(out))


def copy_baseline(eps):
    """Best eps-smoothed copy-last-state NLL: p(same)=1-e, e=empirical change rate, rest uniform over 255."""
    X = np.concatenate([(e[1:] != e[:-1]) for e in eps]); e = X.mean()
    return float(-(1 - e) * math.log(1 - e) - e * math.log(e / 255)), float(e)


def unigram_baseline(train, val):
    K = train[0].shape[1]; tr, va = np.concatenate(train), np.concatenate(val)
    tot = 0.0
    for k in range(K):
        p = (np.bincount(tr[:, k], minlength=256) + 0.1) / (len(tr) + 25.6)
        tot += -np.log(p[va[:, k]]).mean()
    return float(tot / K)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rm")
    ap.add_argument("--nomask", action="store_true", help="include volatile bytes (ablation)")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    addrs = np.load("data/rm_addresses.npy")
    if args.nomask:
        addrs = np.union1d(addrs, np.flatnonzero(np.load("data/masks/tetris_volatile.npy")))
    n_eps = len(glob.glob("data/traces/expert_*.npz"))
    val_ids = list(range(0, n_eps, 10)); tr_ids = [i for i in range(n_eps) if i not in val_ids]  # split by episode
    train, val = load("expert", addrs, tr_ids), load("expert", addrs, val_ids)
    rand = load("random", addrs)
    model = RewardModel(addrs, T=args.T).to(DEV)
    print("params", sum(p.numel() for p in model.parameters()), "K", len(addrs), "tokens/window", args.T * len(addrs))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    out = f"runs/{args.name}"; os.makedirs(out, exist_ok=True)
    log, t0 = open(f"{out}/log.jsonl", "w"), time.time()
    fixed = windows(train, args.bs, args.T, rng)
    for it in range(args.steps):
        w = fixed if args.overfit else windows(train, args.bs, args.T, rng)
        with torch.autocast("cuda", torch.bfloat16):
            loss = nll(model, w).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 250 == 0 or it == args.steps - 1:
            rec = {"it": it, "train": loss.item(), "tok_s": int((it + 1) * args.bs * args.T * len(addrs) / (time.time() - t0))}
            if not args.overfit:
                with torch.autocast("cuda", torch.bfloat16):
                    rec["val"] = evaluate(model, val, args.T)
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if args.overfit: raise SystemExit
    with torch.autocast("cuda", torch.bfloat16):
        res = {"val_nll": evaluate(model, val, args.T), "random_nll": evaluate(model, rand, args.T),
               "train_nll": evaluate(model, train, args.T)}
    res["copy_nll"], res["change_rate"] = copy_baseline(val)
    res["unigram_nll"] = unigram_baseline(train, val)
    res["K"], res["params"] = len(addrs), sum(p.numel() for p in model.parameters())
    print(res); json.dump(res, open(f"{out}/eval.json", "w"), indent=1)
    torch.save({"state": model.state_dict(), "addrs": addrs, "T": args.T}, f"{out}/model.pt")
