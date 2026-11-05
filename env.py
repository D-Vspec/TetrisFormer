"""PyBoy Tetris env exposing WRAM, plus a subprocess vector env and volatile-byte mask."""
import multiprocessing as mp
import numpy as np
from pyboy import PyBoy

ROM = "roms/tetris.gb"
WRAM = (0xC000, 0xE000)
ACTIONS = ["up", "down", "left", "right", "a", "b", "start", None]  # None = no-op
FRAME_SKIP = 24
MAX_STEPS = 2000


class TetrisEnv:
    def __init__(self, seed=0):
        self.pb = PyBoy(ROM, window="null", sound_emulated=False)
        self.pb.set_emulation_speed(0)
        self.w = self.pb.game_wrapper
        self.rng = np.random.default_rng(seed)
        self.w.start_game(timer_div=0)

    def ram(self):
        return np.array(self.pb.memory[WRAM[0]:WRAM[1]], dtype=np.uint8)

    def reset(self, timer_div=None):
        # timer_div seeds Tetris' piece RNG; random per episode unless given (determinism tests pass it)
        td = int(self.rng.integers(256)) if timer_div is None else timer_div
        self.w.reset_game(timer_div=td)
        self.t, self.lines, self.landed = 0, 0, 0
        return self.ram()

    def step(self, a):
        if ACTIONS[a]:
            self.pb.button(ACTIONS[a])  # press 1 frame, then release
        self.pb.tick(FRAME_SKIP, False)
        self.t += 1
        lines = self.w.lines
        landed = int((np.array(self.pb.tilemap_background[2:12, 0:18]) != 47).sum())  # landed blocks only
        dlines = lines - self.lines
        info = {"lines": lines, "dlines": dlines, "score": self.w.score, "t": self.t,
                "dpieces": int(landed - self.landed == 4)}  # lock = exactly +4 cells (pause overlay adds 28: H-002)
        self.lines, self.landed = lines, landed
        done = self.w.game_over() or self.t >= MAX_STEPS
        return self.ram(), done, info


def _worker(conn, seed, placement=False):
    env = TetrisEnv(seed)
    if placement:
        from expert import place  # D-012 (expert imports env, so import here)
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            conn.send(env.reset(arg))
        elif cmd == "step":
            if placement:
                obs, done, info, info["subs"] = place(env, arg)
            else:
                obs, done, info = env.step(arg)
            if done:
                info["final_t"], obs = env.t, env.reset()  # auto-reset
            conn.send((obs, done, info))
        else:
            break


class VecEnv:
    def __init__(self, n, seed=0, placement=False):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, seed * 1000 + i, placement), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)
        self.n = n

    def reset(self):
        for c in self.conns: c.send(("reset", None))
        return np.stack([c.recv() for c in self.conns])

    def step(self, actions):
        for c, a in zip(self.conns, actions): c.send(("step", int(a)))
        obs, done, infos = zip(*[c.recv() for c in self.conns])
        return np.stack(obs), np.array(done), list(infos)

    def close(self):
        for c in self.conns: c.send(("close", None))


def volatile_mask(episodes=10, thresh=0.6, seed=0):
    """Bytes that change on >= thresh of steps under no-op play (input-independent churn: sound engine, stack).
    Random-play rates are returned too but not thresholded: Start pauses Tetris, which deflates them."""
    env, rng = TetrisEnv(seed), np.random.default_rng(seed)
    rates = []
    for policy in ("noop", "random"):
        changes, steps = np.zeros(8192), 0
        for _ in range(episodes):
            prev, done = env.reset(), False
            while not done:
                a = 7 if policy == "noop" else int(rng.integers(8))
                obs, done, _ = env.step(a)
                if not done:  # skip the game-over transition
                    changes += obs != prev; steps += 1
                prev = obs
        rates.append(changes / steps)
    return rates[0] >= thresh, rates


if __name__ == "__main__":
    import time, json, os
    # determinism: same timer_div + same actions -> identical RAM, across two fresh processes' envs
    acts = np.random.default_rng(1).integers(8, size=150)
    traces = []
    for _ in range(2):
        e = TetrisEnv(); tr = [e.reset(timer_div=7)]
        for a in acts:
            o, d, _ = e.step(a); tr.append(o)
            if d: break
        traces.append(np.stack(tr))
    assert traces[0].shape == traces[1].shape and (traces[0] == traces[1]).all(), "nondeterministic"
    print("determinism OK", traces[0].shape)

    for n in (8, 12):
        v = VecEnv(n); v.reset(); t = time.time()
        for _ in range(200): v.step(np.random.randint(8, size=n))
        print(f"{n} envs: {200 * n / (time.time() - t):.0f} steps/s"); v.close()

    mask, rates = volatile_mask()
    os.makedirs("data/masks", exist_ok=True)
    np.save("data/masks/tetris_volatile.npy", mask)
    idx = np.flatnonzero(mask)
    json.dump({"rule": "noop change rate >= 0.6", "thresh": 0.6, "episodes": 10, "n_masked": int(mask.sum()),
               "addresses": [hex(0xC000 + i) for i in idx]}, open("data/masks/tetris_volatile.json", "w"), indent=1)
    print("masked", mask.sum(), [hex(0xC000 + i) for i in idx][:40])
