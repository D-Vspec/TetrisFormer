"""Scripted Tetris expert (D-005): one-piece placement search with a linear board heuristic, executed as button
presses through the same TetrisEnv.step (24-frame steps), closed loop on what is on screen."""
import numpy as np
from env import TetrisEnv

EMPTY = 47
DOWN, LEFT, RIGHT, A, NOOP = 1, 2, 3, 4, 7
# Well-known linear weights (Yiyuan Lee, "Tetris AI – The (Near) Perfect Bot"): height, lines, holes, bumpiness
W = np.array([-0.510066, 0.760666, -0.35663, -0.184483])


def screen(env):
    ga = np.array(env.w.game_area()) != EMPTY              # landed + falling (sprites)
    bg = np.array(env.pb.tilemap_background[2:12, 0:18]) != EMPTY  # landed only
    return bg, ga & ~bg


def norm(cells):
    cells = np.asarray(cells); return frozenset(map(tuple, cells - cells.min(0)))


def rotations(shape):
    out, s = [], shape
    for _ in range(4):
        if s not in out: out.append(s)
        s = norm([(c, -r) for r, c in s])  # rotate 90°
    return out


def drop(board, shape, col):
    """Land shape with left edge at col; return new board + lines, or None if it doesn't fit."""
    cells = np.array(sorted(shape)); cells[:, 1] += col
    if cells[:, 1].max() >= 10: return None
    r = 0
    while True:
        nxt = cells + [r + 1, 0]
        if nxt[:, 0].max() >= 18 or board[nxt[:, 0], nxt[:, 1]].any(): break
        r += 1
    land = cells + [r, 0]
    if board[land[:, 0], land[:, 1]].any(): return None
    b = board.copy(); b[land[:, 0], land[:, 1]] = True
    full = b.all(1); lines = int(full.sum())
    b = np.vstack([np.zeros((lines, 10), bool), b[~full]])
    return b, lines


def features(b, lines):
    filled = b.any(0)
    h = np.where(filled, 18 - b.argmax(0), 0)
    holes = sum(int((~b[b[:, c].argmax():, c]).sum()) for c in range(10) if filled[c])
    return np.array([h.sum(), lines, holes, np.abs(np.diff(h)).sum()])


def plan(board, falling):
    best = None
    for shape in rotations(norm(np.argwhere(falling))):
        width = max(c for _, c in shape) + 1
        for col in range(10 - width + 1):
            res = drop(board, shape, col)
            if res is None: continue
            score = W @ features(*res)
            if best is None or score > best[0]: best = (score, shape, col)
    return best and best[1:]


class Expert:
    def __init__(self, env):
        self.env, self.target, self.landed = env, None, -1

    def act(self):
        board, falling = screen(self.env)
        if not falling.any(): return NOOP
        n = int(board.sum())
        if self.target is None or n != self.landed:  # new piece (board changed => previous piece locked)
            self.target, self.landed = plan(board, falling), n
        if self.target is None: return DOWN
        shape, col = self.target
        cells = np.argwhere(falling)
        if norm(cells) != shape: return A
        c = cells[:, 1].min()
        return LEFT if c > col else RIGHT if c < col else DOWN


def rollout(timer_div, record=True):
    env = TetrisEnv(); obs = env.reset(timer_div=timer_div); ex = Expert(env)
    trace, done = [obs], False
    while not done:
        obs, done, info = env.step(ex.act())
        if record: trace.append(obs)
    return np.stack(trace) if record else None, info


if __name__ == "__main__":
    import time
    for td in range(3):
        t = time.time(); tr, info = rollout(td, record=False)
        print(f"timer_div={td} lines={info['lines']} steps={info['t']} over={info['t'] < 2000} ({time.time() - t:.1f}s)")
