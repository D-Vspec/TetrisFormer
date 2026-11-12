"""Watch a trained policy play Tetris in a window, real-time, forever. Close the window or Ctrl-C to quit.
Usage: python demo.py [run]   (default runs/ours_3m_s0; a 40-way head, e.g. ours_place_s0_5090, plays placements)"""
import sys, torch
from env import TetrisEnv
from ppo import DEV, Policy, policy_addresses

run = sys.argv[1] if len(sys.argv) > 1 else "ours_3m_s0"
sd = torch.load(f"runs/{run}/policy.pt", map_location=DEV)
n_act = sd["pi.weight"].shape[0]
pol = Policy(policy_addresses(), n_act=n_act).to(DEV); pol.load_state_dict(sd); pol.eval()

env = TetrisEnv(window="SDL2")
env.pb.set_emulation_speed(1)
class Rendered:  # env.step ticks with render=False; force drawing, quit when the window closes
    def __init__(self, pb): self.pb = pb
    def __getattr__(self, k): return getattr(self.pb, k)
    def tick(self, n=1, render=True):
        if not self.pb.tick(n, True): sys.exit()
env.pb = Rendered(env.pb)

if n_act == 40: from expert import place
step = (lambda a: place(env, a)[:2]) if n_act == 40 else (lambda a: env.step(a)[:2])
while True:
    obs, done = env.reset(), False
    while not done:
        with torch.no_grad(): a = pol(torch.as_tensor(obs[None], device=DEV))[0].sample().item()
        obs, done = step(a)
