"""Side-by-side GIF: scripted expert | ours (likelihood reward) | curiosity. Same dynamics as TetrisEnv.step
(one button press, then 24 frames), but rendered and captured every 8 frames."""
import numpy as np, torch
from PIL import Image, ImageDraw
from env import ACTIONS, TetrisEnv
from expert import Expert
from ppo import DEV, Policy, policy_addresses

STEPS, CAP = 200, 8  # agent steps, capture every CAP frames


def frames(kind, timer_div=42):
    env = TetrisEnv(seed=0); obs = env.reset(timer_div=timer_div)
    if kind == "expert":
        ex = Expert(env); act = lambda o: ex.act()
    else:
        pol = Policy(policy_addresses()).to(DEV); pol.load_state_dict(torch.load(f"runs/{kind}/policy.pt")); pol.eval()
        torch.manual_seed(0)
        act = lambda o: pol(torch.as_tensor(o[None], device=DEV))[0].sample().item()
    out, done = [], False
    for _ in range(STEPS):
        if not done:
            with torch.no_grad(): a = act(obs)
            if ACTIONS[a]: env.pb.button(ACTIONS[a])
            for _ in range(24 // CAP):
                env.pb.tick(CAP, True); out.append(env.pb.screen.image.convert("RGB"))
            obs = env.ram(); done = env.w.game_over()
        else:
            out += [out[-1]] * (24 // CAP)  # hold the game-over frame
    return out


panels = [("expert", "Scripted expert"), ("pilot_a", "Ours (likelihood)"), ("curiosity", "Curiosity")]
cols = [frames(k) for k, _ in panels]
W, H, PAD, TOP = 160, 144, 8, 18
gif = []
for i in range(len(cols[0])):
    canvas = Image.new("RGB", (3 * W + 4 * PAD, H + TOP + PAD), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    for j, (col, (_, label)) in enumerate(zip(cols, panels)):
        x = PAD + j * (W + PAD); canvas.paste(col[i], (x, TOP)); d.text((x, 3), label, fill=(230, 230, 230))
    gif.append(canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST))
gif[0].save("runs/tetris_agents.gif", save_all=True, append_images=gif[1:], duration=CAP * 1000 // 60, loop=0, optimize=True)
print(len(gif), "frames")
