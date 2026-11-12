"""D-013: you play Tetris in the real emulator; beside it, the reward transformer's prediction of the next RAM
state (every 24 frames, as in training), drawn as a board from RAM, plus how surprised it was by what you did.
Keys: arrows, X = A, Z = B, Enter = Start. Usage: python play_rm.py [runs/rm/model.pt]"""
import base64, io, sys, tkinter as tk
import numpy as np, torch, torch.nn.functional as F
from env import FRAME_SKIP, TetrisEnv
from expert import EMPTY
from rm import DEV, RewardModel

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "runs/rm/model.pt", weights_only=False)
rm = RewardModel(ck["addrs"], T=ck["T"]).to(DEV).eval(); rm.load_state_dict(ck["state"])
ADDRS = torch.as_tensor(ck["addrs"], device=DEV)
KEYS = {"Up": "up", "Down": "down", "Left": "left", "Right": "right", "x": "a", "z": "b", "Return": "start"}
CELL, S = 20, 3  # board cell px, screen scale


def board(ram):
    """18x10 board from WRAM: 0 empty, 1 landed (tilemap copy at 0xC802, 32/row), 2 falling (shadow OAM 0xC000)."""
    b = (np.stack([ram[0x802 + r * 32: 0x802 + r * 32 + 10] for r in range(18)]) != EMPTY).astype(int)
    for y, x in ram[0:160].reshape(40, 4)[:, :2]:
        r, c = (int(y) - 16) // 8, (int(x) - 24) // 8
        if 0 <= r < 18 and 0 <= c < 10: b[r, c] = 2
    return b


class App:
    def __init__(self, root):
        self.env, self.root = TetrisEnv(), root
        self.cv = tk.Canvas(root, width=160 * S + 2 * 10 * CELL + 60, height=144 * S + 60, bg="#181818")
        self.cv.pack()
        root.bind("<KeyPress>", lambda e: e.keysym in KEYS and self.env.pb.button_press(KEYS[e.keysym]))
        root.bind("<KeyRelease>", lambda e: e.keysym in KEYS and self.env.pb.button_release(KEYS[e.keysym]))
        self.reset(); self.loop()

    def reset(self):
        ram = self.env.reset()
        self.hist = torch.as_tensor(ram, device=DEV)[ADDRS][None].repeat(ck["T"], 1)  # as rl.py: repeat first state
        self.frame, self.nlls, self.last, self.acc = 0, [], None, None
        self.predict(ram)

    @torch.no_grad()
    def predict(self, ram):
        self.lp = F.log_softmax(rm(self.hist[None])[0, -1].float(), -1)  # [K, 256] for the next state
        nxt = ram.copy(); nxt[ck["addrs"]] = self.lp.argmax(-1).cpu().numpy()  # unmodelled bytes never vary
        self.pred = board(nxt)

    def step(self):  # one model step: score the real state against the prediction, then predict the next
        ram = self.env.ram(); real = board(ram)
        s = torch.as_tensor(ram, device=DEV)[ADDRS]
        self.last = -self.lp.gather(-1, s.long()[:, None]).sum().item(); self.nlls.append(self.last)
        self.acc = (self.pred == real).mean()
        self.hist = torch.cat([self.hist[1:], s[None]]); self.predict(ram)

    def loop(self):
        self.env.pb.tick(1, True); self.frame += 1
        if self.frame % FRAME_SKIP == 0: self.step()
        if self.env.w.game_over(): self.reset()
        if self.frame % 2 == 0: self.draw()
        self.root.after(16, self.loop)  # ponytail: ~60 fps via after(); drifts a little, fine for a demo

    def draw(self):
        cv = self.cv; cv.delete("all")
        buf = io.BytesIO(); self.env.pb.screen.image.convert("RGB").resize((160 * S, 144 * S), 0).save(buf, "PNG")
        self.img = tk.PhotoImage(data=base64.b64encode(buf.getvalue())); cv.create_image(10, 40, image=self.img, anchor="nw")
        for k, (title, b) in enumerate([("RAM now", board(self.env.ram())), ("Transformer: next", self.pred)]):
            x0 = 160 * S + 25 + k * (10 * CELL + 15)
            cv.create_text(x0, 25, text=title, fill="#ddd", anchor="w")
            for r in range(18):
                for c in range(10):
                    col = ("#2a2a2a", "#8bac0f", "#e0c040")[b[r, c]]
                    cv.create_rectangle(x0 + c * CELL, 40 + r * CELL, x0 + (c + 1) * CELL - 1, 40 + (r + 1) * CELL - 1, fill=col, outline="")
        if self.last is not None:
            cv.create_text(10, 15, anchor="w", fill="#ddd", text=(
                f"-log p last step {self.last:7.1f}   mean {np.mean(self.nlls):6.1f}   board cells right {self.acc:.0%}"
                f"      (reference medians: expert ~5, random play ~106)"))


if __name__ == "__main__":
    root = tk.Tk(); root.title("Reward transformer vs. you"); App(root); root.mainloop()
