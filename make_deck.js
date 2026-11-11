const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/home/dv/.claude/skills/synced/69b3126b-a4cd-485f-b6f6-56f5446a817b_1d5dfc4d-1e64-4602-9c5d-6153c809e71f/pptx/scripts/apply_theme.js");
const ROOT = "/home/dv/Documents/Programming/Projects/MarioFormer";
const OUT = process.argv[2];

// Game Boy DMG palette: darkest green dominates, lime accent
const HEX = { dark: "0F380F", mid: "306230", lime: "8BAC0F", pale: "E8F0D0", ink: "1B1B1B", grey: "5F6B5F", white: "FFFFFF", amber: "C9822B" };
const THEME = {
  name: "TetrisFormer", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.dark, lt2: HEX.pale, accent1: HEX.mid, accent2: HEX.lime,
    accent3: HEX.amber, accent4: "9BBC0F", accent5: HEX.grey, accent6: "4E7A4E", hlink: HEX.mid, folHlink: HEX.grey },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.title = "TetrisFormer"; pres.author = "D-VSpec";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.45, w: 9, h: 0.9, fontFace: "Cambria", fontSize: 38, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.75, fontFace: "Cambria", fontSize: 28, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5 },
});

const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: "Calibri", color: C.text1, fontSize: 14, margin: 0, ...o });
const card = (s, x, y, w, h, name, fill = C.background2) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
const stat = (s, x, y, w, big, label, dark = false) => {
  T(s, big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: 40, bold: true, color: dark ? C.accent2 : C.accent1 });
  T(s, label, { x, y: y + 0.75, w, h: 0.55, fontSize: 12, color: dark ? C.background2 : C.accent5, valign: "top" });
};
const chartStyle = (title) => ({
  showTitle: true, title, titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.dark,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.ink,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  catAxisLabelColor: HEX.ink, valAxisLabelColor: HEX.grey, valGridLine: { color: "E2E6DD", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: false,
});

// 1 — Project
pres.addSection({ title: "Project" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Project" });
s.addText("TetrisFormer", { placeholder: "title" });
T(s, "Learning rewards from machine state", { x: 0.5, y: 1.25, w: 9, h: 0.45, fontFace: "Cambria", fontSize: 20, italic: true, color: C.accent2 });
T(s, [
  { text: "A small transformer learns what competent Game Boy Tetris looks like from the console's RAM alone.", options: { bullet: true, breakLine: true } },
  { text: "Its likelihood of the next memory state becomes the reward for an RL agent that also acts on RAM.", options: { bullet: true, breakLine: true } },
  { text: "No hand-written reward, no pixels, no action labels.", options: { bullet: true } },
], { x: 0.5, y: 1.9, w: 9, h: 1.3, fontSize: 15, color: C.background1, paraSpaceAfter: 6 });
s.addImage({ path: `${ROOT}/runs/fig_agents_frame_3m.png`, x: 1.85, y: 3.3, w: 6.3, h: 2.09, objectName: "Agents frame",
  altText: "Scripted expert, learned-reward agent and curiosity agent playing the same piece sequence" });
s.addNotes("Two-day proof of concept on Game Boy Tetris; the full plan targets Super Mario Land and Pokémon. The image shows the scripted expert, our agent and a curiosity agent on the same piece sequence.");

// 2 — Transformer architecture
pres.addSection({ title: "Reward transformer" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reward transformer" });
s.addText("A transformer that reads RAM byte by byte", { placeholder: "title" });
const flow = [
  ["Game Boy RAM", "8 KB work RAM every 24 frames; 289 bytes that vary are kept"],
  ["Byte tokens", "value + address + time embeddings, d = 128"],
  ["Causal transformer", "4 layers, 4 heads; step t sees all bytes of steps ≤ t"],
  ["Predict next state", "256-way softmax for every byte of step t+1"],
];
flow.forEach(([h, d], i) => {
  const x = 0.5 + i * 2.3;
  card(s, x, 1.35, 2.05, 1.9, `Stage ${i + 1}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: 1.5, w: 0.45, h: 0.45, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `Stage ${i + 1} badge` });
  T(s, String(i + 1), { x: x + 0.15, y: 1.5, w: 0.45, h: 0.45, align: "center", valign: "middle", bold: true, color: C.background1, fontSize: 16 });
  T(s, h, { x: x + 0.15, y: 2.02, w: 1.8, h: 0.55, bold: true, fontSize: 15, color: C.text2, valign: "top" });
  T(s, d, { x: x + 0.15, y: 2.6, w: 1.8, h: 0.6, fontSize: 12, color: C.text1, valign: "top" });
  if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 2.07, y: 2.3, w: 0.21, h: 0, line: { color: C.accent1, width: 2, endArrowType: "triangle" }, objectName: `Arrow ${i + 1}` });
});
stat(s, 0.5, 3.6, 2.8, "0.9M", "parameters");
stat(s, 3.6, 3.6, 2.8, "2,312", "tokens per window (289 bytes × 8 steps)");
stat(s, 6.7, 3.6, 2.8, "Σ log p", "of the next state = the reward signal");
s.addNotes("Each RAM byte is a token. The block-causal mask lets every byte of step t attend to the whole history up to t, and predict the same address at t+1. The summed log-probability of the real next state is the reward.");

// 3 — Training
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reward transformer" });
s.addText("Trained only on RAM traces of expert play", { placeholder: "title" });
const steps = [
  ["Expert plays", "A scripted placement heuristic plays 100 games, 53 lines per game"],
  ["Keep only RAM", "183k steps of memory states; actions and pixels are thrown away"],
  ["Mask the noise", "Bytes that churn regardless of input (sound driver, stack) are removed"],
  ["Train", "3,000 AdamW steps, one-cycle LR 1e-3, batch 32; about 15 min on a laptop RTX 5060"],
];
steps.forEach(([h, d], i) => {
  const y = 1.3 + i * 0.95;
  s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.6, h: 0.6, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: `Step ${i + 1} badge` });
  T(s, String(i + 1), { x: 0.5, y, w: 0.6, h: 0.6, align: "center", valign: "middle", bold: true, color: C.accent2, fontSize: 18 });
  T(s, h, { x: 1.3, y: y - 0.05, w: 4.2, h: 0.35, bold: true, fontSize: 16, color: C.text2 });
  T(s, d, { x: 1.3, y: y + 0.3, w: 4.2, h: 0.5, fontSize: 13, valign: "top" });
});
card(s, 6.0, 1.3, 3.5, 3.6, "Split card", C.text2);
T(s, "Held-out evaluation", { x: 6.3, y: 1.5, w: 3.0, h: 0.4, bold: true, fontSize: 16, color: C.accent2 });
stat(s, 6.3, 2.0, 3.0, "90 / 10", "games for train / held-out validation, split by episode", true);
stat(s, 6.3, 3.3, 3.0, "0 labels", "no rewards, actions or pixels ever seen", true);
s.addNotes("The expert is a hand-written placement search with a standard linear board heuristic. Only its RAM is recorded. Validation games are whole held-out episodes.");

// 4 — Transformer results
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reward transformer" });
s.addText("It predicts expert RAM almost perfectly", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "NLL", labels: ["Expert (held-out)", "Random play", "Copy last state", "Per-address unigram"], values: [0.040, 0.417, 0.505, 0.715] }], {
  x: 0.5, y: 1.2, w: 5.6, h: 3.9, barDir: "bar", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 0.8,
  dataLabelFormatCode: "0.000", chartColors: [HEX.mid], barGapWidthPct: 60, objectName: "NLL chart",
  ...chartStyle("Negative log-likelihood, nats per byte (lower is better)"),
});
card(s, 6.5, 1.3, 3.0, 1.75, "Stat 12x");
stat(s, 6.75, 1.45, 2.6, "12×", "better than copying the last state");
card(s, 6.5, 3.3, 3.0, 1.75, "Stat 10x");
stat(s, 6.75, 3.45, 2.6, "~10×", "less likely per byte for random play than for expert play");
s.addNotes("0.040 nats per byte on 10 held-out expert games. The model trained without the volatile-byte mask reaches the same validation loss.");

// 5 — RL loop
pres.addSection({ title: "RL loop" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "RL loop" });
s.addText("RL loop: the frozen transformer is the reward", { placeholder: "title" });
const box = (x, y, w, h, title, sub, name, fill, tc, sc) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: name });
  T(s, title, { x, y: y + 0.12, w, h: 0.4, align: "center", bold: true, fontSize: 15, color: tc });
  T(s, sub, { x: x + 0.1, y: y + 0.5, w: w - 0.2, h: h - 0.6, align: "center", fontSize: 11, color: sc, valign: "top" });
};
const L = { color: HEX.dark, width: 1.75 };
const seg = (x, y, w, h, name, arrow, flipH = false, flipV = false) =>
  s.addShape(pres.shapes.LINE, { x, y, w, h, flipH, flipV, line: { ...L, ...(arrow ? { endArrowType: "triangle" } : {}) }, objectName: name });
const lab = (text, x, y, w, align = "center") => T(s, text, { x, y, w, h: 0.28, fontSize: 11, italic: true, color: C.accent5, align });
box(0.5, 2.2, 2.0, 1.1, "Emulator", "PyBoy, Game Boy Tetris", "Emulator box", C.text2, C.background1, C.background2);
box(3.85, 1.15, 2.3, 1.0, "Policy", "MLP over one-hot RAM", "Policy box", C.accent1, C.background1, C.background2);
box(3.85, 3.75, 2.3, 1.1, "Reward transformer", "frozen, trained on expert RAM", "Reward box", C.accent2, C.text2, C.text2);
box(7.5, 2.2, 2.0, 1.1, "PPO", "+ RND exploration bonus", "PPO box", C.background2, C.text2, C.text1);
// observation: emulator -> policy
seg(1.0, 1.4, 0, 0.8, "Obs up", false, false, true); seg(1.0, 1.4, 2.85, 0, "Obs right", true); lab("RAM state", 1.1, 1.08, 2.6);
// action: policy -> emulator
seg(2.0, 1.9, 1.85, 0, "Action left", false, true); seg(2.0, 1.9, 0, 0.3, "Action down", true); lab("action", 2.1, 1.92, 1.6);
// history: emulator -> transformer
seg(1.5, 3.3, 0, 1.0, "Hist down"); seg(1.5, 4.3, 2.35, 0, "Hist right", true); lab("RAM history (8 steps)", 1.55, 4.0, 2.3);
// reward: transformer -> PPO
seg(6.15, 4.3, 2.35, 0, "Rew right"); seg(8.5, 3.3, 0, 1.0, "Rew up", true, false, true); lab("r = norm(Σ log p) + β·RND", 6.2, 4.36, 2.3);
// update: PPO -> policy
seg(8.5, 1.65, 0, 0.55, "Upd up", false, false, true); seg(6.15, 1.65, 2.35, 0, "Upd left", true, true); lab("policy update", 6.2, 1.33, 2.3);
s.addNotes("The agent sees only RAM and acts with buttons, or with 40 placement actions (rotation × column) in the placement variant. Each step the frozen transformer scores how likely the real next RAM state is given the last 8; that log-likelihood, normalised, plus an RND exploration bonus, is the reward PPO maximises.");

// 6 — Results: ranking + pause exploit
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("The learned reward ranks behaviour sensibly", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "−log p", labels: ["Expert", "Our agent", "Hand-reward agent", "Random"], values: [5, 16, 40, 106] }], {
  x: 0.5, y: 1.2, w: 5.6, h: 3.9, barDir: "bar", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 120,
  chartColors: [HEX.mid], barGapWidthPct: 60, objectName: "Ranking chart",
  ...chartStyle("Median −log p per step, 3M steps (lower = more expert-like)"),
});
card(s, 6.5, 1.3, 3.0, 3.75, "Pause card", C.text2);
T(s, "Exploit resisted", { x: 6.75, y: 1.5, w: 2.6, h: 0.4, bold: true, fontSize: 16, color: C.accent2 });
T(s, [
  { text: "Both hand-written rewards were hacked by pausing the game.", options: { bullet: true, breakLine: true } },
  { text: "The learned reward scores pausing worst of all.", options: { bullet: true } },
], { x: 6.75, y: 2.0, w: 2.6, h: 1.5, fontSize: 13, color: C.background1, paraSpaceAfter: 8, valign: "top" });
stat(s, 6.75, 3.55, 2.6, "94%", "of steps paused when the sign is flipped (curiosity)", true);
s.addNotes("Online, unseen seeds, 3 seeds each for ours and the hand-reward agent. The comparison uses the same reward model that trained our agent, so the ordering of ours over hand is partly circular; the expert and random endpoints are not.");

// 7 — Results: placement scale-up + takeaways
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Placement actions: the ranking still holds", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "−log p", labels: ["Expert", "Hand-reward agent", "Our agent", "Random placements"], values: [11.1, 25.4, 28.0, 30.7] }], {
  x: 0.5, y: 1.2, w: 5.6, h: 3.9, barDir: "bar", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 36,
  dataLabelFormatCode: "0.0", chartColors: [HEX.mid], barGapWidthPct: 60, objectName: "Placement chart",
  ...chartStyle("Mean −log p per button step, per placement (lower is better)"),
});
const takes = [
  ["RAM is easy to model", "0.9M parameters reach 0.04 nats/byte"],
  ["Reward without labels", "orders expert > agents > random"],
  ["Interpretable", "every gap maps to a few RAM bytes"],
  ["Scales out cheaply", "1M placements in ~4 h on a 5090"],
];
takes.forEach(([h, d], i) => {
  const y = 1.3 + i * 0.95;
  card(s, 6.5, y, 3.0, 0.8, `Takeaway ${i + 1}`);
  T(s, h, { x: 6.7, y: y + 0.08, w: 2.7, h: 0.32, bold: true, fontSize: 14, color: C.text2 });
  T(s, d, { x: 6.7, y: y + 0.4, w: 2.7, h: 0.32, fontSize: 12 });
});
s.addNotes("Placement variant: 40 actions (4 rotations × 10 columns) executed by the expert's button logic, 1M placements per agent. The same frozen transformer, scored per placement over 8 games, orders the policies expert > hand-reward agent > our agent > random placements.");

(async () => { await pres.writeFile({ fileName: OUT }); await applyTheme(OUT, THEME); console.log("wrote", OUT); })();
