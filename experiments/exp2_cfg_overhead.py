"""
Experiment 2 — Classifier-Free Guidance (CFG) overhead
======================================================
With guidance_scale > 1, each denoising step runs the UNet TWICE (conditional +
unconditional, batched as 2) — CFG roughly doubles per-step compute. With
guidance_scale = 1 the unconditional pass is skipped (single UNet pass).

Method: run the Exp-1 step sweep twice — once with CFG off (gs=1.0), once with
CFG on (gs=7.0) — and fit a line to each. Prediction:
  * intercept (fixed text-enc + VAE overhead) ≈ the same for both
  * slope (per-step UNet cost) for CFG-on ≈ 2× the CFG-off slope

NOTE: gs=1.0 produces lower-quality images (no guidance). This experiment is a
load characterization, not a quality recommendation.

Outputs: results/exp2_cfg_result.csv, results/exp2_cfg_overhead.png
"""
import time
import csv
import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")

PROMPT = "product photo of a white leather sneaker on marble, studio lighting, sharp focus, high detail"
NEGATIVE = "blurry, low quality, deformed, watermark, text"
STEP_LIST = [10, 20, 30, 40, 50]
REPEAT = 3
SEED = 42


def run_once(steps, gs):
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.time()
    _ = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
             num_inference_steps=steps, guidance_scale=gs,
             height=1024, width=1024, generator=gen).images[0]
    torch.cuda.synchronize()
    return time.time() - t0


def fit(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxy = sum(x*y for x, y in zip(xs, ys)); sxx = sum(x*x for x in xs)
    b = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    a = (sy - b*sx) / n
    return a, b


print("Warmup (discarded)...")
run_once(20, 7.0)

rows = []
fits = {}
for label, gs in [("cfg_off", 1.0), ("cfg_on", 7.0)]:
    xs, ys = [], []
    for steps in STEP_LIST:
        times = sorted(run_once(steps, gs) for _ in range(REPEAT))
        median_t = times[len(times) // 2]
        rows.append((label, gs, steps, median_t, median_t / steps))
        xs.append(steps); ys.append(median_t)
        print(f"{label} (gs={gs}) | steps={steps:>2} | total {median_t:5.2f}s | per-step {median_t/steps:.3f}s")
    fits[label] = fit(xs, ys)

with open(os.path.join(OUT, "exp2_cfg_result.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config", "guidance_scale", "steps", "total_latency_s", "per_step_s"])
    w.writerows(rows)

a_off, b_off = fits["cfg_off"]; a_on, b_on = fits["cfg_on"]
print(f"\nCFG off: latency ≈ {a_off:.3f} + {b_off:.3f}·steps")
print(f"CFG on : latency ≈ {a_on:.3f} + {b_on:.3f}·steps")
print(f"Per-step slope ratio (on/off) = {b_on/b_off:.2f}×   (expect ≈ 2)")
print(f"Intercept (fixed overhead) off={a_off:.3f}s on={a_on:.3f}s  (expect ≈ equal)")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    off = [r for r in rows if r[0] == "cfg_off"]; on = [r for r in rows if r[0] == "cfg_on"]
    plt.figure(figsize=(7, 5))
    plt.plot([r[2] for r in off], [r[3] for r in off], "o-", color="#16a34a",
             label=f"CFG off (gs=1): {b_off:.3f}s/step")
    plt.plot([r[2] for r in on], [r[3] for r in on], "o-", color="#dc2626",
             label=f"CFG on (gs=7): {b_on:.3f}s/step")
    plt.xlabel("denoising steps"); plt.ylabel("total latency (s)")
    plt.title(f"CFG doubles per-step compute ({b_on/b_off:.2f}× slope)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "exp2_cfg_overhead.png"), dpi=130)
    print("Saved results/exp2_cfg_overhead.png")
except ImportError:
    print("matplotlib not installed — skipped plot")
