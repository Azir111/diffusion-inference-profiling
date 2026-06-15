"""
Experiment 1 — Step count vs. latency / VRAM
============================================
Sweeps the number of denoising steps and measures end-to-end latency and peak
VRAM. Validates that diffusion latency follows  total ≈ fixed_overhead + steps × per_step_cost
(a clean affine function), while peak VRAM stays constant (no KV cache).

Outputs: results/exp1_sweep_result.csv, results/exp1_step_latency.png

Deps: pip install diffusers transformers accelerate safetensors matplotlib
torch: install the CUDA 12.8 build for Blackwell (RTX 50-series).
China mirror: export HF_ENDPOINT=https://hf-mirror.com
"""
import time
import csv
import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

# fp16 VAE numerical-overflow fix (otherwise black images)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")

PROMPT = "product photo of a white leather sneaker on marble, studio lighting, sharp focus, high detail"
NEGATIVE = "blurry, low quality, deformed, watermark, text"
STEP_LIST = [5, 10, 15, 20, 30, 40, 50]
REPEAT = 3
SEED = 42


def run_once(steps):
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()              # GPU is async — sync before timing
    t0 = time.time()
    _ = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
             num_inference_steps=steps, guidance_scale=7.0,
             height=1024, width=1024, generator=gen).images[0]
    torch.cuda.synchronize()              # ...and after, so we time real compute
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1024**3


print("Warmup (discarded)...")
run_once(20)

rows = []
for steps in STEP_LIST:
    samples = [run_once(steps) for _ in range(REPEAT)]
    times = sorted(s[0] for s in samples)
    median_t = times[len(times) // 2]
    peak = samples[-1][1]
    rows.append((steps, median_t, median_t / steps, peak))
    print(f"steps={steps:>2} | total {median_t:5.2f}s | per-step {median_t/steps:.3f}s | peak {peak:.2f}GB")

with open(os.path.join(OUT, "exp1_sweep_result.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["steps", "total_latency_s", "per_step_s", "peak_vram_gb"])
    w.writerows(rows)

# --- linear fit: total = a + b * steps ---
xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
n = len(xs); sx = sum(xs); sy = sum(ys)
sxy = sum(x*y for x, y in zip(xs, ys)); sxx = sum(x*x for x in xs)
b = (n*sxy - sx*sy) / (n*sxx - sx*sx)
a = (sy - b*sx) / n
ss_res = sum((y - (a + b*x))**2 for x, y in zip(xs, ys))
ss_tot = sum((y - sy/n)**2 for y in ys)
r2 = 1 - ss_res/ss_tot
print(f"\nFit: latency ≈ {a:.3f}s + {b:.3f}s/step × steps   (R² = {r2:.5f})")
print(f"  → fixed overhead (text-enc + VAE decode): {a:.3f}s")
print(f"  → pure UNet cost per step:                {b:.3f}s")

# --- plot ---
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(xs, ys, "o", color="#2563eb", label="measured")
    ax1.plot([0, max(xs)], [a, a + b*max(xs)], "--", color="#9ca3af",
             label=f"fit: {a:.2f} + {b:.3f}·steps (R²={r2:.4f})")
    ax1.set_xlabel("denoising steps"); ax1.set_ylabel("total latency (s)")
    ax1.set_title("Latency is affine in step count"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(xs, [r[3] for r in rows], "s-", color="#dc2626")
    ax2.set_ylim(0, max(r[3] for r in rows)*1.3)
    ax2.set_xlabel("denoising steps"); ax2.set_ylabel("peak VRAM (GB)")
    ax2.set_title("VRAM is constant (no KV cache)"); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "exp1_step_latency.png"), dpi=130)
    print("Saved results/exp1_step_latency.png")
except ImportError:
    print("matplotlib not installed — skipped plot")
