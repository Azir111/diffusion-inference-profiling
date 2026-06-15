"""
Experiment 3 — Step distillation (the core diffusion optimization)
==================================================================
Exp-1 showed latency ∝ steps. The dominant lever for diffusion serving is
therefore reducing the step count. This compares standard SDXL sampling against
a consistency-distilled variant via **LCM-LoRA** — a ~135 MB adapter loaded on
top of the already-cached SDXL base. No multi-GB model download, and BOTH paths
run at 1024×1024, so this is a clean same-resolution comparison.

  * SDXL base         : 30 steps, gs=7.0  (standard sampling)
  * SDXL + LCM-LoRA   : 4 / 8 steps, gs=1.0  (distilled; LCM uses no CFG)

Deps: pip install peft   (required for LoRA loading)
China mirror: export HF_ENDPOINT=https://hf-mirror.com

Outputs: results/exp3_distillation.csv
"""
import time
import csv
import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, LCMScheduler

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

PROMPT = "product photo of a white leather sneaker on marble, studio lighting, sharp focus, high detail"
NEGATIVE = "blurry, low quality, deformed, watermark, text"
SEED = 42
REPEAT = 3
rows = []

# Reuse the already-cached SDXL base (loads from local cache, no download)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")


def median_latency(steps, gs, negative):
    def once():
        gen = torch.Generator("cuda").manual_seed(SEED)
        torch.cuda.synchronize(); t0 = time.time()
        pipe(prompt=PROMPT, negative_prompt=negative,
             num_inference_steps=steps, guidance_scale=gs,
             height=1024, width=1024, generator=gen).images[0]
        torch.cuda.synchronize(); return time.time() - t0
    ts = sorted(once() for _ in range(REPEAT))
    return ts[len(ts) // 2]


# ---------- baseline: standard SDXL ----------
print("Warmup...")
median_latency(20, 7.0, NEGATIVE)
lat = median_latency(30, 7.0, NEGATIVE)
rows.append(("SDXL-base (standard)", 30, round(lat, 3)))
print(f"SDXL-base (standard)    30 steps 1024 gs=7.0 -> {lat:.3f}s")

# ---------- distilled: LCM-LoRA ----------
print("Loading LCM-LoRA (~135 MB)...")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# Behind the GFW, the HF mirror can fail on Xet-backed downloads. Workaround:
# pre-download the file and load from a local dir instead of the repo id, e.g.
#   wget https://hf-mirror.com/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors
#   pipe.load_lora_weights(os.path.expanduser("~/sdxl/models/lcm-lora-sdxl"))
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
# LCM uses no classifier-free guidance -> gs=1.0, and no negative prompt
for steps in (4, 8):
    lat = median_latency(steps, 1.0, None)
    rows.append(("SDXL + LCM-LoRA", steps, round(lat, 3)))
    print(f"SDXL + LCM-LoRA         {steps} steps 1024 gs=1.0 -> {lat:.3f}s")

with open(os.path.join(OUT, "exp3_distillation.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "steps", "latency_s"])
    w.writerows(rows)

base_lat = next(r[2] for r in rows if r[0].startswith("SDXL-base"))
lcm4 = next(r[2] for r in rows if r[0] == "SDXL + LCM-LoRA" and r[1] == 4)
print(f"\nSpeedup (base 30-step -> LCM 4-step, same 1024 res): {base_lat/lcm4:.1f}×")
print("Saved results/exp3_distillation.csv")

# ---- Alternative if you specifically want SDXL-Turbo (512, separate ~7GB download) ----
# The mirror sometimes fails on Turbo's Xet-backed LFS files. Robust pre-download:
#   HF_HUB_DISABLE_XET=1 huggingface-cli download stabilityai/sdxl-turbo \
#       --local-dir ~/sdxl/models/sdxl-turbo
# then load from the local path:
#   from diffusers import AutoPipelineForText2Image
#   turbo = AutoPipelineForText2Image.from_pretrained(
#       os.path.expanduser("~/sdxl/models/sdxl-turbo"), torch_dtype=torch.float16).to("cuda")
#   turbo(prompt=PROMPT, num_inference_steps=1, guidance_scale=0.0).images[0]  # 512-native
