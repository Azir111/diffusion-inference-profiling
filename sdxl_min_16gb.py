"""
SDXL 推理最小可跑脚本 — 16GB 显存(RTX 5070Ti),全程驻留 GPU,不用 offload
依赖: pip install torch diffusers transformers accelerate safetensors
首次运行下载约 7GB 权重。国内先执行: export HF_ENDPOINT=https://hf-mirror.com
"""
import time
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# 坑: SDXL 原版 VAE 在 fp16 下会数值溢出出黑图,换官方修复版 VAE(与显存无关,保留)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",          # 只下 fp16 权重,体积减半
    use_safetensors=True,
).to("cuda")                 # 16GB 够用,全部驻留 GPU,比 offload 快

prompt = "product photo of a white leather sneaker on marble, studio lighting, sharp focus, high detail"
negative = "blurry, low quality, deformed, watermark, text"

# 逐步计时:Diffusion 总延迟 ≈ 步数 × 单步 UNet 耗时
step_times, last = [], [time.time()]
def on_step(p, step, timestep, kwargs):
    now = time.time()
    step_times.append(now - last[0])
    last[0] = now
    return kwargs

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=30,
    guidance_scale=7.0,       # CFG>1 → 每步 UNet 实际跑 2 次(cond+uncond)
    height=1024, width=1024,
    callback_on_step_end=on_step,
).images[0]
total = time.time() - t0

image.save("sdxl_out.png")
peak = torch.cuda.max_memory_allocated() / 1024**3
avg = sum(step_times[1:]) / max(len(step_times) - 1, 1)  # 跳过首步预热
print(f"总耗时 {total:.1f}s | 步数 {len(step_times)} | 单步均值 {avg:.2f}s | 峰值显存 {peak:.2f} GB")
print("已保存 sdxl_out.png")
