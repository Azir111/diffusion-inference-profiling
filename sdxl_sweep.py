"""
SDXL 负载特征实验:扫描去噪步数,验证「延迟 ≈ 步数 × 单步耗时」的线性结构
并验证显存与步数无关(恒定)。
输出: sweep_result.csv + sweep_plot.png

依赖: pip install diffusers transformers accelerate safetensors matplotlib
torch 按 5070Ti 走 cu128 单独装。国内先: export HF_ENDPOINT=https://hf-mirror.com
"""
import time
import csv
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# ---- 加载(与上一个脚本一致)----
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")

prompt = "product photo of a white leather sneaker on marble, studio lighting, sharp focus, high detail"
negative = "blurry, low quality, deformed, watermark, text"

STEP_LIST = [5, 10, 15, 20, 30, 40, 50]  # 扫描的步数
REPEAT = 3                                # 每个步数重复次数,取中位数
SEED = 42                                 # 固定种子,排除内容差异

def run_once(steps):
    """单次推理,返回 (耗时秒, 峰值显存GB)。计时已做 GPU 同步。"""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()              # 计时前同步:确保前面的活都干完
    t0 = time.time()
    _ = pipe(
        prompt=prompt, negative_prompt=negative,
        num_inference_steps=steps, guidance_scale=7.0,
        height=1024, width=1024, generator=gen,
    ).images[0]
    torch.cuda.synchronize()              # 计时后同步:确保 GPU 真的算完
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return dt, peak

# ---- 预热:丢弃第一次(含 kernel 编译、显存分配等一次性开销)----
print("预热中(丢弃首次)...")
run_once(20)

# ---- 正式扫描 ----
rows = []
for steps in STEP_LIST:
    samples = [run_once(steps) for _ in range(REPEAT)]
    times = sorted(s[0] for s in samples)
    median_t = times[len(times) // 2]     # 中位数,抗抖动
    peak = samples[-1][1]
    per_step = median_t / steps
    rows.append((steps, median_t, per_step, peak))
    print(f"steps={steps:>2} | 总耗时 {median_t:5.2f}s | 单步 {per_step:.3f}s | 峰值显存 {peak:.2f}GB")

# ---- 存 CSV ----
with open("sweep_result.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["steps", "total_latency_s", "per_step_s", "peak_vram_gb"])
    w.writerows(rows)

# ---- 出图 ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_x = [r[0] for r in rows]
    lat_y   = [r[1] for r in rows]
    vram_y  = [r[3] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图:步数 vs 延迟(应近似过原点的直线)
    ax1.plot(steps_x, lat_y, "o-", color="#2563eb")
    # 拟合一条过原点的参考线,直观看线性
    slope = sum(s*l for s, l in zip(steps_x, lat_y)) / sum(s*s for s in steps_x)
    ax1.plot([0, max(steps_x)], [0, slope*max(steps_x)], "--", color="#9ca3af",
             label=f"线性参考 ~{slope:.3f}s/步")
    ax1.set_xlabel("去噪步数"); ax1.set_ylabel("总延迟 (s)")
    ax1.set_title("延迟随步数近似线性增长"); ax1.legend(); ax1.grid(alpha=0.3)

    # 右图:步数 vs 峰值显存(应基本水平)
    ax2.plot(steps_x, vram_y, "s-", color="#dc2626")
    ax2.set_ylim(0, max(vram_y) * 1.3)
    ax2.set_xlabel("去噪步数"); ax2.set_ylabel("峰值显存 (GB)")
    ax2.set_title("显存与步数无关(恒定)"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sweep_plot.png", dpi=130)
    print("已保存 sweep_result.csv 和 sweep_plot.png")
except ImportError:
    print("已保存 sweep_result.csv(未装 matplotlib,跳过出图)")
