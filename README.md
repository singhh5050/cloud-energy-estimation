# cloud_inference_energy_estimate_w_model_attributes

---

## 📘 Overview

This README outlines the design, rationale, and usage of the `cloud_inference_energy_estimate_w_model_attributes()` function. Key improvements include:

1. **Model & GPU Profiles Built-In**: No external profile dictionaries needed—just provide `model_name` and `gpu_name`.
2. **Reasoning Token Support**: Accounts for hidden chain-of-thought tokens via a `reasoning_factor`.
3. **Attention Mode Toggle**: Switch between quadratic and linear (Flash) attention costs with `attention_mode`.
4. **Dynamic Utilization Curve**: Models per-token decode cost growth as context length increases.

These enhancements make the function both **user-friendly** and **energy-accurate** across different LLMs and hardware.

---

## 🎯 Model Estimation Methodology

We reverse-engineered architecture parameters for OpenAI's GPT-4o, o1, and o3-mini using a **principled approach**:

1. **Public Benchmarks & Behavior**: Observing latency, throughput, cost-per-token, and context-window support in the OpenAI API.
2. **Open-Source Analogs**: Mapping to models like Mixtral (22B MoE), LLaMA-2 (70B dense), and Mistral-7B for proxy sizing.
3. **Scaling Laws**: Applying Kaplan et al. and Chinchilla/Hoffmann guidelines to infer hidden dimensions and layer counts.

```python
MODEL_PROFILES = {
  "gpt-4o": {
    "num_active_params": 44e9,  # 2 experts × 22B each active per token
    "num_layers": 72,
    "num_attn_heads": 96,
    "attn_head_dim": 128,
    "hidden_dim": 12288,
    "reasoning_factor": 1.0
  },
  "o1": {
    "num_active_params": 70e9,  # dense (LLaMA-2-70B proxy)
    "num_layers": 80,
    "reasoning_factor": 3.0
  },
  "o3-mini": {
    "num_active_params": 7e9,   # Mistral-7B proxy
    "num_layers": 32,
    "reasoning_factor": 2.0
  }
}
```

## 💽 GPU Profile Sourcing

Hardware characteristics for NVIDIA A100 (SXM4), H100 (SXM5), and the GB200 Superchip (Grace+Blackwell) were extracted from:

- **NVIDIA Data Center Datasheets** (official FLOPs, TDP, memory bandwidth).
- **MLPerf & TechBench Reports** (practical FLOPs/Watt efficiency).
- **Academic Presentations** on GPU power draw curves (prefill vs decode utilization).

```yaml
GPU_PROFILES:
  A100:
    peak_flops: 312e12
    power_rating: 400
  H100:
    peak_flops: 989e12
    power_rating: 700
  GB200:
    peak_flops: 2000e12
    power_rating: 1200
```

## 🔢 Math Behind Nuanced Improvements

### 1. Dynamic Utilization Curve

Rather than using a single average context, we model each decode token *i* as attending to an expanding context of size `n_in + i`. This lets us account for the increasing attention cost across the autoregressive sequence.

**Quadratic attention scaling (used in standard attention):**

```
AttnFLOPs_decode = 4 * H * d_head * L * sum_{i=0}^{T_r-1} (n_in + i)
                 = 4 * H * d_head * L * [n_in * T_r + T_r(T_r-1)/2]
```

Where:
- T_r: total decode tokens, equal to visible output tokens × reasoning factor
- n_in: number of input tokens (prompt length)
- H: number of attention heads
- d_head: head dimension
- L: number of layers

---

### 2. Flash (Linear) vs Quadratic Attention Toggle

You can toggle between two attention scaling modes:

#### 🔲 Quadratic Attention (Standard Transformers):
FLOPs grow quadratically with context length. Attention cost for prefill and decode:

**Prefill:**
```
AttnFLOPs_prefill = n_in^2 * 4 * H * d_head * L
```

**Decode:**
Uses the dynamic context formula above.

#### 🔲 Linear Attention (FlashAttention or Windowed Attention):
FLOPs grow linearly with the context length:

**Prefill:**
```
AttnFLOPs_prefill = n_in * 4 * H * d_head * L
```

**Decode:**
```
AttnFLOPs_decode = T_r * n_ctx_avg * 4 * H * d_head * L
```

Where `n_ctx_avg` is the average context length seen by the model across the decode tokens.

---

Use the `attention_mode` parameter in the function to toggle between these two regimes. This lets you model either memory-efficient attention architectures like FlashAttention or legacy transformer patterns depending on your target system.

---

## ▶️ How to Use

```python
from energy_tracking import cloud_inference_energy_estimate_w_model_attributes

results = cloud_inference_energy_estimate_w_model_attributes(
    input_tokens=1024,
    output_tokens=256,
    model_name="o1",           # Options: "gpt-4o", "o1", "o3-mini"
    gpu_name="H100",           # Options: "A100", "H100", "GB200"
    attention_mode="quadratic" # or "linear"
)

print(results)
```
