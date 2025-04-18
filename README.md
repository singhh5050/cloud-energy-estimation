# cloud_inference_energy_estimate_w_model_attributes

---

## 📘 Overview

This README outlines the design, rationale, and usage of the `cloud_inference_energy_estimate_w_model_attributes()` function. Key improvements include:

1. **Model & GPU Profiles Built-In**: No external profile dictionaries needed—just provide `model_name` and `gpu_name`.
2. **Reasoning Token Support**: Accounts for hidden chain-of-thought tokens via a `reasoning_factor`.
3. **Attention Mode Toggle**: Switch between quadratic and linear (Flash) attention costs with `attention_mode`.
4. **Dynamic Utilization Curve**: Models per-token decode cost growth as context length increases.

---

## 🎯 Model Estimation Methodology

We reverse-engineered architecture parameters for OpenAI's GPT-4o, o1, and o3-mini using:

- **Open-Source Analogs**: Mixtral (MoE 8×22B), LLaMA-2-70B, and Mistral-7B.
- **Public Benchmarks**: Throughput, context length, and API behavior.

```python
MODEL_PROFILES = {
    "gpt-4o": {
        # Scaled from Mixtral-8x22B (per Epoch AI methodology)
        "num_active_params": 100e9,
        "num_layers": 77,
        "num_attn_heads": 57,
        "attn_head_dim": 150.1,
        "hidden_dim": 8448.42,
        "flops_per_tkn_factor": 2,
        "flops_per_tkn_factor_attn": 4,
        "reasoning_factor": 1.0,
    },
    "o1": {
        # LLaMA-2-70B dense proxy
        "num_active_params": 70e9,
        "num_layers": 80,
        "num_attn_heads": 64,
        "attn_head_dim": 128,
        "hidden_dim": 8192,
        "flops_per_tkn_factor": 2,
        "flops_per_tkn_factor_attn": 4,
        "reasoning_factor": 3.0,
    },
    "o3-mini": {
        # Mistral-7B dense proxy
        "num_active_params": 7.3e9,
        "num_layers": 32,
        "num_attn_heads": 32,
        "attn_head_dim": 128,
        "hidden_dim": 4096,
        "flops_per_tkn_factor": 2,
        "flops_per_tkn_factor_attn": 4,
        "reasoning_factor": 2.0,
    }
}
```

---

## 💽 GPU Profile Sourcing

Derived from official specs and energy measurement studies:

- **Datasheets**: FLOPs, TDP (NVIDIA A100, H100, GB200)
- **Empirical Sources**: MLPerf, ASPLOS'24, Microsoft Research

```python
GPU_PROFILES = {
    "A100": {
        "peak_flops": 624e12,
        "gpu_prefill_util": 0.5,
        "gpu_decoding_util": 0.1,
        "power_rating": 400,
        "power_prefill_util": 1.0,
        "power_decoding_util": 0.75,
    },
    "H100": {
        "peak_flops": 989e12,
        "gpu_prefill_util": 0.5,
        "gpu_decoding_util": 0.1,
        "power_rating": 700,
        "power_prefill_util": 1.0,
        "power_decoding_util": 0.75,
    },
    "GB200": {
        "peak_flops": 10000e12,
        "gpu_prefill_util": 0.5,
        "gpu_decoding_util": 0.1,
        "power_rating": 1200,
        "power_prefill_util": 1.0,
        "power_decoding_util": 0.75,
    },
}
```

---

## 🔢 Math Behind Nuanced Improvements

### 1. Dynamic Utilization Curve

We model each decode token $i$ as attending to an expanding context of size $n_{\mathrm{in}} + i$.

```math
\mathrm{AttnFLOPs}_{\mathrm{decode}} =
4H\,d_{\mathrm{head}}\,L \sum_{i=0}^{T_r - 1}(n_{\mathrm{in}} + i)
= 4H\,d_{\mathrm{head}}\,L \left[n_{\mathrm{in}}T_r + \frac{T_r(T_r - 1)}{2}\right]
```

Where:
- $T_r$: total decode tokens
- $n_{\mathrm{in}}$: input tokens
- $H$: attention heads
- $d_{\mathrm{head}}$: head dimension
- $L$: transformer layers

---

### 2. Flash (Linear) vs Quadratic Attention Toggle

#### Quadratic Attention

```math
\mathrm{AttnFLOPs}_{\mathrm{prefill}} = n_{\mathrm{in}}^2 \cdot 4H\,d_{\mathrm{head}}\,L
```

#### Linear Attention (Flash, etc.)

```math
\mathrm{AttnFLOPs}_{\mathrm{prefill}} = n_{\mathrm{in}} \cdot 4H\,d_{\mathrm{head}}\,L
```

```math
\mathrm{AttnFLOPs}_{\mathrm{decode}} = T_r \cdot n_{\mathrm{ctx,avg}} \cdot 4H\,d_{\mathrm{head}}\,L
```

Where $n_{\mathrm{ctx,avg}}$ is the average context length per decode token.

---

## ▶️ How to Use

```python
from energy_tracking import cloud_inference_energy_estimate_w_model_attributes

results = cloud_inference_energy_estimate_w_model_attributes(
    input_tokens=1024,
    output_tokens=256,
    model_name="o1",           # "gpt-4o", "o1", "o3-mini"
    gpu_name="H100",           # "A100", "H100", "GB200"
    attention_mode="quadratic" # or "linear"
)

print(results)
```
