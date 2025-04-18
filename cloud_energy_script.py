def cloud_inference_energy_estimate_w_model_attributes(
    input_tokens=0,
    output_tokens=500,
    model_name="gpt-4o",
    gpu_name="H100",
    attention_mode="quadratic",  # or "linear"
    inference_wall_time_sec=None,
):
    # === Model definitions ===
    MODEL_PROFILES = {
        "gpt-4o": {
            # GPT-4o approximation based on scaling up Mixtral 8×22B model
            # (source: Epoch AI energy estimate methodology) 
            # https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use#appendix
            # Mixtral 8×22B is an open-source MoE model with known architecture;
            # we scale parameters and architecture to approximate GPT-4o behavior.
            "num_active_params": 100e9,   # active parameters per token (scaled MoE proxy)
            "num_layers": 77.0,           # transformer blocks (scaled estimate)
            "num_attn_heads": 57.0,       # attention heads per layer (scaled proxy)
            "attn_head_dim": 150.1,       # dimension per head (scaled estimate)
            "hidden_dim": 8448.42,        # model embedding dimension
            # FLOPs per token: 2×N_params for feed-forward + 4×N_params for attention
            "flops_per_tkn_factor": 2,
            "flops_per_tkn_factor_attn": 4,
            "reasoning_factor": 1.0,
        },
        "o1": {
            # LLaMA-2-70B dense proxy
            # All 70B parameters active per token (dense inference)
            "num_active_params": 70e9,
            "num_layers": 80,
            "num_attn_heads": 64,
            "attn_head_dim": 128,
            "hidden_dim": 8192,
            "flops_per_tkn_factor": 2,
            "flops_per_tkn_factor_attn": 4,
            # chain-of-thought length multiplier for longer reasoning
            "reasoning_factor": 3.0,
        },
        "o3-mini": {
            # Mistral-7B dense proxy
            # ~7.3B total parameters
            "num_active_params": 7.3e9,
            "num_layers": 32,
            "num_attn_heads": 32,          # from HF config: hidden_size=4096, heads=32
            "attn_head_dim": 4096 // 32,
            "hidden_dim": 4096,
            "flops_per_tkn_factor": 2,
            "flops_per_tkn_factor_attn": 4,
            # moderate chain-of-thought length
            "reasoning_factor": 2.0,
        }
    }

    GPU_PROFILES = {
        """
        GPU utilization higher during prefill stage because of parallel processing of inputs
        during decoding stage, new output tokens are generated sequentially with the auto-regressive function
        (https://arxiv.org/pdf/2410.18038v1)
        power utilization is very high during prefill stage (compute-heavy)
        differs from less compute-intense decoding stage
        (https://www.microsoft.com/en-us/research/uploads/prod/2024/03/GPU_Power_ASPLOS_24.pdf)
        """
        "A100": {
            # Peak theoretical throughput in FP16/BF16 mode with sparsity for A100 SXM4
            # Source: NVIDIA A100 datasheet — 624 TFLOPS (with sparsity)
            "peak_flops": 624e12,
    
            "gpu_prefill_util": 0.5,  # LLM prefill saturation
            "gpu_decoding_util": 0.1,  # Sequential decode = low GPU utilization
    
            # Max TDP for A100 SXM4 variant
            # Source: NVIDIA A100 datasheet — 400W
            "power_rating": 400,
            "power_prefill_util": 1.0,  # Prefill typically hits TDP
            "power_decoding_util": 0.75,  # Decode is partially memory-bound
        },
        "H100": {
            # Peak theoretical throughput in TF32 mode with sparsity for H100 SXM
            # Source: NVIDIA H100 spec — 989 TFLOPS (with sparsity)
            "peak_flops": 989e12,
    
            "gpu_prefill_util": 0.5,
            "gpu_decoding_util": 0.1,
    
            # Configurable TDP up to 700W for H100 SXM
            # Source: NVIDIA H100 spec sheet
            "power_rating": 700,
            "power_prefill_util": 1.0,
            "power_decoding_util": 0.75,
        },
        "GB200": {
            # Peak throughput in FP8 mode per GPU (with sparsity)
            # Source: NVIDIA DGX GB200 system specs — 720 PFLOPS across 72 GPUs = 10 PFLOPS per GPU
            "peak_flops": 10000e12,
    
            "gpu_prefill_util": 0.5,
            "gpu_decoding_util": 0.1,
    
            # Estimated per-GPU power draw in DGX GB200 system (1200W per Blackwell GPU)
            # Source: NVIDIA keynote, public blog estimates (Datacrunch.io, etc.)
            "power_rating": 1200,
            "power_prefill_util": 1.0,
            "power_decoding_util": 0.75,
        },
    }

    model_attr = MODEL_PROFILES[model_name]
    gpu_attr = GPU_PROFILES[gpu_name]

    reasoning_factor = model_attr.get("reasoning_factor", 1.0)
    total_decode_tokens = output_tokens * reasoning_factor

    joules_per_flop = gpu_attr["power_rating"] / gpu_attr["peak_flops"]

    # === Prefill FLOPs ===
    dense_prefill_flops = (
        input_tokens
        * model_attr["flops_per_tkn_factor"]
        * model_attr["num_active_params"]
    )

    if attention_mode == "quadratic":
        attn_prefill_flops = (
            (input_tokens ** 2)
            * model_attr["flops_per_tkn_factor_attn"]
            * model_attr["num_attn_heads"]
            * model_attr["attn_head_dim"]
            * model_attr["num_layers"]
        )
    else:
        attn_prefill_flops = (
            input_tokens
            * model_attr["flops_per_tkn_factor_attn"]
            * model_attr["num_attn_heads"]
            * model_attr["attn_head_dim"]
            * model_attr["num_layers"]
        )

    prefill_flops = dense_prefill_flops + attn_prefill_flops

    # === Decode FLOPs ===
    if attention_mode == "quadratic":
        linear_term = input_tokens * total_decode_tokens
        triangular_term = (total_decode_tokens * (total_decode_tokens - 1)) / 2
        decode_attn_flops = (
            (linear_term + triangular_term)
            * model_attr["flops_per_tkn_factor_attn"]
            * model_attr["num_attn_heads"]
            * model_attr["attn_head_dim"]
            * model_attr["num_layers"]
        )
    else:
        avg_context = input_tokens + (total_decode_tokens - 1) / 2
        decode_attn_flops = (
            total_decode_tokens
            * avg_context
            * model_attr["flops_per_tkn_factor_attn"]
            * model_attr["num_attn_heads"]
            * model_attr["attn_head_dim"]
            * model_attr["num_layers"]
        )

    decode_dense_flops = (
        total_decode_tokens
        * model_attr["flops_per_tkn_factor"]
        * model_attr["num_active_params"]
    )

    decode_flops = decode_dense_flops + decode_attn_flops

    # === Energy ===
    prefill_energy_joules = (
        prefill_flops
        * (gpu_attr["power_prefill_util"] * joules_per_flop)
        / gpu_attr["gpu_prefill_util"]
    )

    decode_energy_joules = (
        decode_flops
        * (gpu_attr["power_decoding_util"] * joules_per_flop)
        / gpu_attr["gpu_decoding_util"]
    )

    total_flops = prefill_flops + decode_flops

    if inference_wall_time_sec is not None:
        empirical_util = total_flops / (
            inference_wall_time_sec * gpu_attr["peak_flops"]
        )
        empirical_energy_joules = inference_wall_time_sec * gpu_attr["power_rating"]
        return {
            "inference_wall_time_sec": inference_wall_time_sec,
            "empirical_utilization": empirical_util,
            "total_energy_joules_empirical": empirical_energy_joules,
            "total_energy_wh_empirical": empirical_energy_joules / 3600,
            "prefill_energy_joules": prefill_energy_joules,
            "decode_energy_joules": decode_energy_joules,
        }

    return {
        "model": model_name,
        "gpu": gpu_name,
        "reasoning_factor": reasoning_factor,
        "attention_mode": attention_mode,
        "prefill_energy_joules": prefill_energy_joules,
        "decode_energy_joules": decode_energy_joules,
        "total_energy_joules": prefill_energy_joules + decode_energy_joules,
        "total_energy_wh": (prefill_energy_joules + decode_energy_joules) / 3600,
        "total_flops": total_flops,
    }
