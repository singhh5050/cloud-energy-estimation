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
            "num_active_params": 44e9,
            "num_layers": 72,
            "num_attn_heads": 96,
            "attn_head_dim": 128,
            "hidden_dim": 12288,
            "flops_per_tkn_factor": 2,
            "flops_per_tkn_factor_attn": 4,
            "reasoning_factor": 1.0,
        },
        "o1": {
            "num_active_params": 70e9,
            "num_layers": 80,
            "num_attn_heads": 96,
            "attn_head_dim": 128,
            "hidden_dim": 12288,
            "flops_per_tkn_factor": 2,
            "flops_per_tkn_factor_attn": 4,
            "reasoning_factor": 3.0,
        },
        "o3-mini": {
            "num_active_params": 7e9,
            "num_layers": 32,
            "num_attn_heads": 32,
            "attn_head_dim": 128,
            "hidden_dim": 4096,
            "flops_per_tkn_factor": 2,
            "flops_per_tkn_factor_attn": 4,
            "reasoning_factor": 2.0,
        }
    }

    GPU_PROFILES = {
        "A100": {
            "peak_flops": 312e12,
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
            "peak_flops": 2000e12,
            "gpu_prefill_util": 0.5,
            "gpu_decoding_util": 0.1,
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
