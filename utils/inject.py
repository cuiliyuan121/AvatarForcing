from peft import LoraConfig, inject_adapter_in_model

def slice_audio_by_windows(audio, s, e, block=4):
    # audio: (B, F, C), windows: [s:e)
    a_s = 0 if s == 0 else 1 + block * (s - 1)
    a_e = 1 if e <= 1 else 1 + block * (e - 1)
    return audio[:, a_s:a_e, :]

SLICE_RULES = {
    "audio_emb": lambda t, s, e: slice_audio_by_windows(t, s, e, block=4),  # 1,4,4,4...
    "y":         lambda t, s, e: t[:, :, s:e, ...],                         # 1,1,1,1...
}

def slice_conditional_dict(cond: dict, start_frame: int, end_frame: int, rules=SLICE_RULES):
    out = {}
    for k, v in cond.items():
        if v is None:
            out[k] = None
        elif k in rules:
            out[k] = rules[k](v, start_frame, end_frame)
        else:
            out[k] = v
    return out

def _apply_lora(model, cfg):
    print(f"Use LoRA: rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']}")

    init_mode = cfg.get("init_lora_weights", None)
    if isinstance(init_mode, str) and init_mode.lower() == "kaiming":
        init_mode = True
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        init_lora_weights=init_mode,
        target_modules=cfg["lora_target_modules"].split(","),
    )

    patched_model = inject_adapter_in_model(lora_config, model)
    if patched_model is not None:
        model = patched_model

    return model
