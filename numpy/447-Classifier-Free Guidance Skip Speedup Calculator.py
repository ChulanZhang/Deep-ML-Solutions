def cfg_skip_speedup(total_steps: int, skip_ratio: float, cfg_scale_cost: float = 2.0) -> float:
    # Standard CFG runs 2 passes per step (cond and uncond) -> cost = 2.0 * steps
    # Skipping CFG means it just runs 1 pass later on.
    
    # skip_ratio = fraction of steps that DO NOT use CFG (they run 1 pass)
    # 1 - skip_ratio = fraction of steps using CFG (they run 2 passes)
    
    cfg_steps = int(total_steps * (1.0 - skip_ratio))
    single_steps = total_steps - cfg_steps
    
    baseline_cost = total_steps * cfg_scale_cost
    new_cost = (cfg_steps * cfg_scale_cost) + (single_steps * 1.0)
    
    speedup = baseline_cost / new_cost
    return float(round(speedup, 4))
