import torch

def detect_attention_sinks(attention_weights: torch.Tensor, sink_size: int = 4, threshold: float = 0.5) -> bool:
    attn = torch.tensor(attention_weights, dtype=float)
    avg_attn_to_keys = torch.mean(attn, dim=(0, 1, 2))
    sink_attn = torch.sum(avg_attn_to_keys[:sink_size])
    return bool(sink_attn >= threshold)
