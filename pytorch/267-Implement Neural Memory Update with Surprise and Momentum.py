import torch

def neural_memory_update(memory: torch.Tensor, new_info: torch.Tensor, surprise_score: float, momentum: float) -> torch.Tensor:
    memory = torch.tensor(memory, dtype=float)
    new_info = torch.tensor(new_info, dtype=float)
    
    updated_memory = momentum * memory + surprise_score * new_info
    return torch.round(updated_memory, 4)
