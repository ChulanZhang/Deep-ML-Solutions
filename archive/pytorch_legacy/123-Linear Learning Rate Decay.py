def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    """
    if num_steps <= 1:
        return [initial_lr]
        
    step_size = (end_lr - initial_lr) / (num_steps - 1)
    return [initial_lr + i * step_size for i in range(num_steps)]
