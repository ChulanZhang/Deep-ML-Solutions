import numpy as np

def mig_resource_allocation(total_sm: int, total_mem: float, num_instances: int) -> tuple:
    # Simple division scheme for MIG
    sm_per_instance = total_sm // num_instances
    mem_per_instance = total_mem / num_instances
    
    return int(sm_per_instance), float(round(mem_per_instance, 4))
