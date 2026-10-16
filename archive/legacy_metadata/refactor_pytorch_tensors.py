import os
import re

pytorch_dir = "pytorch"

for filename in os.listdir(pytorch_dir):
    if not filename.endswith(".py"):
        continue
        
    path = os.path.join(pytorch_dir, filename)
    with open(path, "r") as f:
        content = f.read()
        
    # Replace torch.tensor(var, dtype=...) with var.clone().detach().float()
    # OR simpler: just replace torch.tensor(var, dtype=torch.float32) with var.float() if it's already a tensor.
    # regex for `torch.tensor(var, dtype=torch.float32)`
    content = re.sub(r'torch\.tensor\(([a-zA-Z0-9_]+),\s*dtype=torch\.float32\)', r'\1.clone().detach().float()', content)
    content = re.sub(r'torch\.tensor\(([a-zA-Z0-9_]+),\s*dtype=torch\.bool\)', r'\1.clone().detach().bool()', content)
    
    # regex for `torch.tensor(var)`
    content = re.sub(r'torch\.tensor\(([a-zA-Z0-9_]+)\)', r'\1.clone().detach()', content)
    
    with open(path, "w") as f:
        f.write(content)
        
print("Tensor conversions refactored!")
