import os
import re

pytorch_dir = "pytorch"

for filename in os.listdir(pytorch_dir):
    if not filename.endswith(".py"):
        continue
        
    path = os.path.join(pytorch_dir, filename)
    with open(path, "r") as f:
        content = f.read()
        
    # 替换类型提示
    content = content.replace("list[list[list[list[float]]]]", "torch.Tensor")
    content = content.replace("list[list[list[float]]]", "torch.Tensor")
    content = content.replace("list[list[float]]", "torch.Tensor")
    content = content.replace("list[float]", "torch.Tensor")
    content = content.replace("list[int]", "torch.Tensor")
    
    # 移除内部的张量转换，比如 t = torch.tensor(matrix, dtype=torch.float32)
    # 这部分有点复杂，最好用正则匹配类似 `xxx = torch.tensor(yyy, ...)` 并替换
    # 或者直接把参数名当成原本要操作的对象。
    
    # 我们可以通过正则删掉不必要的 .tolist()
    content = content.replace(".tolist()", "")
    
    with open(path, "w") as f:
        f.write(content)
        
print("Signatures partially refactored!")
