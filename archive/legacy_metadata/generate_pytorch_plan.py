import os

def create_plan():
    with open("numpy_files_list.txt", "r") as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
        
    batch_size = 15
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    
    plan_content = "# PyTorch Conversion Plan\n\n"
    for i, batch in enumerate(batches):
        plan_content += f"## Batch {i+1}\n"
        for file in batch:
            plan_content += f"- [ ] {file}\n"
        plan_content += "\n"
        
    with open("pytorch_task.md", "w") as f:
        f.write(plan_content)

if __name__ == "__main__":
    create_plan()
