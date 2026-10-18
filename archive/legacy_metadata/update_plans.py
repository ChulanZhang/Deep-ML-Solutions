import os

with open('/Users/peng/Documents/GitHub/Deep-ML-Solutions/missing_list_output.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

batches = []
batch_size = 15
for i in range(0, len(lines), batch_size):
    batches.append(lines[i:i+batch_size])

# Update Implementation Plan
plan_path = '/Users/peng/.gemini/antigravity/brain/8583fc13-1c15-4618-9c1b-9d9391201876/implementation_plan.md'
with open(plan_path, 'a') as f:
    f.write('\n\n## Extended Plan: Batches 13-22 (Remaining Category Discoveries)\n')
    f.write('The following batches encapsulate the 140 remaining problems across Deep Learning, Computer Vision, RL, LLM, and Inference tracking metrics.\n')
    batch_num = 13
    for batch in batches:
        f.write(f'\n### Batch {batch_num}: Advanced Scrape Part {batch_num - 12}\n')
        for item in batch:
            f.write(f"{item}\n")
        batch_num += 1

# Update Task Tracker
task_path = '/Users/peng/.gemini/antigravity/brain/8583fc13-1c15-4618-9c1b-9d9391201876/task.md'
with open(task_path, 'r') as f:
    task_content = f.read()

task_updates = "\n- [x] Scan and discover the remaining 140 problems across deep-ml categories\n"
batch_num = 13
for batch in batches:
    task_updates += f"- [ ] Implement Batch {batch_num} (15 advanced problems)\n"
    batch_num += 1

with open(task_path, 'w') as f:
    f.write(task_content + task_updates)

print(f"Successfully appended {len(batches)} new batches to the plan and task documents.")
