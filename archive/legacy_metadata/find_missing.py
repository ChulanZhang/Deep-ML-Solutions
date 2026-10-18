import os
import re

# Read solved files
numpy_dir = '/Users/peng/Documents/GitHub/Deep-ML-Solutions/numpy'
solved_ids = set()
for f in os.listdir(numpy_dir):
    match = re.match(r'^(\d+)-.*\.py$', f)
    if match:
        solved_ids.add(int(match.group(1)))

# Read scraped target problems
missing_problems = []
with open('/Users/peng/Documents/GitHub/Deep-ML-Solutions/raw_problems.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split(':', 1)
        if len(parts) == 2:
            num = int(parts[0].strip())
            title = parts[1].strip()
            if num not in solved_ids:
                missing_problems.append((num, title))

missing_problems.sort(key=lambda x: x[0])

print(f"Total missing problems to solve: {len(missing_problems)}")

out = ""
for num, title in missing_problems:
    out += f"- [NEW] {num}: {title}\n"

with open('/Users/peng/Documents/GitHub/Deep-ML-Solutions/missing_list_output.txt', 'w') as f:
    f.write(out)
    
print("Saved missing ones to missing_list_output.txt")
