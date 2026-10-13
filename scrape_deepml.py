import urllib.request
import re
import os
import json

categories = ['Computer+Vision', 'Deep+Learning', 'Inference', 'Reinforcement+Learning', 'LLM']
base_url = 'https://www.deep-ml.com/problems?category={}&page={}'

all_problem_ids = set()

# Headers to prevent 403 Forbidden
headers = {'User-Agent': 'Mozilla/5.0'}

for cat in categories:
    print(f"Scraping {cat}...")
    for page in range(1, 15):
        url = base_url.format(cat, page)
        try:
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req).read().decode('utf-8')
            
            # Extract links like href="/problems/123"
            links = re.findall(r'href="/problems/(\d+)"', html)
            if not links:
                break # No more pages
            
            all_problem_ids.update(int(l) for l in links)
        except Exception as e:
            print(f"Error on {url}: {e}")
            break

print(f"Total unique problem IDs extracted: {len(all_problem_ids)}")

# Find existing solved ones
numpy_dir = '/Users/peng/Documents/GitHub/Deep-ML-Solutions/numpy'
solved_ids = set()
for f in os.listdir(numpy_dir):
    match = re.match(r'^(\d+)-.*\.py$', f)
    if match:
        solved_ids.add(int(match.group(1)))

print(f"Total solved problem IDs: {len(solved_ids)}")

missing_ids = all_problem_ids - solved_ids
print(f"Total missing problem IDs: {len(missing_ids)}")

with open('missing_problems.json', 'w') as f:
    json.dump(list(missing_ids), f)
