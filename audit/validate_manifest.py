"""Verify one canonical entry per official ID and preserve every legacy source."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'audit/manifest.json').read_text())
ids=[r['official_id'] for r in manifest]
assert len(ids)==len(set(ids)), 'Duplicate official ID in canonical manifest'
paths={r['solution_path'] for r in manifest if r['solution_path']}
actual={str(p.relative_to(ROOT)) for p in (ROOT/'pytorch').rglob('*.py')}
assert paths==actual, ('Unmapped or missing reviewed implementation',paths^actual)
assert len(actual)==40
examples=json.loads((ROOT/'audit/public_examples.json').read_text())['problems']
for entry in manifest:
    pid=entry['official_id']
    if str(pid) in examples:
        assert entry['official_title'].strip()==examples[str(pid)]['title'].strip()
    if entry['solution_path']:
        assert int(Path(entry['solution_path']).name.split('_')[0])==pid
    for old in entry['legacy_files']:
        assert (ROOT/old['archive_path']).is_file()
original=json.loads((ROOT/'audit/inventory_original.json').read_text())
for row in original:
    folder,name=row['path'].split('/',1)
    archive='archive/pytorch_legacy/' if folder=='pytorch' else 'archive/interview_core/'
    file=ROOT/(archive+name)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256'], ('Original source changed',row['path'])
nested=json.loads((ROOT/'audit/nested_legacy.json').read_text())['python_files']
assert len(nested)==6
for row in nested:
    assert hashlib.sha256((ROOT/row['archive_path']).read_bytes()).hexdigest()==row['sha256']
print(f'Canonical IDs: {len(ids)}; reviewed solutions: {len(actual)}; all {len(original)+len(nested)} original Python files preserved byte-for-byte.')
