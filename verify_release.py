"""Verify the distributed frozen files using only the Python standard library."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
files=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
for item in files:
    p=ROOT/item['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise SystemExit('File hash mismatch: '+item['path'])
with zipfile.ZipFile(ROOT/'data/observational_inputs.zip') as z:
    if z.testzip() is not None:raise SystemExit('Observational archive CRC failed')
    for item in json.loads((ROOT/'input_sources.json').read_text()):
        if hashlib.sha256(z.read(item['file'])).hexdigest()!=item['sha256']:raise SystemExit('Input hash mismatch: '+item['file'])
assert json.loads((ROOT/'audit/gc25_cpl_validation.json').read_text())['passed']
assert (ROOT/'LICENSE').read_text().startswith('MIT License')
optional=json.loads((ROOT/'plotting_samples_manifest.json').read_text(encoding='utf8'))
present=0
for item in optional:
    p=ROOT/item['path']
    if p.exists():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise SystemExit('Optional sample hash mismatch: '+item['path'])
        present+=1
print('Verified',len(files),'frozen release files and four observational inputs.')
print('Optional plotting arrays present:',str(present)+'/'+str(len(optional)))
