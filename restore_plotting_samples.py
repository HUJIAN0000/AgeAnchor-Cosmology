"""Restore the optional plotting arrays after verifying the entire archive."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path);args=parser.parse_args()
records=json.loads((ROOT/'plotting_samples_manifest.json').read_text(encoding='utf8'))
with zipfile.ZipFile(args.archive) as z:
    if set(z.namelist())!={r['path'] for r in records}:raise SystemExit('Plotting archive file list mismatch')
    for r in records:
        target=ROOT/r['path']
        if not target.resolve().is_relative_to(ROOT.resolve()):raise SystemExit('Unsafe sample path')
        data=z.read(r['path'])
        if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise SystemExit('Plotting sample hash mismatch: '+r['path'])
    for r in records:
        target=ROOT/r['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(r['path']))
print('Restored',len(records),'verified posterior plotting arrays.')
