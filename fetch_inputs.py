"""Download pinned public observational files and verify their exact SHA-256."""
from pathlib import Path
import hashlib,json,zipfile
import requests
ROOT=Path(__file__).resolve().parent
def main():
    (ROOT/'code').mkdir(exist_ok=True)
    archive=ROOT/'data/observational_inputs.zip'
    for item in json.loads((ROOT/'input_sources.json').read_text()):
        dest=ROOT/'code'/item['file']
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256']:
            print('Verified existing:',item['file']);continue
        if archive.exists():
            with zipfile.ZipFile(archive) as z:content=z.read(item['file'])
            source='Packaged'
        else:
            r=requests.get(item['url'],timeout=90);r.raise_for_status();content=r.content;source='Downloaded'
        if hashlib.sha256(content).hexdigest()!=item['sha256']:raise RuntimeError('Input differs from frozen analysis: '+item['file'])
        dest.write_bytes(content);print(source+' and verified:',item['file'])
if __name__=='__main__':main()
