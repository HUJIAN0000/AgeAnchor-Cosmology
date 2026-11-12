"""Portable validation, plotting, and independent sampling entry point."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
envdir=Path(sys.executable).parent
os.environ['PATH']=os.pathsep.join([str(envdir),str(envdir/'Library/bin'),str(envdir/'Scripts'),os.environ.get('PATH','')])
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage',choices=['validate','figures','verify','sample'])
parser.add_argument('--suite',choices=['principal','sensitivity','formation','gc25-priors'])
parser.add_argument('--output',type=Path)
parser.add_argument('--samples-archive',type=Path,help='Restore the optional version-matched plotting archive before figures')
parser.add_argument('--workers',type=int,default=2)
parser.add_argument('--threads',type=int,default=4)
args=parser.parse_args()
if args.workers<1 or args.threads<1:parser.error('workers and threads must be positive')
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS=str(args.threads),PYTHONIOENCODING='utf-8')
if args.samples_archive is not None and args.stage!='figures':parser.error('--samples-archive applies to figures')
if args.stage=='verify':
    subprocess.run([sys.executable,str(ROOT/'verify_release.py')],cwd=ROOT,check=True);raise SystemExit(0)
if args.stage=='figures':
    if args.samples_archive is not None:
        subprocess.run([sys.executable,str(ROOT/'restore_plotting_samples.py'),str(args.samples_archive.resolve())],cwd=ROOT,check=True)
    import json
    sources=json.loads((ROOT/'audit/figure_sources.json').read_text(encoding='utf8'))
    needed={ROOT/s['folder']/'posterior.npz' for f in sources['figures'] for s in f['sources'].values()}
    if any(not p.is_file() for p in needed):
        raise SystemExit('Plotting arrays are omitted from this compact source release. Use figures --samples-archive PATH, or run independent sampling and update the source resolver. Prebuilt figures are in reference_outputs/.')
for folder in ['logs','audit','aa_revision/figures']:(ROOT/folder).mkdir(exist_ok=True,parents=True)
subprocess.run([sys.executable,str(ROOT/'fetch_inputs.py')],cwd=ROOT,check=True)
if args.stage=='sample':
    if not args.suite or args.output is None:parser.error('sample requires --suite and a separate --output directory')
    output=args.output if args.output.is_absolute() else ROOT/args.output
    if any(output.resolve()==(ROOT/f).resolve() for f in ['fresh_results/production','release_results/production','reanalysis/runs']):
        parser.error('Use a separate output directory to preserve the supplied evidence')
    driver,config={
      'principal':('fresh_results/run_analysis.py','fresh_results/study_configurations.json'),
      'sensitivity':('reanalysis/run_analysis.py','reanalysis/production_configurations.json'),
      'formation':('reanalysis/run_analysis.py','reanalysis/information_scan_configurations.json'),
      'gc25-priors':('fresh_results/run_analysis.py','release_configurations.json')}[args.suite]
    cmd=[sys.executable,str(ROOT/driver),'--config-file',str(ROOT/config),'--results-dir',str(output),
         '--workers',str(args.workers),'--threads',str(args.threads),'--tag','rerun_'+args.suite]
else:cmd=[sys.executable,str(ROOT/('reanalysis/validate.py' if args.stage=='validate' else 'integration/review_products.py'))]
subprocess.run(cmd,cwd=ROOT,check=True)
