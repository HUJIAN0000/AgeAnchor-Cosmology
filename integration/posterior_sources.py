# Frozen posterior source resolver and unchanged toolbox interface.
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='4';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import csv,hashlib,importlib.util,json,re,shutil,sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'reanalysis';OUT=ROOT/'aa_revision/figures'
sys.path.insert(0,str(HERE));sys.stdout.reconfigure(encoding='utf8')
import build_products as tables
import make_figures as basic
import refresh_figures as refresh

ALIASES={
 'gc2026_no_bao':'gc26_no_bao','gc2026_bao':'gc26_flat','gc2026_curvature':'gc26_curved',
 'gc2026_cpl':'gc26_cpl','gc2026_bao_seed2':'gc26_flat_seed2','gc2026_cpl_seed2':'gc26_cpl_seed2',
 'gc2026_cpl_restricted_Om':'gc26_cpl_Om025','gc2026_cpl_early_subdominant':'gc26_cpl_subdominant',
 'gc2026_bao_zmax50':'earlyform_flat','gc2026_bao_zmin6':'lateform_flat',
 'stellar_zform_bao':'gc25_flat','stellar_zform_no_bao':'gc25_no_bao',
 'stellar_zform_curvature':'gc25_curved','stellar_zform_cpl':'gc25_cpl'}
RECORD=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def directory(name):
    if (ROOT/'release_results/production'/name/'summary.json').exists():
        return ROOT/'release_results/production'/name
    return ROOT/'fresh_results/production'/ALIASES.get(name,name) if name in ALIASES or (ROOT/'fresh_results/production'/name).exists() else HERE/'runs'/name
def summary(name):return json.loads((directory(name)/'summary.json').read_text())
def posterior(name):
    z=np.load(directory(name)/'posterior.npz')
    return {n:z['samples'][:,i] for i,n in enumerate(z['names'].tolist())}
def register(name,cases,kind):
    RECORD.append({'figure':name,'renderer':kind,'cases':cases,
      'sources':{c:{'folder':directory(c).relative_to(ROOT).as_posix(),
                   'posterior_sha256':sha(directory(c)/'posterior.npz'),
                   'sampling_batch':'gc25_cpl_followup' if directory(c).is_relative_to(ROOT/'release_results') else ('principal' if directory(c).is_relative_to(ROOT/'fresh_results') else 'supplementary')} for c in cases}})
def triangle(name,cases,params,legends=None):
    bt=refresh.bt
    if not bt.HAS_GETDIST:raise RuntimeError('The specified basetools requires GetDist')
    arrays=[np.column_stack([posterior(c)[p] for p in params]) for c in cases]
    labels=[refresh.LABEL[p] for p in params]
    ranges=[{f'p{i}':refresh.BOUND[p] for i,p in enumerate(params) if p in refresh.BOUND} for _ in cases]
    common=dict(width_inch=6.0 if len(params)==3 else 7.05,
      axes_fontsize=10 if len(params)<5 else 9,lab_fontsize=12 if len(params)<5 else 10,
      legend_fontsize=9,smooth_1d=.45,smooth_2d=.65,return_plot=True)
    if len(cases)==1:
        stats=bt.calculate_stats(arrays[0],labels)
        for i,st in enumerate(stats):st['2sigma']=np.percentile(arrays[0][:,i],[2.5,97.5]).tolist()
        g=bt.plot_getdist_advanced(arrays[0],labels,stats,filename=str(OUT/(name+'.pdf')),
           show_titles=True,title_fontsize=9 if len(params)<5 else 7.8,
           interval_labels=['95% interval','68% interval'],parameter_ranges=ranges[0],**common)
    else:
        g=bt.plot_getdist_comparison(arrays,labels,legend_labels=legends,
          colors=[basic.BLUE,basic.ORANGE,basic.GREEN,basic.PURPLE][:len(cases)],
          filename=str(OUT/(name+'.pdf')),ranges_list=ranges,
          filled=[True]+[False]*(len(cases)-1),alpha_filled=.6,**common)
    if g is None:raise RuntimeError('Specified plotting API returned no plot')
    if name=='age_anchor_comparison':
        for p,lo,hi in [('H0',72.,74.08),('MB',-19.280,-19.226)]:
            i=params.index(p);g.subplots[i,i].axvspan(lo,hi,color='#BD5665',alpha=.15,zorder=.1)
    g.fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    g.fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(g.fig)
    register(name,cases,'UNMODIFIED basetools/cosmo_tools.py / GetDist')
    RECORD[-1].update(parameters=params,contours=[.68,.95],diagonal_intervals=[.68,.95],
       array_sizes=[len(a) for a in arrays],sample_filtering='none',parameter_ranges=ranges,
       numeric_titles=len(cases)==1,smoothing_1d=.45,smoothing_2d=.65)
def check_aliases():
    records=[]
    for old,new in ALIASES.items():
        a=json.loads((HERE/'runs'/old/'summary.json').read_text())['config']
        b=summary(old)['config']
        # Compare scientific settings, allowing explicit defaults and seed/name changes.
        defaults={'radiation':True,'use_zhel':True,'H0_bounds':[45.,100.],'Om_bounds':[.1,.6]}
        keys=(set(a)|set(b))-{'name','seed'}
        differences={k:[a.get(k,defaults.get(k)),b.get(k,defaults.get(k))] for k in keys
                     if a.get(k,defaults.get(k))!=b.get(k,defaults.get(k))}
        if differences:raise RuntimeError(f'Scientific config mismatch {old}: {differences}')
        records.append({'original_alias':old,'fresh_configuration':new,'same_scientific_settings':True,
          'historical_seed':a['seed'],'fresh_seed':b['seed'],'fresh_summary_sha256':sha(directory(old)/'summary.json')})
    (ROOT/'audit/result_aliases.json').write_text(json.dumps(records,indent=2),encoding='utf8')
