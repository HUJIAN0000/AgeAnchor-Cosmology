"""Build numerical TeX tables, frozen configuration lists and audit summaries."""
from pathlib import Path
import csv
import html
import hashlib
import json
import re
import unicodedata
import numpy as np

HERE=Path(__file__).resolve().parent;PAPER=HERE.parent/'aa_revision'
MAIN=[('gc2026_no_bao',r'GC26, flat $\Lambda$CDM, no BAO'),
      ('gc2026_bao',r'GC26, flat $\Lambda$CDM + BAO'),
      ('gc2026_curvature',r'GC26, curved $\Lambda$CDM + BAO'),
      ('gc2026_cpl',r'GC26, flat CPL + BAO'),
      ('stellar_zform_no_bao',r'GC25, flat $\Lambda$CDM, no BAO'),
      ('stellar_zform_bao',r'GC25, flat $\Lambda$CDM + BAO')]
def load(name):return json.loads((HERE/'runs'/name/'summary.json').read_text())
def fmt(s,n,d=2):
    if n not in s['parameters']:return r'\cdots'
    q=s['parameters'][n];m=q['median']
    return f'{m:.{d}f}'+'^{+'+f'{q["q84"]-m:.{d}f}'+'}_{-'+f'{m-q["q16"]:.{d}f}'+'}'
def w(name,txt): (PAPER/name).write_text(txt,encoding='utf8')
def mathnum(x,d=2):return f'{float(x):.{d}f}'

def tables():
    rows=[]
    for name,lab in MAIN:
        s=load(name)
        rows.append(lab+' & '+' & '.join('$'+fmt(s,n,d)+'$' for n,d in [('Om',3),('H0',2),('MB',3),('rd',1)])+r' \\')
    w('table_main.tex','\n'.join(rows)+'\n')
    rows=[]
    for name,lab in MAIN[:4]:
        s=load(name);f=s['bestfit']
        rows.append(lab.replace('GC26, ','')+' & '+' & '.join(f'{f[n]:.3f}' for n in ['chi2_SN','chi2_BAO','minus2logL_age_up_to_constant'])+r' \\')
    w('table_fit.tex','\n'.join(rows)+'\n')
    cases=[('gc2026_bao','GC26: stellar, uniform $z_f$ 11--30'),
      ('gc2026_bao_zmax50','GC26: stellar, uniform $z_f$ 11--50'),
      ('gc2026_bao_zmin6','GC26: stellar, uniform $z_f$ 6--30'),
      ('gc2026_bao_logprior',r'GC26: stellar, uniform $\ln(1+z_f)$'),
      ('gc2026_cosmic_gaussian_bao','GC26: published cosmic-age Gaussian'),
      ('stellar_zform_bao','GC25: stellar, uniform $z_f$ 11--30'),
      ('corrected2021_bao','GC21: published cosmic-age Gaussian')]
    w('table_formation.tex','\n'.join(lab+' & $'+fmt(load(n),'H0')+'$ & $'+fmt(load(n),'rd',1)+r'$ \\' for n,lab in cases)+'\n')
    rows=[]
    diagnostic=[('gc2026_no_bao','GC26, no BAO'),('gc2026_bao','GC26, flat + BAO'),
      ('gc2026_curvature','GC26, curved + BAO'),('gc2026_cpl','GC26, CPL + BAO'),
      ('gc2026_cpl_seed2','GC26, CPL, second seed'),
      ('gc2026_cpl_restricted_Om',r'GC26, CPL, $\Omega_{\rm m}>0.25$'),
      ('gc2026_cpl_early_subdominant',r'GC26, CPL, $w_0+w_a<0$'),
      ('stellar_zform_cpl','GC25, CPL + BAO'),('stellar_zform_cpl_seed2','GC25, CPL, second seed')]
    for n,lab in diagnostic:
        s=load(n)
        rows.append(lab+f' & {s["steps"]:,} & {max(s["tau_production"]):.0f} & {min(s["ESS_production"]):.0f}'+r' \\')
    w('table_convergence.tex','\n'.join(rows)+'\n')
    rows=[]
    for n,lab in [('gc2026_cpl','Broad prior'),('gc2026_cpl_restricted_Om',r'$\Omega_{\rm m}>0.25$'),
                  ('gc2026_cpl_early_subdominant',r'$w_0+w_a<0$')]:
        s=load(n)
        rows.append(lab+' & '+' & '.join('$'+fmt(s,k,d)+'$' for k,d in [('Om',3),('w0',3),('wa',2)])+r' \\')
    w('table_cpl.tex','\n'.join(rows)+'\n')

def numbers():
    lines=[]
    def add(k,v):lines.append('\\newcommand{\\'+k+'}{'+str(v)+'}')
    for n,prefix in [('gc2026_no_bao','NoBao'),('gc2026_bao','Bao'),
                     ('gc2026_curvature','Curve'),('gc2026_cpl','Cpl'),
                     ('stellar_zform_bao','OldBao')]:
        s=load(n)
        for k,l,d in [('H0','H',2),('MB','M',3),('rd','R',1),('Om','Om',3),('Ok','Ok',3),('w0','W',3),('wa','Wa',2)]:
            if k in s['parameters']:add(prefix+l,fmt(s,k,d))
        add(prefix+'SHdiff',mathnum(s['H0_reference_difference_sigma']['SH0ES2022'],2))
    for n,k in [('gc_stellar_lower_bound','Lower'),('gc_stellar_lower_bound_H0min50','LowerFifty')]:
        s=load(n);q=s['parameters']['H0'];add(k+'Median',mathnum(q['median'],1));add(k+'Upper',mathnum(q['q975'],1))
    for n,k in [('stellar_zform_sigma_0.70','WideOld'),('gc2026_sigma_0.70','WideNew')]:
        s=load(n);add(k+'H',fmt(s,'H0',2));add(k+'SHdiff',mathnum(s['H0_reference_difference_sigma']['SH0ES2022'],2))
    for n,k in [('fixed_delay_gc25_0p00','FixedOldZero'),('fixed_delay_gc25_0p60','FixedOldSix'),
                 ('fixed_delay_gc26_0p00','FixedNewZero'),('fixed_delay_gc26_0p60','FixedNewSix')]:
        if (HERE/'runs'/n/'summary.json').exists():
            s=load(n);add(k+'H',fmt(s,'H0',2));add(k+'M',fmt(s,'MB',3))
    v=json.loads((HERE/'validation.json').read_text());assert v['passed']
    b=json.loads((HERE/'parallel_benchmark.json').read_text())
    add('SpeedFour',mathnum(b['speedup_vs_one_thread']['4'],1))
    add('SpeedEight',mathnum(b['speedup_vs_one_thread']['8'],1))
    w('numbers.tex','% Automatically generated from stored posterior summaries.\n'+'\n'.join(lines)+'\n')

def bibliography():
    refs=json.loads((HERE/'references'/'verified.json').read_text())
    journals={'The Astrophysical Journal':'ApJ','The Astrophysical Journal Letters':'ApJL',
      'Astronomy & Astrophysics':r'A\&A','Monthly Notices of the Royal Astronomical Society':'MNRAS',
      'Physical Review D':'Phys. Rev. D','Physical Review Letters':'Phys. Rev. Lett.',
      'Journal of Cosmology and Astroparticle Physics':'JCAP','Publications of the Astronomical Society of the Pacific':'PASP',
      'International Journal of Modern Physics D':'Int. J. Mod. Phys. D','Journal of High Energy Astrophysics':'JHEAp'}
    journals.update({'Nature Astronomy':'Nat. Astron.','Classical and Quantum Gravity':'Class. Quantum Grav.',
      'Physics Reports':'Phys. Rep.','Annual Review of Nuclear and Particle Science':'Annu. Rev. Nucl. Part. Sci.',
      'The Astrophysical Journal Supplement Series':'ApJS','The Astronomical Journal':'AJ',
      'Publications of the Astronomical Society of Australia':'PASA',
      'Communications in Applied Mathematics and Computational Science':'Commun. Appl. Math. Comput. Sci.',
      'Bayesian Analysis':'Bayesian Anal.'})
    def esc(t):
        if t.isupper():t=t.title()
        t=t.replace('ø','o').replace('Ø','O').replace('ł','l')
        t=unicodedata.normalize('NFKD',t).encode('ascii','ignore').decode()
        return t.replace('&',r'\&').replace('_',r'\_')
    entries=[]
    for key,r in refs.items():
        year=2020 if key=='Planck2018' else r['issued']['date-parts'][0][0]
        aa=r['authors'];first=aa[0].get('family',aa[0].get('name','Collaboration'))
        if key=='Planck2018':label='Planck Collaboration';authors='Planck Collaboration'
        elif key=='DESI2025':label='DESI Collaboration';authors='DESI Collaboration (Abdul Karim, M., et al.)'
        elif key=='GaiaEDR3':label='Gaia Collaboration';authors='Gaia Collaboration (Brown, A.~G.~A., et al.)'
        else:
            label=esc(first)+(' et al.' if len(aa)>2 else (r' \& '+esc(aa[1]['family']) if len(aa)==2 else ''))
            parts=[]
            for a in aa[:3]:
                initials='~'.join(p[0]+'.' for p in a.get('given','').split() if p)
                parts.append(esc(a.get('family',a.get('name','')))+', '+initials)
            authors=', '.join(parts)+(', et al.' if len(aa)>3 else '')
        jname=html.unescape(r['journal'][0])
        journal=journals.get(jname,esc(jname));vol=r['volume']
        page=(r.get('article_number') or r.get('page') or '').split('-')[0]
        if journal=='JCAP':
            vol=r['doi'].split('/')[-2]
        body=f'{authors} {year}, '+r'\href{https://doi.org/'+r['doi']+'}{'+f'{journal}, {vol}, {page}'+'}'
        entries.append((label,year,r'\bibitem['+label+f'({year})'+']{'+key+'} '+body))
    entries.extend([('Valcin et al.',2025,r'\bibitem[Valcin et al.(2025)]{Valcin2025} Valcin, D., Jimenez, R., Seljak, U., \& Verde, L. 2025, arXiv e-prints, \href{https://arxiv.org/abs/2503.19481}{arXiv:2503.19481v1}'),
     ('Valcin et al.',2026,r'\bibitem[Valcin et al.(2026)]{Valcin2026} Valcin, D., Jimenez, R., Lardo, C., Seljak, U., \& Verde, L. 2026, arXiv e-prints, \href{https://arxiv.org/abs/2603.04872}{arXiv:2603.04872v1}')])
    entries.append(('Hogg',1999,r'\bibitem[Hogg(1999)]{Hogg1999} Hogg, D.~W. 1999, arXiv e-prints, \href{https://arxiv.org/abs/astro-ph/9905116}{arXiv:astro-ph/9905116}'))
    entries.sort(key=lambda x:(x[0].lower(),x[1]))
    w('references.tex',r'\begin{thebibliography}{99}'+'\n'+'\n\n'.join(e[2] for e in entries)+'\n'+r'\end{thebibliography}'+'\n')

def audit():
    primary={c['name'] for c in json.loads((HERE/'production_configurations.json').read_text())}
    summaries=[json.loads(p.read_text()) for p in sorted((HERE/'runs').glob('*/summary.json'))
               if p.parent.name in primary]
    if len(summaries)!=50:raise RuntimeError(f'Expected 50 complete configurations, found {len(summaries)}')
    assert all(s['converged'] and min(s['ESS_production'])>4000 for s in summaries)
    assert all(s['execution']['numba_threads']>1 for s in summaries)
    frozen=[];records=[]
    for s in summaries:
        c=json.loads(json.dumps(s['config']))
        if c.get('mock'):c['age']['mu']=13.57
        frozen.append(c)
        records.append({'name':c['name'],'steps':s['steps'],'burnin':s['burnin'],
          'walkers':s['walkers'],'tau_max':max(s['tau_production']),
          'ESS_min':min(s['ESS_production']),'acceptance':s['acceptance_mean'],
          'seed':c['seed'],'numba_threads':s['execution']['numba_threads'],'wall_seconds':s['wall_seconds']})
    (HERE/'production_configurations.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')
    with (HERE/'convergence_all.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    pairs=[('gc2025_bao','gc2025_bao_seed2'),('gc2025_cpl','gc2025_cpl_seed2'),
      ('stellar_zform_bao','stellar_zform_bao_seed2'),('stellar_zform_cpl','stellar_zform_cpl_seed2'),
      ('gc2026_bao','gc2026_bao_seed2'),('gc2026_cpl','gc2026_cpl_seed2')]
    comparisons=[]
    for a,b in pairs:
        sa=load(a);sb=load(b)
        for n,qa in sa['parameters'].items():
            qb=sb['parameters'][n];sig=(qa['q84']-qa['q16']+qb['q84']-qb['q16'])/4
            comparisons.append({'case':a,'parameter':n,'median_difference_in_posterior_sigma':abs(qa['median']-qb['median'])/sig})
    out={'n_configurations':len(summaries),'all_converged':True,
      'numerical_thread_counts':sorted({s['execution']['numba_threads'] for s in summaries}),
      'independent_seed_comparisons':comparisons,
      'max_seed_difference_in_posterior_sigma':max(r['median_difference_in_posterior_sigma'] for r in comparisons),
      'mock_recovery':load('gc2025_bao_mock')['mock_truth_in_95_interval']}
    (HERE/'final_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    (HERE/'progress_all.json').write_text(json.dumps({'state':'complete','total':len(summaries),
       'completed':[s['config']['name'] for s in summaries]},indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='independent_seed_comparisons'},indent=2))

if __name__=='__main__':
    audit();tables();numbers();bibliography()
