"""Publication products from frozen posterior samples; no MCMC is run here."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='4',MPLBACKEND='Agg')
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'aa_revision'
spec=importlib.util.spec_from_file_location('source_resolver',ROOT/'integration/posterior_sources.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
B=R.basic
MODELS=[('flat',r'Flat $\Lambda$CDM'),('curved',r'Curved $\Lambda$CDM'),('cpl','Flat CPL')]
def quantile(n,k):
    p=R.posterior(n);return np.percentile(p['H0']*p['rd'] if k=='H0rd' else p[k],[16,50,84])
def tex(n,k,d=2):
    if k!='H0rd':return R.tables.fmt(R.summary(n),k,d)
    a,b,c=quantile(n,k);return f'{b:.{d}f}^{{+{c-b:.{d}f}}}_{{-{b-a:.{d}f}}}'
def write(n,rows): (PAPER/n).write_text('\n'.join(rows)+'\n',encoding='utf8')
def save(fig,n,cases):
    B.save(fig,n);R.register(n,cases,'Matplotlib; frozen posterior intervals')
def tables():
    macros={}
    for prefix,n in [('BaseNoBao','gc25_no_bao'),('Base','gc25_flat'),('BaseCurve','gc25_curved'),('BaseCpl','gc25_cpl'),('AltNoBao','gc26_no_bao'),('Alt','gc26_flat'),('AltCurve','gc26_curved'),('AltCpl','gc26_cpl')]:
        for k,s,d in [('H0','H',2),('MB','M',3),('rd','R',1),('Om','Om',3),('Ok','Ok',3),('w0','W',3),('wa','Wa',2)]:
            if k in R.summary(n)['parameters']:macros[prefix+s]=tex(n,k,d)
        macros[prefix+'SHdiff']=f"{R.summary(n)['H0_reference_difference_sigma']['SH0ES2022']:.2f}"
    macros.update(BaseQ=tex('gc25_flat','H0rd',0),NoAgeH=tex('noage_flat','H0'),NoAgeQ=tex('noage_flat','H0rd',0))
    for n,pre in [('gc_stellar_lower_bound','Lower'),('gc_stellar_lower_bound_H0min50','LowerFifty')]:
        q=R.summary(n)['parameters']['H0'];macros[pre+'Median']=f"{q['median']:.1f}";macros[pre+'Upper']=f"{q['q975']:.1f}"
    for n,pre in [('stellar_zform_sigma_0.70','WideBase'),('gc2026_sigma_0.70','WideAlt')]:
        macros[pre+'H']=tex(n,'H0');macros[pre+'SHdiff']=f"{R.summary(n)['H0_reference_difference_sigma']['SH0ES2022']:.2f}"
    for age,pre in [('gc25','Base'),('gc26','Alt')]:
        for end,delay in [('Zero','0p00'),('Six','0p60')]:macros['Fixed'+pre+end+'H']=tex(f'fixed_delay_{age}_{delay}','H0')
        late='stellar_zform_bao_zmin6' if age=='gc25' else 'gc2026_bao_zmin6'
        macros[pre+'FormationShift']=f"{R.summary(late)['parameters']['H0']['median']-R.summary(age+'_flat')['parameters']['H0']['median']:.2f}"
    macros['BaseTailFraction']=f"{100*np.mean(R.posterior('gc25_cpl')['Om']<.25):.1f}"
    rows=[]
    for age in ['gc25','gc26']:
        for model,lab in [('no_bao',r'Flat $\Lambda$CDM, no BAO')]+MODELS:
            n=age+'_'+model;rows.append(age.upper()+', '+lab+' & '+' & '.join('$'+tex(n,k,d)+'$' for k,d in [('Om',3),('H0',2),('MB',3),('rd',1)])+r' \\')
    write('table_main.tex',rows)
    write('table_age_information.tex',[lab+', '+age.upper()+' & '+' & '.join('$'+tex(age+'_'+m,k,d)+'$' for k,d in [('H0',2),('rd',1),('H0rd',0)])+r' \\' for m,lab in MODELS for age in ['noage','gc25']])
    form=[('stellar_zform_bao','GC25, uniform $z_f$: 11--30'),('stellar_zform_bao_zmax50','GC25, uniform $z_f$: 11--50'),('stellar_zform_bao_zmin6','GC25, uniform $z_f$: 6--30'),('stellar_zform_bao_logprior',r'GC25, uniform $\ln(1+z_f)$'),('gc2026_bao','GC26, uniform $z_f$: 11--30'),('gc2026_bao_zmax50','GC26, uniform $z_f$: 11--50'),('gc2026_bao_zmin6','GC26, uniform $z_f$: 6--30'),('gc2026_bao_logprior',r'GC26, uniform $\ln(1+z_f)$')]
    write('table_formation.tex',[lab+' & $'+tex(n,'H0')+'$ & $'+tex(n,'rd',1)+r'$ \\' for n,lab in form])
    rows=[]
    for n,lab in [('gc25_no_bao','GC25, no BAO'),('gc25_flat','GC25, flat'),('gc25_curved','GC25, curved'),('gc25_cpl','GC25, CPL')]:
        f=R.summary(n)['bestfit'];rows.append(lab+' & '+' & '.join(f'{f[k]:.3f}' for k in ['chi2_SN','chi2_BAO','minus2logL_age_up_to_constant'])+r' \\')
    write('table_fit.tex',rows)
    write('table_cpl.tex',[lab+' & '+' & '.join('$'+tex(n,k,d)+'$' for k,d in [('Om',3),('w0',3),('wa',2)])+r' \\' for n,lab in [('gc25_cpl','Broad prior'),('gc25_cpl_Om025',r'$\Om>0.25$'),('gc25_cpl_subdominant',r'$w_0+w_a<0$')]])
    labels={
      'gc25_no_bao':r'GC25, flat $\Lambda$CDM, no BAO',
      'gc25_flat':r'GC25, flat $\Lambda$CDM',
      'gc25_curved':r'GC25, curved $\Lambda$CDM',
      'gc25_cpl':'GC25, CPL',
      'gc25_cpl_Om025':r'GC25, CPL, $\Om>0.25$',
      'gc25_cpl_Om025_seed2':r'GC25, CPL, $\Om>0.25$, seed 2',
      'gc25_cpl_subdominant':r'GC25, CPL, $w_0+w_a<0$',
      'gc25_cpl_subdominant_seed2':r'GC25, CPL, $w_0+w_a<0$, seed 2',
      'gc26_flat':r'GC26, flat $\Lambda$CDM',
      'gc26_curved':r'GC26, curved $\Lambda$CDM',
      'gc26_cpl':'GC26, CPL','gc26_cpl_seed2':'GC26, CPL, seed 2',
      'noage_flat':r'No age, flat $\Lambda$CDM','noage_curved':r'No age, curved $\Lambda$CDM',
      'noage_cpl':'No age, CPL','noage_cpl_seed2':'No age, CPL, seed 2'}
    cases=list(labels)
    rows=[]
    for n in cases:
        s=R.summary(n);rows.append(labels[n]+f' & {s["steps"]:,} & {max(s["tau_production"]):.0f} & {min(s["ESS_production"]):.0f}'+r' \\')
    write('table_convergence.tex',rows)
    (ROOT/'audit/review_numbers.json').write_text(json.dumps({'macros':macros,'additional_MCMC_configurations':4,'new_configs':'release_configurations.json'},indent=2),encoding='utf8')
    write('numbers.tex',['% Generated from frozen posterior products.']+['\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items()])
    return form
def triangles():
    for m,params,n in [('flat',['Om','H0','MB','rd'],'flat_joint'),('curved',['Om','Ok','H0','MB','rd'],'curved_joint'),('cpl',['Om','H0','MB','rd','w0','wa'],'cpl_joint')]:
        R.triangle(n,['gc25_'+m,'gc26_'+m],params,['GC25: published calibration','GC26: alternative calibration'])
    R.triangle('age_information_cpl',['gc25_cpl','noage_cpl'],['Om','H0','MB','rd','w0','wa'],['SN + BAO + GC25','SN + BAO, no age'])
    R.triangle('cpl_priors',['gc25_cpl','gc25_cpl_Om025','gc25_cpl_subdominant'],['Om','w0','wa'],['GC25, broad prior',r'GC25, $\Omega_m>0.25$',r'GC25, $w_0+w_a<0$'])
def model_forest():
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.8),sharey=True,layout='constrained');cases=[]
    for i,(m,lab) in enumerate([('no_bao',r'Flat $\Lambda$CDM, no BAO')]+MODELS):
        for age,col,dy in [('gc25',B.BLUE,-.12),('gc26',B.ORANGE,.12)]:
            n=age+'_'+m;cases.append(n)
            for ax,k in zip(axs,['H0','MB','rd']):
                if k in R.summary(n)['parameters']:B.error(ax,R.summary(n),k,i+dy,col,age.upper() if i==0 else None)
    for ax,lab in zip(axs,[r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]',r'$M_B$ [mag]',r'$r_d$ [Mpc]']):ax.set_xlabel(lab);ax.grid(axis='x',alpha=.15);ax.set_ylim(3.5,-.6)
    axs[0].set_yticks(range(4),[r'Flat $\Lambda$CDM, no BAO']+[m[1]+' + BAO' for m in MODELS]);axs[0].axvspan(66.9,67.9,color='#999999',alpha=.2);axs[0].axvspan(72.,74.08,color='#BD5665',alpha=.13);axs[0].legend(frameon=False,fontsize=8,loc='lower left')
    save(fig,'model_comparison',cases)
def formation(form):
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.6),sharey=True,layout='constrained')
    for i,(n,lab) in enumerate(form):
        for ax,k in zip(axs,['H0','rd']):B.error(ax,R.summary(n),k,i,B.BLUE if i<4 else B.ORANGE)
    axs[0].set_yticks(range(8),[lab for _,lab in form])
    for ax,lab in zip(axs,[r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]',r'$r_d$ [Mpc]']):ax.set_xlabel(lab);ax.grid(axis='x',alpha=.15);ax.set_ylim(7.6,-.6)
    save(fig,'formation_sensitivity',[n for n,_ in form])
def scale_information():
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.8),layout='constrained');cases=[]
    for j,k in enumerate(['H0','rd','H0rd']):
        for i,(m,lab) in enumerate(MODELS):
            for age,marker,col,dy in [('noage','s','#777777',.18),('gc25','o',B.BLUE,0),('gc26','^',B.ORANGE,-.18)]:
                n=age+'_'+m;a,b,c=quantile(n,k);axs[j].errorbar(b,i+dy,xerr=[[b-a],[c-b]],fmt=marker,color=col,ms=4,capsize=2,label=age.upper() if i==0 else None)
                if n not in cases:cases.append(n)
        axs[j].set_yticks(range(3),[x[1] for x in MODELS] if j==0 else ['']*3);axs[j].set_ylim(2.5,-.5);axs[j].grid(axis='x',alpha=.15)
        axs[j].set_xlabel({'H0':r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]','rd':r'$r_d$ [Mpc]','H0rd':r'$H_0r_d$ [km s$^{-1}$]'}[k])
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False,fontsize=8);save(fig,'scale_information',cases)
def main():
    R.check_aliases();R.RECORD.clear();B.summary=R.summary;B.posterior=R.posterior;R.refresh.summary=R.summary;R.refresh.posterior=R.posterior
    form=tables();triangles();model_forest();formation(form);scale_information()
    B.age_width();R.register('age_width',['stellar_zform_no_bao','stellar_zform_sigma_0.40','stellar_zform_sigma_0.50','stellar_zform_sigma_0.70','gc2026_no_bao','gc2026_sigma_0.40','gc2026_sigma_0.50','gc2026_sigma_0.70'],'Matplotlib uncertainty scenarios')
    B.fixed_delay();R.register('fixed_delay_scan',[f'fixed_delay_{tag}_{.05*j:.2f}'.replace('.','p') for tag in ['gc25','gc26'] for j in range(13)]+['stellar_zform_no_bao','gc2026_no_bao'],'Matplotlib fixed-delay sensitivity')
    B.summary=lambda n:R.summary('gc25_flat' if n=='gc2026_bao' else n)
    B.residuals();B.summary=R.summary;R.register('residuals',['gc25_flat'],'Full-covariance residuals at GC25 optimized parameters')
    R.refresh.register=R.register;R.refresh.bound()
    (ROOT/'audit/figure_sources.json').write_text(json.dumps({'basetools_sha256':R.sha(ROOT/'basetools/cosmo_tools.py'),'all_triangles_use_specified_tool':True,'additional_MCMC_configurations':4,'reference_calibration':'GC25','alternative_calibration':'GC26','figures':R.RECORD},indent=2),encoding='utf8')
    print('Publication figures:',len(R.RECORD),'triangles:',sum('parameters' in r for r in R.RECORD),flush=True)
if __name__=='__main__':main()
