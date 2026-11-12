"""Restore the original figure topics using the user's basetools GetDist API.

No MCMC chains are changed. New numerical diagrams evaluate the same corrected
background, with four Numba threads. Original art is archived separately.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='4'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,importlib.util,json,hashlib,math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from numba import set_num_threads,njit,prange
from cosmology import Data,Likelihood,integrals_batch,formation_times_batch,OMEGA_R_H2,GYR_KM_S_MPC
import make_figures as basic

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=ROOT/'aa_revision/figures'
spec=importlib.util.spec_from_file_location('user_basetools',ROOT/'basetools/cosmo_tools.py')
bt=importlib.util.module_from_spec(spec);spec.loader.exec_module(bt)
if not bt.HAS_GETDIST:raise RuntimeError('GetDist is required; no substitute triangle renderer is permitted')
LABEL={'Om':r'\Omega_{\mathrm{m}}','H0':r'H_0','MB':r'M_B','rd':r'r_{\mathrm{d}}',
       'Ok':r'\Omega_k','w0':r'w_0','wa':r'w_a','t0':r't_0'}
BOUND={'Om':(.1,.6),'H0':(45,100),'rd':(80,220),'Ok':(-.5,.5),'w0':(-2,0),'wa':(-3,2)}
# Display limits are independent of sampling priors. The automatic 1D limit
# at Om=0.3591 left the smoothed CPL joint contours against the right frame.
DISPLAY_LIMITS={'cpl_triangle':{'Om':(.1,.4)},
                'age_anchor_comparison':{'H0':(50,77),'MB':(-20.,-19.2)}}
BLUE=basic.BLUE;ORANGE=basic.ORANGE;GREEN=basic.GREEN;PURPLE=basic.PURPLE
RECORD=[]
def posterior(name):return basic.posterior(name)
def summary(name):return basic.summary(name)
def filehash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def register(name,cases,kind):
    RECORD.append({'figure':name,'renderer':kind,'cases':cases,
      'posterior_sha256':{c:filehash(HERE/'runs'/c/'posterior.npz') for c in cases}})
def save(fig,name):basic.save(fig,name)

@njit(cache=True,parallel=True)
def formation_each(physical,zf,ax,aw):
    """One continuously proposed formation redshift per posterior cosmology."""
    out=np.empty(len(physical))
    for i in prange(len(physical)):
        om,h,ok,w0,wa=physical[i];ore=OMEGA_R_H2/(h/100.)**2
        ode=1-om-ok-ore;xmax=1/math.sqrt(1+zf[i]);integ=0.
        for q in range(len(ax)):
            x=ax[q]*xmax;a=x*x
            de=a**(-3*(1+w0+wa))*math.exp(-3*wa*(1-a))
            val=ore+om*a+ok*a*a+ode*a**4*de
            integ+=aw[q]*2*x**3/math.sqrt(val)*xmax
        out[i]=GYR_KM_S_MPC/h*integ
    return out

def triangle(name,cases,params,legends=None):
    labels=[LABEL[p] for p in params]
    arrays=[np.column_stack([posterior(c)[p] for p in params]) for c in cases]
    ranges=[{f'p{i}':BOUND[p] for i,p in enumerate(params) if p in BOUND} for c in cases]
    common=dict(width_inch=6.0 if len(params)==3 else 7.05,
          axes_fontsize=10 if len(params)==3 else 9,lab_fontsize=12,legend_fontsize=10,
          smooth_1d=.45,smooth_2d=.65,return_plot=True)
    display=DISPLAY_LIMITS.get(name,{})
    if display:
        common['param_limits']={f'p{params.index(p)}':limit for p,limit in display.items()}
    if len(cases)==1:
        stats=bt.calculate_stats(arrays[0],labels)
        for i,st in enumerate(stats):st['2sigma']=np.percentile(arrays[0][:,i],[2.5,97.5]).tolist()
        g=bt.plot_getdist_advanced(arrays[0],labels,stats,filename=str(OUT/(name+'.pdf')),
           title_fontsize=10 if len(params)==3 else 8.5,show_titles=len(params)<5,
           interval_labels=['95% interval','68% interval'],parameter_ranges=ranges[0],**common)
    else:
        g=bt.plot_getdist_comparison(arrays,labels,legend_labels=legends,
           colors=[BLUE,ORANGE,GREEN,PURPLE][:len(cases)],filename=str(OUT/(name+'.pdf')),
           ranges_list=ranges,filled=[True]+[False]*(len(cases)-1),alpha_filled=.65,**common)
    if g is None:raise RuntimeError('User plotting API did not return a plot')
    if name=='age_anchor_comparison':
        for p,lo,hi in [('H0',72.,74.08),('MB',-19.280,-19.226)]:
            i=params.index(p);g.subplots[i,i].axvspan(lo,hi,color='#BD5665',alpha=.15,zorder=.1)
        g.fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    g.fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(g.fig);register(name,cases,'basetools.cosmo_tools / GetDist')
    if display:RECORD[-1]['display_limits']=display

def triangles():
    triangle('core_noBAO',['gc2026_no_bao'],['Om','H0','MB'])
    triangle('age_anchor_comparison',['gc2026_no_bao','stellar_zform_no_bao','hd_only_extra_0.5'],
             ['Om','H0','MB'],['GC26','GC25','HD 140283, extra systematic 0.5 Gyr'])
    triangle('bao_triangle',['gc2026_bao'],['Om','H0','MB','rd'])
    triangle('main_triangle',['gc2026_no_bao','gc2026_bao'],['Om','H0','MB'],
             ['Pantheon+ + GC26','Pantheon+ + GC26 + DESI DR2'])
    triangle('curvature_triangle',['gc2026_curvature'],['Om','Ok','H0','rd'])
    triangle('cpl_triangle',['gc2026_cpl'],['Om','H0','rd','w0','wa'])
    triangle('joint_anchors_triangle',['gc2025_no_bao','hd_only_extra_0.5','joint_rho_0.0'],
             ['Om','H0','MB'],['GC25 cosmic-age summary','HD 140283, extra systematic 0.5 Gyr',
                              'GC25 + HD, illustrative correlation zero'])

def mechanism(data):
    om=np.linspace(.24,.43,170);h=np.linspace(59,77,170)
    x,y=np.meshgrid(om,h);p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size),-np.ones(x.size),np.zeros(x.size)]
    age=integrals_batch(p,np.array([.01]),data.gx,data.gw,data.ax,data.aw,True)[1].reshape(x.shape)
    fig,ax=plt.subplots(figsize=(7.05,3.85))
    ax.axvspan(.31499,.35129,color='#777777',alpha=.11,label='Pantheon+ shape, 68% interval')
    ax.axhspan(66.9,67.9,color=BLUE,alpha=.12,label='Planck 2018 reference')
    ax.axhspan(72.,74.08,color=ORANGE,alpha=.10,label='SH0ES 2022 reference')
    cs=ax.contour(x,y,age,levels=[12.5,13.5,14.5],colors=[ORANGE,BLUE,GREEN],linewidths=1.6)
    ax.clabel(cs,fmt=lambda z:f'{z:.1f} Gyr',fontsize=9)
    for case,col,lab in [('gc2026_no_bao',BLUE,'GC26, no BAO'),('gc2026_bao',ORANGE,'GC26 + BAO')]:
        s=summary(case)['parameters'];ax.plot(s['Om']['median'],s['H0']['median'],'o',c=col,ms=5,label=lab)
    ax.set_xlabel(r'$\Omega_{\rm m}$');ax.set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
    ax.legend(loc='lower left',ncol=2,frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'mechanism');register('mechanism',[], 'corrected parallel age integral')

def ruler():
    cases=['stellar_zform_bao','gc2026_bao','hd_bao_extra_0.5']
    low=min(np.percentile(posterior(n)['rd'],.05) for n in cases)-2
    high=max(np.percentile(posterior(n)['rd'],99.95) for n in cases)+2
    fig,ax=plt.subplots(figsize=(7.05,3.0));x=np.linspace(low,high,800)
    ax.axvspan(147.09-.26,147.09+.26,color='#999999',alpha=.25,
        label=r'Planck 2018 flat $\Lambda$CDM (external)')
    for n,col,lab in [('stellar_zform_bao',BLUE,'GC25 + BAO'),('gc2026_bao',ORANGE,'GC26 + BAO'),
                     ('hd_bao_extra_0.5',GREEN,'HD + BAO, extra systematic 0.5 Gyr')]:
        s=posterior(n)['rd'];p=gaussian_kde(s)(x)
        ax.plot(x,p,c=col,label=lab,ls='--' if n.startswith('hd') else '-')
        ax.fill_between(x,0,p,color=col,alpha=.10)
        ax.axvline(np.median(s),c=col,ls='--',lw=.8)
    ax.set_xlabel(r'$r_{\rm d}$ [Mpc]');ax.set_ylabel('Posterior density [Mpc$^{-1}$]')
    ax.legend(frameon=False,fontsize=7,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02))
    ax.set_xlim(low,high);ax.set_ylim(bottom=0)
    fig.tight_layout();save(fig,'rd_distribution');register('rd_distribution',
       cases,'three age-calibration rulers with a Planck external reference; no BBN-only prior')

def age_curve(data):
    age_grid=np.linspace(12.1,16.1,260)
    om_quantiles=np.array([.3149905607,.3329138152,.3512890742])
    target=np.tile(age_grid,3);om=np.repeat(om_quantiles,len(age_grid))
    h=np.full(len(target),69.)
    for _ in range(6):
        p=np.c_[om,h,np.zeros(len(h)),-np.ones(len(h)),np.zeros(len(h))]
        ages=integrals_batch(p,np.array([.01]),data.gx,data.gw,data.ax,data.aw,True)[1]
        h*=ages/target
    curves=h.reshape(3,-1)
    fig,ax=plt.subplots(figsize=(7.05,3.5))
    ax.fill_between(age_grid,curves[2],curves[0],color='#97B8D3',alpha=.35,
         label='Mapping of SN-only 68% matter-density interval')
    ax.plot(age_grid,curves[1],c=BLUE,lw=1.5,label=r'SN-only median $\Omega_{\rm m}$')
    ax.axhspan(72,74.08,color=ORANGE,alpha=.11,label='SH0ES 2022 reference')
    for n,c,lab in [('stellar_zform_no_bao',BLUE,'GC25 + SN'),('gc2026_no_bao',ORANGE,'GC26 + SN'),
                    ('hd_only_extra_0.5',GREEN,'HD + SN, extra systematic 0.5 Gyr')]:
        s=summary(n)['parameters'];a=s['t0'];b=s['H0']
        ax.errorbar(a['median'],b['median'],xerr=[[a['median']-a['q16']],[a['q84']-a['median']]],
            yerr=[[b['median']-b['q16']],[b['q84']-b['median']]],fmt='o',color=c,capsize=2,label=lab)
    ax.set_xlabel(r'Cosmic age $t_0$ [Gyr]');ax.set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
    ax.set_xlim(age_grid[[0,-1]]);ax.set_ylim(56,79);ax.legend(frameon=False,fontsize=6.9,ncol=2)
    fig.tight_layout();save(fig,'H0_vs_t0')
    np.savez_compressed(HERE/'age_expansion_curve.npz',cosmic_age=age_grid,H0_curves=curves,
                        Om_q16_q50_q84=om_quantiles)
    register('H0_vs_t0',['stellar_zform_no_bao','gc2026_no_bao','hd_only_extra_0.5'],'corrected parallel age mapping with three stellar calibration scenarios')

def formation(data):
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.05));rng=np.random.default_rng(2026102314)
    cases=['gc2026_bao','gc2026_bao_zmax50','gc2026_bao_zmin6','gc2026_bao_logprior']
    labs=[r'Uniform $z_f$: 11--30',r'Uniform $z_f$: 11--50',r'Uniform $z_f$: 6--30',r'Uniform $\ln(1+z_f)$: 11--30']
    for n,col,lab in zip(cases,[BLUE,ORANGE,GREEN,PURPLE],labs):
        cfg=summary(n)['config'];like=Likelihood(data,cfg);po=posterior(n)
        ix=np.arange(0,len(po['H0']),4);t=np.column_stack([po[p][ix] for p in like.names]);p=like.physical(t)
        age=po['t0'][ix];mu=cfg['age']['stellar_mu'];sig=cfg['age']['stellar_sigma']
        # Continuous conditional draws avoid displaying quadrature-node artifacts.
        pending=np.arange(len(t));formation_draw=np.full(len(t),np.nan)
        lo,hi=cfg['age']['zform_range'];attempts=0
        tf_lo=formation_each(p,np.full(len(p),hi),data.ax,data.aw)
        tf_hi=formation_each(p,np.full(len(p),lo),data.ax,data.aw)
        closest=np.maximum(age-tf_hi-mu,0)+np.minimum(age-tf_lo-mu,0)
        envelope=-.5*(closest/sig)**2
        while len(pending):
            u=rng.random(len(pending))
            if cfg['age']['zform_prior']=='uniform_log1pz':
                zf=np.expm1(np.log1p(lo)+u*(np.log1p(hi)-np.log1p(lo)))
            else:zf=lo+u*(hi-lo)
            proposed=formation_each(p[pending],zf,data.ax,data.aw)
            lp=-.5*((age[pending]-proposed-mu)/sig)**2-envelope[pending]
            accepted=rng.random(len(pending))<np.exp(np.minimum(lp,0))
            formation_draw[pending[accepted]]=proposed[accepted]
            pending=pending[~accepted];attempts+=1
            if attempts>20000:raise RuntimeError('Conditional formation draws did not finish')
        check_z=np.array([6.,11.,20.]);check_p=p[:3]
        check=formation_times_batch(check_p,check_z,data.ax,data.aw,True).diagonal()
        assert np.max(np.abs(check-formation_each(check_p,check_z,data.ax,data.aw)))<1e-12
        x=np.linspace(.025,1.2,600);density=gaussian_kde(formation_draw)(x)
        axs[0].plot(x,density,c=col,label=lab)
        if n==cases[0]:
            star=age-formation_draw;x2=np.linspace(12.4,14.9,550)
            axs[1].plot(x2,gaussian_kde(star)(x2),c=BLUE,label=r'Inferred stellar age $t_\star$')
            axs[1].plot(x2,gaussian_kde(age)(x2),c=ORANGE,label=r'Inferred cosmic age $t_0$')
            np.savez_compressed(HERE/'formation_time_draws.npz',formation_time=formation_draw,
                 stellar_age=star,cosmic_age=age,seed=2026102314)
    axs[0].set_xlabel(r'Formation time $t(z_f)$ [Gyr]');axs[0].set_ylabel('Posterior density [Gyr$^{-1}$]')
    axs[0].set_xlim(.025,1.2);axs[0].legend(frameon=False,fontsize=6.7)
    axs[1].set_xlabel('Age [Gyr]');axs[1].legend(frameon=False,fontsize=7,
         loc='upper left',bbox_to_anchor=(0,1.20))
    for ax in axs:ax.set_ylim(bottom=0)
    fig.tight_layout();save(fig,'formation_delay');register('formation_delay',cases,'conditional formation-redshift posterior')

def bound():
    fig,ax=plt.subplots(figsize=(3.4,2.7));x=np.linspace(44,80,600)
    for n,c,lab in [('gc_stellar_lower_bound',BLUE,r'Bound only, $H_0>45$'),
                    ('gc_stellar_lower_bound_H0min50',ORANGE,r'Bound only, $H_0>50$'),
                    ('stellar_zform_no_bao',GREEN,'Two-sided GC25 clock')]:
        den=gaussian_kde(posterior(n)['H0'])(x)
        lo=50 if 'min50' in n else 45;den[x<lo]=np.nan
        ax.plot(x,den,c=c,label=lab)
    ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]');ax.set_ylabel('Marginal posterior density')
    ax.legend(frameon=False,fontsize=7);fig.tight_layout();save(fig,'age_bound')
    register('age_bound',['gc_stellar_lower_bound','gc_stellar_lower_bound_H0min50','stellar_zform_no_bao'],'one-sided age posterior comparison')

def main():
    OUT.mkdir(exist_ok=True);set_num_threads(4)
    triangles();data=Data()
    for f in [lambda:mechanism(data),ruler,lambda:age_curve(data),lambda:formation(data),bound,
              basic.model_forest,basic.formation,basic.cpl,basic.age_width,basic.fixed_delay,basic.residuals,basic.hd]:
        f();print('Saved',getattr(f,'__name__','plot'),flush=True)
    register('model_comparison',['gc2026_no_bao','gc2026_bao','gc2026_curvature','gc2026_cpl'],'posterior summary forest')
    register('formation_sensitivity',['gc2026_bao','gc2026_bao_zmax50','gc2026_bao_zmin6',
        'gc2026_bao_logprior','gc2026_cosmic_gaussian_bao','stellar_zform_bao','corrected2021_bao'],'posterior summary forest')
    register('cpl_priors',['gc2026_cpl','gc2026_cpl_restricted_Om','gc2026_cpl_early_subdominant'],'two-dimensional prior diagnostic')
    register('age_width',['stellar_zform_no_bao','stellar_zform_sigma_0.40','stellar_zform_sigma_0.50',
        'stellar_zform_sigma_0.70','gc2026_no_bao','gc2026_sigma_0.40','gc2026_sigma_0.50','gc2026_sigma_0.70'],
        'H0, MB and descriptive H0 difference versus total age uncertainty')
    fixed=[f'fixed_delay_{tag}_{.05*j:.2f}'.replace('.','p') for tag in ['gc25','gc26'] for j in range(13)]
    register('fixed_delay_scan',fixed+['stellar_zform_no_bao','gc2026_no_bao'],
        'fixed stellar formation-delay slices with H0 and MB; not a new cosmic-age observation')
    register('residuals',['gc2026_bao'],'full-covariance residual diagnostic')
    register('hd_sensitivity',['hd_only_extra_0.0','hd_only_extra_0.5','hd_only_extra_1.0',
        'joint_rho_0.0','joint_rho_0.5','joint_rho_1.0'],'single-star covariance sensitivity')
    manifest={'tool_file':str(ROOT/'basetools/cosmo_tools.py'),
      'tool_sha256':filehash(ROOT/'basetools/cosmo_tools.py'),'figures':RECORD,
      'all_triangles_use_user_basetools':True,'existing_MCMC_results_changed':False,
      'additional_diagnostic_configurations':30,
      'contour_probabilities':[.68,.95],'one_dimensional_shading_central_probabilities':[.68,.95],
      'numerical_threads_for_new_diagrams':4}
    (HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('User GetDist rendering complete:',len(RECORD),'new/replaced diagrams')
if __name__=='__main__':main()
