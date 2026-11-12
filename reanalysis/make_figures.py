"""Regenerate publication figures directly from the frozen posterior files."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.linalg import cho_solve,cho_factor
import corner
from cosmology import Data,Likelihood,C_LIGHT

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'aa_revision'/'figures'
OUT.mkdir(exist_ok=True)
BLUE='#2369A1';ORANGE='#D77728';GREEN='#29836C';PURPLE='#815398'
plt.rcParams.update({'font.size':9,'axes.labelsize':10,'legend.fontsize':8,
 'xtick.labelsize':8,'ytick.labelsize':8,'font.family':'serif',
 'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.7})

def summary(name):return json.loads((HERE/'runs'/name/'summary.json').read_text())
def posterior(name):
    z=np.load(HERE/'runs'/name/'posterior.npz')
    return {n:z['samples'][:,i] for i,n in enumerate(z['names'])}
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=180)
    plt.close(fig)
def error(ax,s,param,y,color=BLUE,label=None):
    q=s['parameters'][param];m=q['median']
    ax.errorbar(m,y,xerr=[[m-q['q16']],[q['q84']-m]],fmt='o',color=color,
                ms=3.5,capsize=2,lw=1.1,label=label)

def triangle():
    from refresh_figures import triangle as user_triangle
    user_triangle('main_triangle',['gc2026_no_bao','gc2026_bao'],['Om','H0','MB'],
                  ['Pantheon+ + GC26','Pantheon+ + GC26 + DESI DR2'])

def model_forest():
    cases=['gc2026_no_bao','gc2026_bao','gc2026_curvature','gc2026_cpl']
    labs=[r'Flat $\Lambda$CDM, no BAO',r'Flat $\Lambda$CDM + BAO',r'Curved $\Lambda$CDM + BAO',r'Flat CPL + BAO']
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.65),sharey=True)
    for ax,param,xlab in zip(axs,['H0','MB','rd'],
          [r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]',r'$M_B$ [mag]',r'$r_{\rm d}$ [Mpc]']):
        for y,name,c in zip(range(4),cases,[BLUE,ORANGE,GREEN,PURPLE]):
            if param in summary(name)['parameters']:error(ax,summary(name),param,y,c)
        ax.set_xlabel(xlab);ax.grid(axis='x',alpha=.15);ax.set_ylim(3.5,-.8)
    axs[0].axvspan(66.9,67.9,color='#999999',alpha=.22,label='Planck 2018')
    axs[0].axvspan(72.,74.08,color='#BD5665',alpha=.15,label='SH0ES 2022')
    axs[0].set_yticks(range(4),labs);axs[0].set_xlim(60,75)
    axs[1].set_xticks([-19.5,-19.4,-19.3])
    axs[0].legend(loc='upper left',bbox_to_anchor=(0,1.2),ncol=2,fontsize=6,frameon=False)
    fig.tight_layout(w_pad=1.5);save(fig,'model_comparison')

def formation():
    cases=['gc2026_bao','gc2026_bao_zmax50','gc2026_bao_zmin6',
           'gc2026_bao_logprior','gc2026_cosmic_gaussian_bao','stellar_zform_bao','corrected2021_bao']
    labels=[r'GC26: uniform $z_f\in[11,30]$',r'GC26: uniform $z_f\in[11,50]$',
            r'GC26: uniform $z_f\in[6,30]$',r'GC26: uniform $\ln(1+z_f)$',
            r'GC26: published cosmic-age Gaussian','GC25: stellar-age model','GC21: cosmic-age model']
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.35),sharey=True)
    for i,name in enumerate(cases):
        for ax,p in zip(axs,['H0','rd']):error(ax,summary(name),p,i,BLUE if i<5 else ORANGE)
    for ax,xlab in zip(axs,[r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]',r'$r_{\rm d}$ [Mpc]']):
        ax.set_xlabel(xlab);ax.set_ylim(6.6,-.6);ax.grid(axis='x',alpha=.15)
    axs[0].set_yticks(range(len(labels)),labels)
    fig.tight_layout();save(fig,'formation_sensitivity')

def density(ax,x,y,color):
    h,xe,ye=np.histogram2d(x,y,bins=65)
    h=gaussian_filter(h,1.15)
    ordered=np.sort(h.ravel())[::-1];cum=np.cumsum(ordered)/h.sum()
    lev=sorted([ordered[np.searchsorted(cum,p)] for p in [.68,.95]])
    xc=(xe[1:]+xe[:-1])/2;yc=(ye[1:]+ye[:-1])/2
    ax.contour(xc,yc,h.T,levels=lev,colors=color,linewidths=[.8,1.3])
def cpl():
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.6))
    cases=['gc2026_cpl','gc2026_cpl_restricted_Om','gc2026_cpl_early_subdominant']
    for name,col in zip(cases,[BLUE,ORANGE,GREEN]):
        p=posterior(name)
        for ax,a,b in zip(axs,['Om','w0','H0'],['wa','wa','rd']):density(ax,p[a],p[b],col)
    for ax,xlab,ylab in zip(axs,[r'$\Omega_{\rm m}$',r'$w_0$',r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]'],
                           [r'$w_a$',r'$w_a$',r'$r_{\rm d}$ [Mpc]']):
        ax.set_xlabel(xlab);ax.set_ylabel(ylab)
    axs[0].set_xlim(.10,.35);axs[1].plot([-1.08,-.7],[1.08,.7],ls=':',c='gray',lw=.8)
    fig.legend([Line2D([0],[0],color=c) for c in [BLUE,ORANGE,GREEN]],
       [r'$\Omega_{\rm m}>0.1$',r'$\Omega_{\rm m}>0.25$',r'$w_0+w_a<0$'],
       loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.06))
    fig.tight_layout();save(fig,'cpl_priors')

def age_width():
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.95))
    for prefix,sigmas,col,label in [('stellar_zform',[np.hypot(.1,.23),.4,.5,.7],BLUE,'GC25'),
                                  ('gc2026',[np.hypot(.25,.23),.4,.5,.7],ORANGE,'GC26')]:
        for j,sig in enumerate(sigmas):
            name=prefix+'_no_bao' if j==0 else prefix+f'_sigma_{sig:.2f}'
            s=summary(name);q=s['parameters']['H0'];m=q['median'];e=(q['q84']-q['q16'])/2
            axs[0].errorbar(sig,m,yerr=[[m-q['q16']],[q['q84']-m]],fmt='o',color=col,
                           capsize=2,label=label if j==0 else None)
            mb=s['parameters']['MB'];b=mb['median']
            axs[1].errorbar(sig,b,yerr=[[b-mb['q16']],[mb['q84']-b]],fmt='o',color=col,capsize=2)
            axs[2].plot(sig,abs(73.04-m)/np.hypot(e,1.04),'o',color=col)
    axs[0].axhspan(72,74.08,color='#BD5665',alpha=.12)
    axs[1].axhspan(-19.280,-19.226,color='#BD5665',alpha=.12)
    axs[0].set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]');axs[0].legend(frameon=False)
    axs[1].set_ylabel(r'$M_B$ [mag]')
    axs[2].set_ylabel(r'Approximate $H_0$ difference [$\sigma$]')
    for ax in axs:ax.set_xlabel(r'Age uncertainty $\sigma_\star$ [Gyr]');ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'age_width')

def fixed_delay():
    """Conditional sensitivity slices, not an inferred formation history."""
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.0))
    for tag,col,label in [('gc25',BLUE,'GC25'),('gc26',ORANGE,'GC26')]:
        cases=[f'fixed_delay_{tag}_{.05*j:.2f}'.replace('.','p') for j in range(13)]
        delay=np.array([summary(n)['config']['age']['fixed_delay_gyr'] for n in cases])
        for ax,param in zip(axs,['H0','MB']):
            q=[summary(n)['parameters'][param] for n in cases]
            med=np.array([v['median'] for v in q])
            ax.fill_between(delay,[v['q16'] for v in q],[v['q84'] for v in q],color=col,alpha=.15)
            ax.plot(delay,med,'o-',c=col,ms=2.5,lw=1.2,label=label)
        base=summary('stellar_zform_no_bao' if tag=='gc25' else 'gc2026_no_bao')['parameters']
        for ax,param in zip(axs,['H0','MB']):
            ax.axhline(base[param]['median'],color=col,ls=':',lw=.9)
    axs[0].axhspan(72,74.08,color='#BD5665',alpha=.12,label='SH0ES 2022 reference')
    axs[1].axhspan(-19.280,-19.226,color='#BD5665',alpha=.12)
    axs[0].set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
    axs[1].set_ylabel(r'$M_B$ [mag]')
    for ax in axs:
        ax.set_xlabel(r'Fixed formation delay $\Delta t_{\rm form}$ [Gyr]')
        ax.set_xlim(0,.6);ax.grid(alpha=.12)
    axs[0].legend(frameon=False,fontsize=7,loc='upper right')
    fig.tight_layout();save(fig,'fixed_delay_scan')

def residuals():
    data=Data();s=summary('gc2026_bao');like=Likelihood(data,s['config'])
    theta=np.array([s['bestfit']['sampled'][n] for n in like.names])
    sn,ba,ag,alpha,age,coeff=like.components(theta)
    residual=data.r-data.vander@coeff[0]-alpha[0]
    edges=np.geomspace(data.z.min()*.999,data.z.max()*1.001,25)
    groups=np.minimum(np.searchsorted(edges,data.z,side='right')-1,len(edges)-2)
    design=np.eye(len(edges)-1)[groups]
    cinv_design=cho_solve(data.cf,design);cov=np.linalg.inv(design.T@cinv_design)
    binned=cov@cinv_design.T@residual
    centers=np.array([np.exp(np.mean(np.log(data.z[groups==i]))) for i in range(len(edges)-1)])
    dm,_,dh=__import__('cosmology').integrals_batch(like.physical(theta),data.bao_z,
                               data.gx,data.gw,data.ax,data.aw,True)
    scale=C_LIGHT/theta[1];dm=dm[0]*scale;dh=dh[0]*scale
    dv=np.cbrt(data.bao_z*dm**2*dh)
    model=np.where(data.bao_type==0,dm,np.where(data.bao_type==1,dh,dv))/theta[2]
    whiten=np.linalg.solve(np.linalg.cholesky(data.bao_cov),data.bao_y-model)
    fig,axs=plt.subplots(2,1,figsize=(7.05,4.1),gridspec_kw={'height_ratios':[1.3,1]})
    axs[0].scatter(data.z,residual,s=2,color='gray',alpha=.16,rasterized=True)
    axs[0].errorbar(centers,binned,yerr=np.sqrt(np.diag(cov)),fmt='o',ms=3,capsize=2,c=BLUE)
    axs[0].axhline(0,c='black',lw=.7);axs[0].set_xscale('log');axs[0].set_ylim(-.65,.65)
    axs[0].set_xlabel(r'$z_{\rm HD}$');axs[0].set_ylabel('SN residual [mag]')
    axs[1].axhspan(-1,1,color='gray',alpha=.12)
    axs[1].plot(np.arange(1,14),whiten,'o',c=ORANGE,ms=4)
    axs[1].axhline(0,c='black',lw=.7);axs[1].set_xticks(range(1,14))
    axs[1].set_xlabel('BAO vector index (release ordering)');axs[1].set_ylabel('Whitened BAO residual')
    fig.tight_layout();save(fig,'residuals')

def hd():
    cases=['hd_only_extra_0.0','hd_only_extra_0.5','hd_only_extra_1.0',
            'joint_rho_0.0','joint_rho_0.5','joint_rho_1.0']
    labs=[r'HD only: $s_{\rm sys}=0$',r'HD only: $s_{\rm sys}=0.5$ Gyr',r'HD only: $s_{\rm sys}=1.0$ Gyr',
          r'GC25 + HD: $\rho=0$',r'GC25 + HD: $\rho=0.5$',r'GC25 + HD: $\rho=1$']
    fig,ax=plt.subplots(figsize=(3.4,2.55))
    for y,n in enumerate(cases):error(ax,summary(n),'H0',y,BLUE if y<3 else ORANGE)
    ax.set_yticks(range(6),labs);ax.set_ylim(5.6,-.6)
    ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();save(fig,'hd_sensitivity')

if __name__=='__main__':
    from refresh_figures import main
    main()
