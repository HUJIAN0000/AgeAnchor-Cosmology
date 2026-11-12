"""Cosmology and full-covariance likelihood for the revised age-anchored analysis.

The SN intercept is analytically marginalized with a broad flat prior, then
conditionally reconstructed. Chebyshev interpolation compresses the *model*,
not the observations/covariance; validate.py checks against direct integration.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from numba import njit,prange
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr, ndtr,logsumexp

C_LIGHT=299792.458
GYR_KM_S_MPC=3.0856775814913673e19/(365.25*86400*1e9)
OMEGA_R_H2=2.469e-5*(1+0.22710731766*3.046)
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True,parallel=True)
def integrals_batch(theta, zs, gx, gw, ax, aw, radiation=True):
    """theta columns: Omega_m, H0, Omega_k, w0, wa. Distances in c/H0 units.

    Integrate age after a=x^2 substitution; reject non-positive H^2 rather
    than dropping intervals. Fixed massless-radiation background is explicit.
    """
    n=theta.shape[0]
    dm=np.empty((n,len(zs))); ages=np.empty(n); dh=np.empty_like(dm)
    # Each row is an independent cosmology/walker. Keep each quadrature sum
    # serial inside its row, so parallel scheduling does not reorder reductions.
    for i in prange(n):
        om,h,ok,w0,wa=theta[i]
        ore=OMEGA_R_H2/(h/100.)**2 if radiation else 0.
        ode=1.-om-ok-ore
        valid=ode>0.
        age=0.
        for q in range(len(ax)):
            x=ax[q]; a=x*x
            de=a**(-3*(1+w0+wa))*math.exp(-3*wa*(1-a))
            val=ore+om*a+ok*a*a+ode*a**4*de
            if val<=0 or not math.isfinite(val): valid=False; break
            age+=aw[q]*2*x**3/math.sqrt(val)
        ages[i]=GYR_KM_S_MPC/h*age if valid else np.nan
        for j in range(len(zs)):
            z=zs[j]; integral=0.
            for q in range(len(gx)):
                zz=z*(gx[q]+1)/2
                de=(1+zz)**(3*(1+w0+wa))*math.exp(-3*wa*zz/(1+zz))
                ee=ore*(1+zz)**4+om*(1+zz)**3+ok*(1+zz)**2+ode*de
                if ee<=0 or not math.isfinite(ee): valid=False; break
                integral+=gw[q]/math.sqrt(ee)
            dc=z/2*integral
            if abs(ok)<1e-10: dist=dc
            elif ok>0: dist=math.sinh(math.sqrt(ok)*dc)/math.sqrt(ok)
            else: dist=math.sin(math.sqrt(-ok)*dc)/math.sqrt(-ok)
            if dist<=0 or not valid: dist=np.nan
            dm[i,j]=dist
            de=(1+z)**(3*(1+w0+wa))*math.exp(-3*wa*z/(1+z))
            ee=ore*(1+z)**4+om*(1+z)**3+ok*(1+z)**2+ode*de
            dh[i,j]=1/math.sqrt(ee) if ee>0 and valid else np.nan
    return dm,ages,dh

def log_cdf_difference(upper, lower):
    """Stable log(Phi(upper)-Phi(lower)), including both tails."""
    upper=np.asarray(upper); lower=np.asarray(lower)
    pos=lower>0
    a=np.where(pos,log_ndtr(-lower),log_ndtr(upper))
    b=np.where(pos,log_ndtr(-upper),log_ndtr(lower))
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        return a+np.log(-np.expm1(np.minimum(b-a,0.)))

@njit(cache=True,parallel=True)
def formation_times_batch(theta,zformation,ax,aw,radiation=True):
    """Cosmic time of formation, consistently evaluated in each sampled cosmology."""
    result=np.empty((len(theta),len(zformation)))
    for i in prange(len(theta)):
        om,h,ok,w0,wa=theta[i]
        ore=OMEGA_R_H2/(h/100.)**2 if radiation else 0.
        ode=1.-om-ok-ore
        for j in range(len(zformation)):
            xmax=1/math.sqrt(1+zformation[j]);integral=0.;valid=ode>0
            for q in range(len(ax)):
                x=ax[q]*xmax;a=x*x
                de=a**(-3*(1+w0+wa))*math.exp(-3*wa*(1-a))
                val=ore+om*a+ok*a*a+ode*a**4*de
                if val<=0 or not math.isfinite(val):valid=False;break
                integral+=aw[q]*2*x**3/math.sqrt(val)*xmax
            result[i,j]=GYR_KM_S_MPC/h*integral if valid else np.nan
    return result

def age_loglike(t, config):
    mode=config.get('mode','gaussian')
    if mode=='stellar_fixed_delay':
        # A sensitivity slice in a *stellar*-age likelihood. The delay is
        # counted once; it is never added to an already cosmic-age summary.
        delay=float(config['fixed_delay_gyr'])
        if delay<0 or not math.isfinite(delay):raise ValueError('Invalid fixed formation delay')
        return -.5*((t-delay-config['stellar_mu'])/config['stellar_sigma'])**2
    mu=config.get('mu',13.57); sigma=config.get('sigma',float(np.hypot(.15,.23)))
    if mode=='none': return np.zeros_like(t)
    if mode=='gaussian': return -.5*((t-mu)/sigma)**2
    if mode=='lower_bound': return log_ndtr((t-mu)/sigma)
    lo,hi=config.get('formation_range',[.1,1.])
    hd_sigma=np.hypot(.4,config.get('hd_extra_systematic',.5))
    delta=t-14.2
    if mode=='joint':
        gcvar=sigma*sigma
        covariance=config.get('rho',0.)*.23*config.get('hd_extra_systematic',.5)
        delta=delta-covariance/gcvar*(t-mu)
        hd_sigma=np.sqrt(hd_sigma**2-covariance**2/gcvar)
        gc=-.5*((t-mu)/sigma)**2
    elif mode=='hd': gc=np.zeros_like(t)
    else: raise ValueError(f'Unknown age model {mode}')
    return gc+log_cdf_difference((delta-lo)/hd_sigma,(delta-hi)/hd_sigma)-np.log(hi-lo)

class Data:
    def __init__(self, data_dir=None, order=32, quadrature=64, age_quadrature=160,
                 use_zhel=True):
        directory=Path(data_dir) if data_dir else ROOT/'code'
        files=['Pantheon+SH0ES.dat','Pantheon+SH0ES_STAT+SYS.cov',
               'desi_gaussian_bao_ALL_GCcomb_mean.txt','desi_gaussian_bao_ALL_GCcomb_cov.txt']
        self.provenance={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files}
        # Pin the actual released DR2 inputs; mislabeled/reordered data fail loudly.
        expected={'desi_gaussian_bao_ALL_GCcomb_mean.txt':'9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585',
          'desi_gaussian_bao_ALL_GCcomb_cov.txt':'252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509'}
        for name,value in expected.items():
            if self.provenance[name]!=value: raise ValueError(f'DESI DR2 hash mismatch: {name}')
        pan=pd.read_csv(directory/files[0],sep=r'\s+')
        mask=(pan.IS_CALIBRATOR==0)&(pan.zHD>.01)
        idx=np.flatnonzero(mask)
        with (directory/files[1]).open() as handle: n=int(handle.readline())
        raw=np.loadtxt(directory/files[1],skiprows=1)
        if n!=len(pan) or raw.size!=n*n: raise ValueError('SN covariance dimension mismatch')
        self.cov=raw.reshape(n,n)[np.ix_(idx,idx)]
        if not np.allclose(self.cov,self.cov.T,rtol=0,atol=1e-12): raise ValueError('Asymmetric covariance')
        self.cf=cho_factor(self.cov,lower=True)
        self.z=pan.loc[mask,'zHD'].to_numpy()
        self.zhel=pan.loc[mask,'zHEL'].to_numpy() if use_zhel else self.z.copy()
        self.mb=pan.loc[mask,'m_b_corr'].to_numpy()
        self.indices=idx
        self.order=order
        xmax=np.log1p(self.z.max())
        angle=np.pi*(np.arange(order)+.5)/order
        self.znodes=np.expm1((np.cos(angle)+1)*xmax/2)
        self.transform=2/order*np.cos(np.arange(order)[:,None]*angle[None,:])
        self.transform[0]/=2
        self.vander=np.polynomial.chebyshev.chebvander(2*np.log1p(self.z)/xmax-1,order-1)
        self.r=self.mb-25-5*np.log10(C_LIGHT)-5*np.log10(self.z)-5*np.log10(1+self.zhel)
        self._compress()
        table=pd.read_csv(directory/files[2],sep=r'\s+',comment='#',names=['z','value','type'])
        if len(table)!=13: raise ValueError('Expected the 13-component DESI DR2 vector')
        mapping={'DM_over_rs':0,'DH_over_rs':1,'DV_over_rs':2}
        self.bao_type=np.array([mapping[t] for t in table.type])
        self.bao_z=table.z.to_numpy(); self.bao_y=table.value.to_numpy()
        self.bao_cov=np.loadtxt(directory/files[3])
        self.bao_cf=cho_factor(self.bao_cov,lower=True)
        self.bao_icov=cho_solve(self.bao_cf,np.eye(13))
        self.gx,self.gw=np.polynomial.legendre.leggauss(quadrature)
        x,w=np.polynomial.legendre.leggauss(age_quadrature)
        self.ax=(x+1)/2; self.aw=w/2
        self.zall=np.r_[self.znodes,self.bao_z]

    def _compress(self):
        one=np.ones(len(self.z))
        iv=cho_solve(self.cf,self.vander)
        ir=cho_solve(self.cf,self.r)
        io=cho_solve(self.cf,one)
        self.cc=float(one@io)
        self.br=float(one@ir)
        self.bv=one@iv
        self.rr=float(self.r@ir)
        self.vr=self.vander.T@ir
        self.vv=self.vander.T@iv
        # Exact Gaussian marginalization of the intercept in coefficient space.
        self.profile_rr=self.rr-self.br**2/self.cc
        self.profile_vr=self.vr-self.bv*self.br/self.cc
        self.profile_vv=self.vv-np.outer(self.bv,self.bv)/self.cc

    def evaluate(self, physical, radiation=True):
        dm,age,dh=integrals_batch(np.ascontiguousarray(physical),self.zall,
                                self.gx,self.gw,self.ax,self.aw,radiation)
        with np.errstate(invalid='ignore',divide='ignore'):
            g=5*np.log10(dm[:,:self.order]/self.znodes[None,:])
        coeff=g@self.transform.T
        chi=self.profile_rr-2*(coeff@self.profile_vr)+np.einsum('bi,ij,bj->b',coeff,self.profile_vv,coeff)
        alpha=(self.br-coeff@self.bv)/self.cc
        return chi,alpha,age,dm[:,self.order:],dh[:,self.order:],coeff

class Likelihood:
    def __init__(self,data,config):
        self.data=data; self.config=config
        self.model=config.get('model','flat'); self.bao=config.get('bao',False)
        self.names=['Om','H0']+(['rd'] if self.bao else [])
        if self.model=='curved': self.names+=['Ok']
        elif self.model=='cpl': self.names+=['w0','wa']
        self.bounds={'Om':config.get('Om_bounds',[.1,.6]),'H0':config.get('H0_bounds',[45.,100.]),
                     'rd':config.get('rd_bounds',[80.,220.]),'Ok':[-.5,.5], 'w0':[-2.,0.], 'wa':[-3.,2.]}
        if config['age'].get('mode')=='stellar_zform':
            x,w=np.polynomial.legendre.leggauss(32)
            lo,hi=config['age'].get('zform_range',[11.,30.])
            if config['age'].get('zform_prior','uniform_z')=='uniform_log1pz':
                self.zform=np.expm1((x+1)/2*(np.log1p(hi)-np.log1p(lo))+np.log1p(lo))
            else:self.zform=(x+1)/2*(hi-lo)+lo
            self.zform_logweights=np.log(w/2)

    def physical(self,theta):
        theta=np.atleast_2d(theta)
        p=np.zeros((len(theta),5));p[:,0]=theta[:,0];p[:,1]=theta[:,1];p[:,3]=-1
        if self.model=='curved': p[:,2]=theta[:,self.names.index('Ok')]
        if self.model=='cpl':
            p[:,3]=theta[:,self.names.index('w0')];p[:,4]=theta[:,self.names.index('wa')]
        return p

    def components(self,theta):
        theta=np.atleast_2d(theta)
        p=self.physical(theta)
        sn,alpha,age,dm,dh,coeff=self.data.evaluate(p,self.config.get('radiation',True))
        if self.config['age'].get('mode')=='stellar_zform':
            formation=formation_times_batch(p,self.zform,self.data.ax,self.data.aw,self.config.get('radiation',True))
            mu=self.config['age'].get('stellar_mu',13.39)
            sig=self.config['age'].get('stellar_sigma',float(np.hypot(.10,.23)))
            values=-.5*((age[:,None]-formation-mu)/sig)**2+self.zform_logweights[None,:]
            age_chi=-2*logsumexp(values,axis=1)
        else:age_chi=-2*age_loglike(age,self.config['age'])
        bao_chi=np.zeros(len(p))
        if self.bao:
            scale=C_LIGHT/p[:,1,None]
            dm=dm*scale;dh=dh*scale
            dv=np.cbrt(self.data.bao_z[None,:]*dm*dm*dh)
            pred=np.where(self.data.bao_type[None,:]==0,dm,
                          np.where(self.data.bao_type[None,:]==1,dh,dv))
            pred/=theta[:,self.names.index('rd'),None]
            resid=self.data.bao_y[None,:]-pred
            bao_chi=np.einsum('bi,ij,bj->b',resid,self.data.bao_icov,resid)
        return sn,bao_chi,age_chi,alpha,age,coeff

    def __call__(self,theta):
        theta=np.atleast_2d(theta)
        mask=np.ones(len(theta),bool)
        for j,name in enumerate(self.names):
            lo,hi=self.bounds[name];mask&=(theta[:,j]>lo)&(theta[:,j]<hi)
        if self.model=='cpl' and self.config.get('early_subdominant',False):
            mask&=(theta[:,self.names.index('w0')]+theta[:,self.names.index('wa')]<0)
        out=np.full(len(theta),-np.inf)
        if mask.any():
            a,b,c,_,_,_=self.components(theta[mask])
            val=-.5*(a+b+c)
            out[mask]=np.where(np.isfinite(val),val,-np.inf)
        return out

    def derived(self,theta,rng):
        a,b,c,alpha,age,_=self.components(theta)
        mb=alpha+5*np.log10(theta[:,1])+rng.normal(0,1/np.sqrt(self.data.cc),len(theta))
        return np.c_[theta,mb,age],self.names+['MB','t0']
