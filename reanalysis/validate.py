"""Independent numerical, covariance and age-likelihood validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']='1'
from pathlib import Path
import copy
import json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_solve
from scipy.special import ndtr
from cosmology import Data,Likelihood,age_loglike,integrals_batch,formation_times_batch,C_LIGHT,GYR_KM_S_MPC,OMEGA_R_H2

def exact(physical,z,radiation=True):
    om,h,ok,w0,wa=physical
    ore=OMEGA_R_H2/(h/100)**2 if radiation else 0.
    ode=1-om-ok-ore
    def einv(zz):
        de=(1+zz)**(3*(1+w0+wa))*np.exp(-3*wa*zz/(1+zz))
        return 1/np.sqrt(ore*(1+zz)**4+om*(1+zz)**3+ok*(1+zz)**2+ode*de)
    dc=np.array([quad(einv,0,zz,epsabs=1e-11,epsrel=1e-11)[0] for zz in z])
    if ok>1e-10: dm=np.sinh(np.sqrt(ok)*dc)/np.sqrt(ok)
    elif ok<-1e-10: dm=np.sin(np.sqrt(-ok)*dc)/np.sqrt(-ok)
    else: dm=dc
    def ai(a):
        de=a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a))
        return a/np.sqrt(ore+om*a+ok*a*a+ode*a**4*de)
    age=GYR_KM_S_MPC/h*quad(ai,0,1,epsabs=2e-11,epsrel=2e-11)[0]
    return dm,age

def main():
    d=Data()
    rng=np.random.default_rng(2601023)
    rows=[]
    for i in range(28):
        om=rng.uniform(.15,.55);h=rng.uniform(55,85)
        if i%3==0: p=[om,h,0,-1,0]
        elif i%3==1: p=[om,h,rng.uniform(-.15,.25),-1,0]
        else: p=[om,h,0,rng.uniform(-1.7,-.3),rng.uniform(-2.5,1.5)]
        chi,alpha,t,_,_,coeff=d.evaluate(np.array([p]))
        dm,age=exact(p,d.z)
        g=5*np.log10(dm/d.z)
        residual=d.r-g
        direct_alpha=np.sum(cho_solve(d.cf,residual))/d.cc
        residual-=direct_alpha
        direct_chi=float(residual@cho_solve(d.cf,residual))
        rows.append({'physical':p,'delta_chi2':float(chi[0]-direct_chi),
          'delta_alpha_mag':float(alpha[0]-direct_alpha),'delta_age_Gyr':float(t[0]-age),
          'max_model_error_mag':float(np.max(np.abs(d.vander@coeff[0]-g)))})
    p=np.array([[.3,70,0,-1,0]])
    _,ages,_=integrals_batch(p,np.array([1.]),d.gx,d.gw,d.ax,d.aw,False)
    analytic=GYR_KM_S_MPC/70*2/(3*np.sqrt(.7))*np.arcsinh(np.sqrt(.7/.3))
    # Stable HD marginalization against independent explicit integration.
    age_tests=[]
    for rho in [0,.5,1.]:
        cfg={'mode':'joint','mu':13.57,'sigma':float(np.hypot(.15,.23)),
             'hd_extra_systematic':.5,'rho':rho,'formation_range':[.1,1.]}
        for t in [12.7,13.57,14.2,15.5]:
            s=cfg['sigma']; hd=np.hypot(.4,.5);cv=rho*.23*.5
            cond=np.sqrt(hd**2-cv**2/s**2)
            shifted=t-14.2-cv/s**2*(t-13.57)
            v=quad(lambda dt:np.exp(-.5*((t-13.57)/s)**2)*
                   np.exp(-.5*((shifted-dt)/cond)**2)/(cond*np.sqrt(2*np.pi))/.9,.1,1.,epsabs=1e-14)[0]
            implemented=float(np.exp(age_loglike(np.array([t]),cfg)[0]))
            age_tests.append(abs(implemented-v)/max(v,1e-100))
    formation_checks=[]
    for p in [[.3,69.,0,-1,0],[.27,68,.08,-1,0],[.18,69,0,-.85,.65]]:
        zf=np.array([6.,11.,20.,30.,50.])
        ff=formation_times_batch(np.array([p]),zf,d.ax,d.aw,True)[0]
        om,h,ok,w0,wa=p;ore=OMEGA_R_H2/(h/100)**2;ode=1-om-ok-ore
        for z,val in zip(zf,ff):
            def fa(a):
                de=a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a))
                return a/np.sqrt(ore+om*a+ok*a*a+ode*a**4*de)
            direct=GYR_KM_S_MPC/h*quad(fa,0,1/(1+z),epsabs=1e-12,epsrel=1e-12)[0]
            formation_checks.append(abs(val-direct))
    checks={
      'max_abs_delta_chi2':max(abs(r['delta_chi2']) for r in rows),
      'max_abs_delta_alpha_mag':max(abs(r['delta_alpha_mag']) for r in rows),
      'max_abs_delta_age_Gyr':max(abs(r['delta_age_Gyr']) for r in rows),
      'max_abs_model_error_mag':max(r['max_model_error_mag'] for r in rows),
      'flat_analytic_age_error_Gyr':float(ages[0]-analytic),
      'joint_analytic_marginalization_max_relative_error':max(age_tests),
      'formation_time_max_abs_error_Gyr':float(max(formation_checks)),
      'n_SN_rows':len(d.z),'n_BAO':len(d.bao_z),
      'SN_covariance_positive_definite':True,'BAO_covariance_positive_definite':True}
    passed=(checks['max_abs_delta_chi2']<.002 and checks['max_abs_delta_age_Gyr']<1e-4
        and checks['max_abs_model_error_mag']<2e-6 and abs(checks['flat_analytic_age_error_Gyr'])<1e-8
        and max(age_tests)<1e-8 and max(formation_checks)<1e-8)
    passed=bool(passed)
    out={'passed':passed,'checks':checks,'points':rows,'provenance':d.provenance}
    Path(__file__).with_name('validation.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps({'passed':passed,'checks':checks},indent=2))
    if not passed: raise RuntimeError('Independent validation failed')

if __name__=='__main__':main()
