"""Reproducible corrected MCMCs. All outputs are kept in runs/<configuration>/."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.metadata
import json
import sys
import time
import numpy as np
import emcee
from numba import set_num_threads,get_num_threads,threading_layer
from scipy.optimize import minimize,brentq
from cosmology import Data,Likelihood

HERE=Path(__file__).resolve().parent
SIGMA_2025=float(np.hypot(.15,.23))
WORKER_DATA=None
RESULTS=HERE/'runs'

def worker_init(threads,results_dir=None):
    global WORKER_DATA,RESULTS
    set_num_threads(threads)
    if results_dir is not None:RESULTS=Path(results_dir)
    WORKER_DATA=Data()

def worker_run(config):
    local=None if config.get('use_zhel') is False or config.get('mock') else WORKER_DATA
    return run_one(config,local)

def configurations():
    gaussian=lambda mu=13.57,sigma=SIGMA_2025:{'mode':'gaussian','mu':mu,'sigma':sigma}
    configs=[]
    def add(name,model='flat',bao=False,age=None,**kw):
        configs.append(dict(name=name,model=model,bao=bao,age=age or gaussian(),**kw))
    add('gc2025_no_bao')
    add('gc2025_bao',bao=True)
    add('gc2025_curvature',model='curved',bao=True)
    add('gc2025_cpl',model='cpl',bao=True)
    add('gc2025_bao_seed2',bao=True,seed=2026102402)
    add('gc2025_cpl_seed2',model='cpl',bao=True,seed=2026102404)
    add('corrected2021_no_bao',age=gaussian(13.5,.27))
    add('corrected2021_bao',bao=True,age=gaussian(13.5,.27))
    add('bridge_duplicate_delay_no_bao',age=gaussian(13.65,float(np.hypot(.27,.05))),
        radiation=False,use_zhel=False,H0_bounds=[50,90])
    add('cpl_restricted_Om',model='cpl',bao=True,Om_bounds=[.25,.6])
    add('cpl_early_subdominant',model='cpl',bao=True,early_subdominant=True)
    for sig in [.4,.5,.7]: add(f'gc2025_sigma_{sig:.2f}',age=gaussian(sigma=sig))
    add('gc2025_bao_sigma_0.70',bao=True,age=gaussian(sigma=.7))
    add('gc_stellar_lower_bound',age={'mode':'lower_bound','mu':13.39,'sigma':float(np.hypot(.1,.23))})
    add('gc_stellar_lower_bound_H0min50',age={'mode':'lower_bound','mu':13.39,'sigma':float(np.hypot(.1,.23))},H0_bounds=[50,100])
    for extra in [0.,.5,1.]:
        add(f'hd_only_extra_{extra:.1f}',age={'mode':'hd','hd_extra_systematic':extra,'formation_range':[.1,1.]})
    for rho in [0.,.5,1.]:
        age=gaussian();age.update(mode='joint',hd_extra_systematic=.5,rho=rho,formation_range=[.1,1.])
        add(f'joint_rho_{rho:.1f}',age=age)
    add('gc2025_bao_mock',bao=True,mock=True)
    stellar={'mode':'stellar_zform','stellar_mu':13.39,'stellar_sigma':float(np.hypot(.1,.23)),
             'zform_range':[11.,30.],'zform_prior':'uniform_z'}
    add('stellar_zform_no_bao',age=stellar.copy())
    add('stellar_zform_bao',bao=True,age=stellar.copy())
    add('stellar_zform_curvature',model='curved',bao=True,age=stellar.copy())
    add('stellar_zform_cpl',model='cpl',bao=True,age=stellar.copy())
    for suffix,updates in [('zmax50',{'zform_range':[11.,50.]}),
                            ('zmin6',{'zform_range':[6.,30.]}),
                            ('logprior',{'zform_prior':'uniform_log1pz'})]:
        age=stellar.copy();age.update(updates)
        add('stellar_zform_bao_'+suffix,bao=True,age=age)
    for sig in [.4,.5,.7]:
        age=stellar.copy();age['stellar_sigma']=sig
        add(f'stellar_zform_sigma_{sig:.2f}',age=age)
    age=stellar.copy();age['stellar_sigma']=.7
    add('stellar_zform_bao_sigma_0.70',bao=True,age=age)
    add('stellar_zform_bao_seed2',bao=True,age=stellar.copy(),seed=2026102501)
    add('stellar_zform_cpl_seed2',model='cpl',bao=True,age=stellar.copy(),seed=2026102503)
    stellar26=stellar.copy()
    stellar26.update(stellar_mu=13.61,stellar_sigma=float(np.hypot(.25,.23)))
    add('gc2026_no_bao',age=stellar26.copy())
    add('gc2026_bao',bao=True,age=stellar26.copy())
    add('gc2026_curvature',model='curved',bao=True,age=stellar26.copy())
    add('gc2026_cpl',model='cpl',bao=True,age=stellar26.copy())
    add('gc2026_cpl_restricted_Om',model='cpl',bao=True,age=stellar26.copy(),Om_bounds=[.25,.6])
    add('gc2026_cpl_early_subdominant',model='cpl',bao=True,age=stellar26.copy(),early_subdominant=True)
    add('gc2026_bao_seed2',bao=True,age=stellar26.copy(),seed=2026102601)
    add('gc2026_cpl_seed2',model='cpl',bao=True,age=stellar26.copy(),seed=2026102603)
    for suffix,updates in [('zmax50',{'zform_range':[11.,50.]}),
                            ('zmin6',{'zform_range':[6.,30.]}),
                            ('logprior',{'zform_prior':'uniform_log1pz'})]:
        age=stellar26.copy();age.update(updates)
        add('gc2026_bao_'+suffix,bao=True,age=age)
    add('gc2026_cosmic_gaussian_bao',bao=True,age=gaussian(13.81,float(np.hypot(.25,.23))))
    age=stellar26.copy();age['stellar_sigma']=.7
    add('gc2026_sigma_0.70',age=age)
    for i,c in enumerate(configs):c.setdefault('seed',2026102300+i)
    return configs

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')

def run_one(config,data=None):
    name=config['name'];output=RESULTS/name;output.mkdir(parents=True,exist_ok=True)
    prefix_chain=None;prefix_logp=None;prefix_acceptance=None
    resume_random_state=None
    previous_wall_seconds=0.
    if (output/'summary.json').exists():
        previous=json.loads((output/'summary.json').read_text())
        if previous['converged']:
            print('EXISTS',name,flush=True);return previous
        write_json(output/f'attempt_{previous["steps"]}_summary.json',previous)
        previous_wall_seconds=previous.get('total_recorded_wall_seconds',previous['wall_seconds'])
        saved=np.load(output/'raw_chain.npz')
        prefix_chain=saved['chain'];prefix_logp=saved['log_prob']
        prefix_acceptance=saved['acceptance_fraction']
        if 'random_state_keys' in saved:
            resume_random_state=(str(saved['random_state_algorithm']),saved['random_state_keys'],
                int(saved['random_state_position']),int(saved['random_state_has_gauss']),
                float(saved['random_state_cached_gaussian']))
        print('RESUME',name,'from',len(prefix_chain),'steps',flush=True)
    data=data or Data(use_zhel=config.get('use_zhel',True))
    seed=config['seed'];rng=np.random.default_rng(seed)
    offset=0 if prefix_chain is None else len(prefix_chain)
    # Older checkpoints lack RNG state: use a separately keyed continuation
    # stream, rather than repeating the initial segment of the original stream.
    mcmc_seed=seed if not offset else int(np.random.SeedSequence([seed,offset,7291]).generate_state(1)[0])
    np.random.seed(mcmc_seed)
    like=Likelihood(data,config)
    truth=None
    if config.get('mock'):
        h=brentq(lambda h:data.evaluate(np.array([[.30,h,0,-1,0]]))[2][0]-13.57,60,80)
        physical=np.array([[.30,h,0,-1,0]])
        _,alpha,t,dm,dh,coeff=data.evaluate(physical)
        model_alpha=-19.4-5*np.log10(h)
        residual_noise=np.linalg.cholesky(data.cov)@rng.normal(size=len(data.z))
        data.r=data.vander@coeff[0]+model_alpha+residual_noise
        data._compress()
        dm*=299792.458/h;dh*=299792.458/h
        dv=np.cbrt(data.bao_z[None,:]*dm**2*dh)
        pred=np.where(data.bao_type[None,:]==0,dm,np.where(data.bao_type[None,:]==1,dh,dv))[0]/147.
        data.bao_y=pred+np.linalg.cholesky(data.bao_cov)@rng.normal(size=13)
        config['age']['mu']=float(13.57+rng.normal(0,config['age']['sigma']))
        truth={'Om':.3,'H0':h,'rd':147.,'MB':-19.4,'t0':13.57}
    center={'Om':.31,'H0':69.,'rd':146.,'Ok':.03,'w0':-.9,'wa':-.2}
    scatter={'Om':.025,'H0':1.5,'rd':3.,'Ok':.04,'w0':.08,'wa':.5}
    if config['age']['mode'] in ['hd','joint']: center['H0']=65.
    if config['age']['mode']=='lower_bound': center['H0']=58.
    nwalkers=48;ndim=len(like.names);burn=2000
    p0=[]
    while len(p0)<nwalkers:
        point=np.array([center[n]+scatter[n]*rng.normal() for n in like.names])
        if np.isfinite(like(point)[0]):p0.append(point)
    p0=np.array(p0)
    sampler=emcee.EnsembleSampler(nwalkers,ndim,like,vectorize=True,
             moves=[(emcee.moves.StretchMove(),.7),(emcee.moves.DEMove(),.3)])
    tic=time.time();old_tau=None;stable=0;converged=False
    state=p0 if prefix_chain is None else emcee.State(prefix_chain[-1],log_prob=prefix_logp[-1],random_state=resume_random_state)
    for end in range(offset+1000,200001,1000):
        state=sampler.run_mcmc(state,1000,progress=False)
        if end<6000: continue
        whole=sampler.get_chain() if prefix_chain is None else np.concatenate([prefix_chain,sampler.get_chain()])
        production=whole[burn:]
        tau=emcee.autocorr.integrated_time(production,tol=0,quiet=True)
        ess=(end-burn)*nwalkers/tau
        change=np.max(np.abs(tau-old_tau)/tau) if old_tau is not None else np.inf
        stable=stable+1 if change<.05 else 0
        old_tau=tau
        progress={'name':name,'steps':end,'burnin':burn,'tau':tau.tolist(),
                  'ESS':ess.tolist(),'stable_checks':stable,'wall_seconds':time.time()-tic}
        write_json(output/'progress.json',progress)
        if end-burn>60*np.max(tau) and min(ess)>4000 and stable>=2:
            converged=True;break
    whole=sampler.get_chain() if prefix_chain is None else np.concatenate([prefix_chain,sampler.get_chain()])
    whole_logp=sampler.get_log_prob() if prefix_logp is None else np.concatenate([prefix_logp,sampler.get_log_prob()])
    acceptance=sampler.acceptance_fraction if prefix_acceptance is None else (prefix_acceptance*offset+sampler.acceptance_fraction*(len(whole)-offset))/len(whole)
    production=whole[burn:]
    flat=production.reshape(-1,ndim)
    logp=whole_logp[burn:].reshape(-1)
    keep=rng.choice(len(flat),min(60000,len(flat)),replace=False)
    parts=[]
    for start in range(0,len(keep),2048):
        derived,names=like.derived(flat[keep[start:start+2048]],rng)
        parts.append(derived)
    posterior=np.vstack(parts)
    if not np.isfinite(posterior).all():raise RuntimeError('Non-finite derived samples')
    q=np.percentile(posterior,[2.5,16,50,84,97.5],axis=0)
    summary={'config':config,'sampled_names':like.names,'names':names,'steps':len(whole),
       'walkers':nwalkers,'burnin':burn,'production_draws':len(flat),'derived_draws':len(posterior),
       'converged':converged,'tau_production':tau.tolist(),'ESS_production':ess.tolist(),
       'acceptance_mean':float(np.mean(acceptance)), 'resumed_from_steps':offset,
       'continuation_rng':{'seed':mcmc_seed,'restored_state':resume_random_state is not None},
       'wall_seconds':time.time()-tic,'truth':truth,
       'total_recorded_wall_seconds':previous_wall_seconds+time.time()-tic,
       'execution':{'pid':os.getpid(),'numba_threads':get_num_threads(),'threading_layer':threading_layer(),
          'BLAS_thread_limit':1,'batch_likelihood':'Numba prange over independent walker cosmologies'},
       'parameters':{n:{'q025':float(q[0,j]),'q16':float(q[1,j]),'median':float(q[2,j]),
         'q84':float(q[3,j]),'q975':float(q[4,j])} for j,n in enumerate(names)}}
    # Optimize the marginalized likelihood, then report all residual components.
    initial=flat[np.argmax(logp)]
    fit=minimize(lambda x:-like(x)[0],initial,method='Nelder-Mead',options={'maxiter':5000,'xatol':1e-7,'fatol':1e-8})
    point=fit.x if np.isfinite(like(fit.x)[0]) and like(fit.x)[0]>=like(initial)[0] else initial
    sn,ba,ag,alpha,age,_=like.components(point)
    summary['bestfit']={'sampled':dict(zip(like.names,point.tolist())), 'MB':float(alpha[0]+5*np.log10(point[1])),
          't0':float(age[0]),'chi2_SN':float(sn[0]),'chi2_BAO':float(ba[0]),
          'minus2logL_age_up_to_constant':float(ag[0]),'optimizer_success':bool(fit.success)}
    summary['H0_reference_difference_sigma']={}
    h=summary['parameters']['H0']; err=(h['q84']-h['q16'])/2
    for label,ref,se in [('Planck2018',67.4,.5),('SH0ES2022',73.04,1.04)]:
        summary['H0_reference_difference_sigma'][label]=float(abs(h['median']-ref)/np.hypot(err,se))
    if truth: summary['mock_truth_in_95_interval']={n:summary['parameters'][n]['q025']<=v<=summary['parameters'][n]['q975'] for n,v in truth.items()}
    summary['provenance']=data.provenance
    summary['software']={n:importlib.metadata.version(n) for n in ['numpy','scipy','numba','pandas','emcee']}
    rs=sampler.random_state
    np.savez_compressed(output/'raw_chain.npz',chain=whole,log_prob=whole_logp,
                         acceptance_fraction=acceptance,names=np.array(like.names),burnin=burn,seed=seed,
                         random_state_algorithm=rs[0],random_state_keys=rs[1],random_state_position=rs[2],
                         random_state_has_gauss=rs[3],random_state_cached_gaussian=rs[4])
    np.savez_compressed(output/'posterior.npz',samples=posterior,names=np.array(names),seed=seed)
    write_json(output/'config.json',config);write_json(output/'summary.json',summary)
    print('FINISHED',name,'steps',summary['steps'],'ESSmin',round(float(min(ess))),
          'H0',h['median'],'converged',converged,'seconds',round(summary['wall_seconds']),flush=True)
    if not converged: raise RuntimeError(f'{name} convergence target not achieved')
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*')
    parser.add_argument('--config-file',type=Path,help='Replay a frozen list of configurations and seeds')
    parser.add_argument('--results-dir',type=Path,default=HERE/'runs',help='Use a separate output directory for an independent repeat')
    parser.add_argument('--workers',type=int,default=min(4,max(1,(os.cpu_count() or 4)//4)))
    parser.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--archive-existing',action='store_true',help='Preserve completed serial pilot runs before a fresh parallel production run')
    parser.add_argument('--tag',default='production',help='Separate progress and execution records for concurrent batches')
    args=parser.parse_args()
    if args.workers<1 or args.threads<1 or args.workers*args.threads>(os.cpu_count() or 1):
        parser.error('Positive worker/thread counts must fit within the logical CPU count')
    if args.archive_existing and args.results_dir.resolve()!=(HERE/'runs').resolve():
        parser.error('--archive-existing applies only to the default results directory')
    os.environ['NUMBA_NUM_THREADS']=str(args.threads)
    cfg=json.loads(args.config_file.read_text(encoding='utf8')) if args.config_file else configurations()
    if args.only:
        missing=set(args.only)-{c['name'] for c in cfg}
        if missing:parser.error('Unknown configuration(s): '+', '.join(sorted(missing)))
        cfg=[c for c in cfg if c['name'] in args.only]
    if args.archive_existing:
        archive=HERE/'serial_pilot_runs';archive.mkdir(exist_ok=True)
        for c in cfg:
            src=(HERE/'runs'/c['name']).resolve();dst=(archive/c['name']).resolve()
            # Verify both absolute paths before moving a non-empty directory.
            if HERE.resolve() not in src.parents or HERE.resolve() not in dst.parents:
                raise RuntimeError('Archive target escaped the analysis workspace')
            if src.exists():
                if dst.exists(): raise RuntimeError(f'Archive already exists: {dst}')
                src.rename(dst)
    reports=[];started=time.time()
    execution={'process_workers':args.workers,'numba_threads_per_process':args.threads,
       'max_numeric_threads':args.workers*args.threads,'logical_cpus':os.cpu_count(),
       'BLAS_threads_per_process':1,'start_method':'spawn (Windows ProcessPoolExecutor)'}
    write_json(HERE/f'parallel_execution_{args.tag}.json',execution)
    pending=[c['name'] for c in cfg]
    progress_path=HERE/('progress.json' if args.tag=='production' else f'progress_{args.tag}.json')
    write_json(progress_path,{'state':'running','pending':pending,'completed':[],'total':len(cfg)})
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(args.threads,str(args.results_dir.resolve()))) as pool:
        futures={pool.submit(worker_run,c):c['name'] for c in cfg}
        for future in as_completed(futures):
            name=futures[future]
            reports.append(future.result());pending.remove(name)
            write_json(progress_path,{'state':'running','pending':pending,
                 'completed':[r['config']['name'] for r in reports],'total':len(cfg),
                 'wall_seconds':time.time()-started})
    write_json(progress_path,{'state':'complete','total':len(cfg),'wall_seconds':time.time()-started})
    write_json(HERE/f'run_index_{args.tag}.json',{r['config']['name']:r['parameters'] for r in reports})

if __name__=='__main__':main()
