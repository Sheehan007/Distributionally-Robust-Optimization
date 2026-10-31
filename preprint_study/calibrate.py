"""Bounded simulated likelihood for a daily Euler Heston state-space model.

Returns are observed, variance is latent. The observation-conditioned variance
transition retains price/variance leverage. This is approximate physical-measure
estimation, not exact continuous-time MLE and not option-price calibration.
"""
import json,math,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize,differential_evolution
from numba import njit
from forecast import ROOT,P,data

@njit(cache=True)
def particle_filter(returns,params,mu,vinit,normals,uniforms):
    kappa,theta,xi,rho=params
    days,particles=normals.shape;dt=1/252.;scale=math.sqrt(1-rho*rho)
    state=np.full(particles,vinit);logscore=np.empty(days);means=np.empty(days);ess=np.empty(days)
    lw=np.empty(particles);zs=np.empty(particles);cdf=np.empty(particles);new=np.empty(particles)
    for t in range(days):
        maximum=-1e300
        for j in range(particles):
            var=max(state[j],1e-8);variance=var*dt
            z=(returns[t]-(mu-.5*var)*dt)/math.sqrt(variance)
            zs[j]=z;lw[j]=-.5*(math.log(2*math.pi*variance)+z*z)
            maximum=max(maximum,lw[j])
        total=0.;squares=0.
        for j in range(particles):
            val=math.exp(lw[j]-maximum);total+=val;cdf[j]=total;squares+=val*val
        logscore[t]=maximum+math.log(total)-math.log(particles);ess[t]=total*total/squares
        u=uniforms[t]/particles;parent=0;mean=0.
        for j in range(particles):
            position=(u+j/particles)*total
            while parent<particles-1 and cdf[parent]<position:parent+=1
            vp=max(state[parent],1e-8)
            new[j]=state[parent]+kappa*(theta-vp)*dt+xi*math.sqrt(vp*dt)*(rho*zs[parent]+scale*normals[t,j])
            mean+=max(new[j],1e-8)
        means[t]=mean/particles
        state[:]=new
    return logscore,means,ess

BOUNDS=[(np.log(.2),np.log(30)),(np.log(.001),np.log(.5)),(.05,.98),(-.95,.95)]

def decode(q):
    k=np.exp(q[0]);theta=np.exp(q[1]);return np.array([k,theta,q[2]*np.sqrt(2*k*theta),q[3]])

def draws(n,particles,seed):
    rng=np.random.default_rng(seed)
    return rng.standard_normal((n,particles)),rng.random(n)

def fit(returns,proxy,seed,initial=None):
    warm=P['calibration_warmup'];r=np.asarray(returns,dtype=float)
    sample_var=max(float(np.var(r[warm:])*252),.0011)
    mu=float(np.mean(r[warm:])*252+.5*sample_var)
    vinit=max(float(np.var(r[:warm])*252),1e-4)
    baseline=np.array(P['original_fixed_parameters'],dtype=float)
    guess=np.array([np.log(3),np.log(np.clip(sample_var,.0011,.49)),.70,-.40])
    vv=np.asarray(proxy,dtype=float)**2
    b=float(np.corrcoef(vv[:-1],vv[1:])[0,1]);k=float(np.clip(-252*np.log(max(b,.01)),.21,29))
    second=np.array([np.log(k),guess[1],.85,np.clip(np.corrcoef(r[1:],np.diff(vv))[0,1],-.9,.9)])
    fixed_q=np.array([np.log(baseline[0]),np.log(baseline[1]),baseline[2]/np.sqrt(2*baseline[0]*baseline[1]),baseline[3]])
    candidates=[guess,second,fixed_q];optim=[];t0=time.time()
    if initial is not None:candidates.append(np.array([np.log(initial['kappa']),np.log(initial['theta']),initial['beta'],initial['rho']]))
    # A small population search supplies a start without Powell's bracket lattice.
    zg,ug=draws(len(r),128,seed+40)
    global_objective=lambda q:-float(particle_filter(r,decode(q),mu,vinit,zg,ug)[0][warm:].sum())
    glob=differential_evolution(global_objective,BOUNDS,seed=seed,popsize=6,maxiter=12,polish=False,tol=1e-5)
    candidates.append(glob.x)
    optim.append(dict(stage='population_start',evaluations=int(glob.nfev),success=bool(glob.success),message=str(glob.message)))
    for stage,particles,budget,xtol,offset in [
        ('coarse',P['particles_fit'],P['calibration_max_evaluations_coarse'],P['calibration_xtol_coarse'],0),
        ('refine',P['particles_refine'],P['calibration_max_evaluations_refine'],P['calibration_xtol_refine'],10)]:
        randoms=[draws(len(r),particles,seed+offset+j) for j in range(P['likelihood_objective_seeds'])]
        objective=lambda q:-float(np.mean([particle_filter(r,decode(q),mu,vinit,z,u)[0][warm:].sum() for z,u in randoms]))
        best=min(candidates,key=objective)
        res=minimize(objective,best,method='Powell',bounds=BOUNDS,
            options=dict(maxfev=budget,xtol=xtol,ftol=P['calibration_ftol']))
        candidates.append(res.x)
        optim.append(dict(stage=stage,particles=particles,objective_seeds=len(randoms),xtol=xtol,
            evaluations=int(res.nfev),success=bool(res.success),message=str(res.message)))
    checkdraws=[draws(len(r),P['particles_check'],seed+101+j) for j in range(P['likelihood_check_seeds'])]
    score=lambda params:np.array([particle_filter(r,params,mu,vinit,zz,uu)[0][warm:].sum() for zz,uu in checkdraws])
    candidate_scores=[score(decode(q)) for q in candidates]
    selected=int(np.argmax([x.mean() for x in candidate_scores]));q=candidates[selected];params=decode(q)
    fitted_scores=candidate_scores[selected];fixed_scores=score(baseline)
    boundary=any(min((q[i]-lo)/(hi-lo),(hi-q[i])/(hi-lo))<.01 for i,(lo,hi) in enumerate(BOUNDS))
    return dict(kappa=float(params[0]),theta=float(params[1]),xi=float(params[2]),rho=float(params[3]),mu=mu,
        beta=float(q[2]),feller_gap=float(2*params[0]*params[1]-params[2]**2),
        loglik_fit=float(fitted_scores.mean()),loglik_fixed=float(fixed_scores.mean()),
        loglik_fit_seed_sd=float(fitted_scores.std()),loglik_fixed_seed_sd=float(fixed_scores.std()),
        warmup=warm,observations=len(r)-warm,initial_variance=vinit,boundary=boundary,
        optimizer=optim,selected_candidate=selected,selected_q=q.tolist(),candidate_scores=[s.tolist() for s in candidate_scores],
        check_particles=P['particles_check'],check_seeds=P['likelihood_check_seeds'],elapsed_seconds=time.time()-t0)

def calibrate_asset(ticker):
    d=data(ticker);records=[];states=[]
    for year in P['evaluation_years']:
        path=ROOT/'results'/f'{ticker}_calibration_{year}.json'
        hist=d[d.index<pd.Timestamp(year,1,1)].tail(P['calibration_window']+P['calibration_warmup'])
        if path.exists():fitrec=json.loads(path.read_text())
        else:
            starts=json.loads((ROOT/'calibration_starts.json').read_text())
            fitrec=fit(hist.r.to_numpy(),hist.rv.to_numpy(),1000+year,starts.get(f'{ticker}:{year}'))
            fitrec.update(ticker=ticker,year=year,fit_start=str(hist.index.min().date()),fit_end=str(hist.index.max().date()))
            path.write_text(json.dumps(fitrec,indent=2))
        params=np.array([fitrec[x] for x in ['kappa','theta','xi','rho']])
        test=d[d.index.year==year];combined=pd.concat([hist,test]);z,u=draws(len(combined),P['particles_check'],8000+year)
        ll,var,ess=particle_filter(combined.r.to_numpy(),params,fitrec['mu'],fitrec['initial_variance'],z,u)
        ll0,var0,ess0=particle_filter(combined.r.to_numpy(),np.array(P['original_fixed_parameters']),fitrec['mu'],fitrec['initial_variance'],z,u)
        fitrec['test_logscore_mean']=float(ll[len(hist):].mean());fitrec['fixed_test_logscore_mean']=float(ll0[len(hist):].mean())
        fitrec['test_ess_fraction']=float(ess[len(hist):].mean()/P['particles_check'])
        path.write_text(json.dumps(fitrec,indent=2))
        flat={k:v for k,v in fitrec.items() if not isinstance(v,(list,dict))};records.append(flat)
        for j,date in enumerate(test.index):states.append(dict(date=str(date.date()),year=year,filtered_variance=float(var[len(hist)+j]),
            fixed_filtered_variance=float(var0[len(hist)+j]),logscore=float(ll[len(hist)+j]),fixed_logscore=float(ll0[len(hist)+j]),ess_fraction=float(ess[len(hist)+j]/P['particles_check'])))
        print(ticker,year,'fitted',','.join(f'{v:.4g}' for v in params),'LL',f"{fitrec['loglik_fit']-fitrec['loglik_fixed']:.2f}",flush=True)
    pd.DataFrame(records).to_csv(ROOT/'results'/f'{ticker}_calibration.csv',index=False)
    pd.DataFrame(states).to_csv(ROOT/'results'/f'{ticker}_filtered_states.csv',index=False)

if __name__=='__main__':
    for ticker in P['tickers']:calibrate_asset(ticker)
