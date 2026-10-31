"""Joint four-ETF scenarios and convex multi-asset DRO with curved risk."""
from pathlib import Path
import json,math,warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from forecast import ROOT,P,data

warnings.filterwarnings('ignore',message='.*not DPP.*')
TICKERS=P['tickers'];D=len(TICKERS);Q=1-P['alpha']

def es(a):
    a=np.sort(np.asarray(a,dtype=float))[::-1];n=Q*len(a);k=int(np.floor(n+1e-10));f=n-k
    return float((a[:k].sum()+(f*a[k] if f>1e-9 else 0))/n)

class ConvexAllocator:
    def __init__(self,n,cash=False,gamma=None):
        self.n=n;self.gamma=P['gamma'] if gamma is None else gamma;self.cash=cash
        self.w=cp.Variable(D);self.eta=cp.Variable();self.z=cp.Variable(n,nonneg=True)
        self.turn=cp.Variable(D,nonneg=True)
        self.scen=cp.Parameter((n,D));self.mean=cp.Parameter(D);self.root=cp.Parameter((D,D))
        self.prev=cp.Parameter(D);self.eps=cp.Parameter(nonneg=True);self.cost=cp.Parameter(nonneg=True)
        rf=(1+P['cash_annual'])**(P['horizon_days']/252)-1
        income=(1-cp.sum(self.w))*rf if cash else 0
        self.constraints=[self.w>=0,self.z>=-self.scen@self.w-income-self.eta,
            self.turn>=self.w-self.prev,self.turn>=self.prev-self.w]
        self.constraints.append(cp.sum(self.w)<=1 if cash else cp.sum(self.w)==1)
        tail=self.eta+cp.sum(self.z)/(n*Q)+self.eps*cp.norm(self.w,2)/Q
        objective=self.mean@self.w+income-self.gamma*tail-.5*cp.sum_squares(self.root@self.w)-self.cost*cp.sum(self.turn)
        self.problem=cp.Problem(cp.Maximize(objective),self.constraints)
        assert self.problem.is_dcp() and self.problem.is_dpp()

    def solve(self,scenarios,mean,previous,epsilon=0.,cost=None,lam=None):
        cost=P['transaction_cost'] if cost is None else cost;lam=P['quadratic_risk_lambda'] if lam is None else lam
        cov=np.atleast_2d(np.cov(scenarios,rowvar=False,ddof=0))
        cov+=.05*np.trace(cov)/D*np.eye(D)
        self.scen.value=np.asarray(scenarios);self.mean.value=np.asarray(mean);self.prev.value=previous
        self.root.value=np.linalg.cholesky(lam*cov).T if lam>0 else np.zeros((D,D))
        self.eps.value=max(float(epsilon),0);self.cost.value=float(cost)
        self.problem.solve(solver='CLARABEL',warm_start=True,max_iter=150,tol_gap_abs=1e-8,tol_feas=1e-8,tol_gap_rel=1e-8)
        if self.problem.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(self.problem.status)
        w=np.asarray(self.w.value).reshape(-1)
        violation=max(float(np.maximum(-w,0).max()),max(0,float(w.sum()-1)) if self.cash else abs(float(w.sum()-1)))
        assert violation<2e-6,(self.problem.status,w)
        w=np.maximum(w,0)
        if not self.cash:w/=w.sum()
        elif w.sum()>1:w/=w.sum()
        return w,dict(status=self.problem.status,primal_violation=violation,iterations=int(self.problem.solver_stats.num_iters),
            objective=float(self.problem.value),risk_quadratic=float(.5*lam*w@cov@w))

def correlated_simulation(v0,params,mu,z,u,gaussian=False):
    n=z.shape[1];dt=1/(252*P['substeps_per_day']);v=np.broadcast_to(v0,(n,D)).copy();logr=np.zeros((n,D))
    for j in range(z.shape[0]):
        pos=np.maximum(v,1e-8)
        logr+=(mu-.5*pos)*dt+np.sqrt(pos*dt)*z[j]
        if not gaussian:
            k,theta,xi,rho=params.T
            v+=k*(theta-pos)*dt+xi*np.sqrt(pos*dt)*(rho*z[j]+np.sqrt(1-rho*rho)*u[j])
    return np.expm1(logr)

def scenario_catalog(scenario_count=None,seed_offset=0,only_year=None,save=True,engines=None,signal_positions=None):
    frames={t:data(t) for t in TICKERS}
    dates=frames[TICKERS[0]].index
    for t in TICKERS[1:]:dates=dates.intersection(frames[t].index)
    frames={t:d.loc[dates] for t,d in frames.items()}
    forecasts={t:pd.read_csv(ROOT/'results'/f'{t}_forecasts.csv').set_index('date') for t in TICKERS}
    states={t:pd.read_csv(ROOT/'results'/f'{t}_filtered_states.csv').set_index('date') for t in TICKERS}
    fits={t:pd.read_csv(ROOT/'results'/f'{t}_calibration.csv').set_index('year') for t in TICKERS}
    close_returns=np.column_stack([frames[t].r for t in TICKERS]);opens=np.column_stack([frames[t].Open for t in TICKERS])
    first=int(np.flatnonzero(dates>=pd.Timestamp(2019,1,1))[0]);h=P['horizon_days'];n=P['scenario_count'] if scenario_count is None else scenario_count
    catalog=[];summaries=[]
    for signal_position,i in enumerate(range(first,len(dates)-h-1,h)):
        if signal_positions is not None and signal_position not in signal_positions:continue
        date=str(dates[i].date());year=int(dates[i].year)
        if only_year is not None and year!=only_year:continue
        corr=np.corrcoef(close_returns[max(0,i-251):i+1],rowvar=False);corr=.9*corr+.1*np.eye(D)
        chol=np.linalg.cholesky(corr);rng=np.random.default_rng(81000+i+seed_offset)
        zz=rng.standard_normal((h*P['substeps_per_day'],n//2,D));uu=rng.standard_normal(zz.shape)
        zz=np.concatenate([zz,-zz],axis=1)@chol.T;uu=np.concatenate([uu,-uu],axis=1)
        params=np.array([[fits[t].loc[year,x] for x in ['kappa','theta','xi','rho']] for t in TICKERS])
        mu=np.array([fits[t].loc[year,'mu'] for t in TICKERS]);means=np.expm1(mu*h/252)
        residual=np.array([np.exp(forecasts[t].loc[date,'residual']) for t in TICKERS]).clip(.01,2)
        persistence=np.array([np.exp(forecasts[t].loc[date,'persistence']) for t in TICKERS]).clip(.01,2)
        filtered=np.array([states[t].loc[date,'filtered_variance'] for t in TICKERS])
        verror=np.array([forecasts[t].loc[date,'validation_rmse'] for t in TICKERS])
        epsilon=.1*np.linalg.norm(residual*verror)*np.sqrt(h/252)
        historical=np.expm1(pd.DataFrame(close_returns[max(0,i-251):i+1]).rolling(h).sum().dropna().to_numpy())
        builders={
            'historical':lambda:historical,
            'filtered':lambda:correlated_simulation(filtered,params,mu,zz,uu),
            'persistence':lambda:correlated_simulation(persistence**2,params,mu,zz,uu),
            'residual':lambda:correlated_simulation(residual**2,params,mu,zz,uu),
            'fixed':lambda:correlated_simulation(residual**2,np.tile(P['original_fixed_parameters'],(D,1)),mu,zz,uu),
            'gaussian':lambda:correlated_simulation(residual**2,params,mu,zz,uu,gaussian=True)}
        scenarios={name:make() for name,make in builders.items() if engines is None or name in engines}
        item=dict(date=date,year=year,index=i,scenarios=scenarios,means=means,epsilon=epsilon,
            risky_realized=opens[i+1+h]/opens[i+1]-1,correlation=corr)
        catalog.append(item)
        for engine,sc in scenarios.items():
            for j,ticker in enumerate(TICKERS):
                summaries.append(dict(date=date,year=year,engine=engine,ticker=ticker,
                    mean=float(sc[:,j].mean()),analytic_mean=float(means[j]),es95=es(-sc[:,j]),q05=float(np.quantile(sc[:,j],.05)),
                    realized=float(item['risky_realized'][j]),epsilon=epsilon))
    if save:pd.DataFrame(summaries).to_csv(ROOT/'results'/'scenario_summaries.csv',index=False)
    return frames,dates,opens,catalog

STRATEGIES={
    'Buy and hold':('buyhold',0.),'Equal weight':('equal',0.),'Minimum variance':('minvar',0.),
    'Historical CVaR':('historical',0.),'Filtered Heston':('filtered',0.),
    'Persistence Heston':('persistence',0.),'Residual Heston':('residual',0.),
    'Fixed parameter ablation':('fixed',0.),'Residual Gaussian':('gaussian',0.),
    'Fixed DRO':('residual','fixed'),'Adaptive DRO':('residual','adaptive')}

def backtest(opens,dates,catalog,cash=False,strategies=None,epsilon=None,cost=None,lam=None,allocator=ConvexAllocator):
    cost=P['transaction_cost'] if cost is None else cost
    selected=STRATEGIES if strategies is None else {s:STRATEGIES[s] for s in strategies}
    solvers={};daily=[];trades=[];rf=(1+P['cash_annual'])**(1/252)-1
    for strategy,(engine,epsmode) in selected.items():
        risky=np.zeros(D);cash_holding=1.
        for k,s in enumerate(catalog):
            wealth=risky.sum()+cash_holding;previous=risky/wealth
            sc=s['scenarios']['historical' if engine=='minvar' else engine] if engine not in ['equal','buyhold'] else s['scenarios']['residual']
            eps=0.;diagnostic=dict(status='constant',primal_violation=0,iterations=0,objective=np.nan,risk_quadratic=0.)
            if engine=='equal':w=np.ones(D)/D
            elif engine=='buyhold':w=np.ones(D)/D if k==0 else previous
            else:
                if epsmode=='fixed':eps=P['fixed_epsilon'] if epsilon is None else epsilon
                elif epsmode=='adaptive':eps=s['epsilon']
                gamma=0. if engine=='minvar' else P['gamma']
                key=(len(sc),cash,gamma)
                if key not in solvers:solvers[key]=allocator(len(sc),cash,gamma)
                mean=sc.mean(axis=0) if engine in ['historical','minvar'] else s['means']
                if engine=='minvar':mean=np.zeros(D)
                w,diagnostic=solvers[key].solve(sc,mean,previous,eps,cost,lam)
            turnover=float(np.abs(w-previous).sum());fee=cost*turnover
            new=wealth*(1-fee);risky=new*w;cash_holding=new*(1-w.sum())
            gross=float(w@s['risky_realized']+(1-w.sum())*((1+P['cash_annual'])**(P['horizon_days']/252)-1))
            q05=float(np.quantile(sc@w+(1-w.sum())*((1+P['cash_annual'])**(P['horizon_days']/252)-1),.05))
            trade=dict(strategy=strategy,date=s['date'],year=s['year'],turnover=turnover,epsilon=eps,
                gross_risky_weight=float(w.sum()),max_weight=float(w.max()),active_assets=int(np.sum(w>1e-5)),
                effective_assets=float(w.sum()**2/(w@w)) if w@w>1e-10 else 0.,realized_gross=gross,scenario_q05=q05,**diagnostic)
            for j,t in enumerate(TICKERS):trade[f'w_{t}']=float(w[j])
            trades.append(trade)
            i=s['index']
            for j in range(P['horizon_days']):
                before=wealth if j==0 else risky.sum()+cash_holding
                risky*=opens[i+2+j]/opens[i+1+j];cash_holding*=1+rf;nav=float(risky.sum()+cash_holding)
                daily.append(dict(strategy=strategy,date=str(dates[i+2+j].date()),year=int(dates[i+2+j].year),return_net=nav/before-1,nav=nav))
    return pd.DataFrame(daily),pd.DataFrame(trades)

def metrics(daily,trades):
    output=[];rfd=(1+P['cash_annual'])**(1/252)-1
    for strategy,g in daily.groupby('strategy',sort=False):
        a=g.return_net.to_numpy();nav=np.cumprod(1+a);vol=a.std(ddof=1);t=trades[trades.strategy==strategy]
        output.append(dict(strategy=strategy,n=len(a),cagr=float(nav[-1]**(252/len(a))-1),
            volatility=float(vol*np.sqrt(252)),sharpe=float(np.mean(a-rfd)/vol*np.sqrt(252)) if vol>1e-12 else np.nan,
            es95=es(-a),max_drawdown=float(np.min(nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1)),
            turnover=float(t.turnover.sum()*252/len(a)),mean_gross=float(t.gross_risky_weight.mean()),
            interior_fraction=float((t.active_assets>=2).mean()),mean_effective_assets=float(t.effective_assets.mean()),
            mean_max_weight=float(t.max_weight.mean())))
    return pd.DataFrame(output)

def run():
    from tail_allocator import TailAllocator
    frames,dates,opens,catalog=scenario_catalog()
    dd,tt=backtest(opens,dates,catalog,allocator=TailAllocator)
    dd.to_csv(ROOT/'results'/'portfolio_daily.csv',index=False);tt.to_csv(ROOT/'results'/'portfolio_trades.csv',index=False)
    results=[]
    for phase,mask in [('exploratory',dd.year<2025),('confirmation',dd.year==2025),('all',dd.year>0)]:
        m=metrics(dd[mask],tt[(tt.year<2025) if phase=='exploratory' else (tt.year==2025) if phase=='confirmation' else (tt.year>0)])
        m['phase']=phase;results.append(m)
    pd.concat(results).to_csv(ROOT/'results'/'portfolio_metrics.csv',index=False)
    print('Primary portfolio complete',flush=True)
    secondary=['Persistence Heston','Residual Heston','Fixed DRO','Adaptive DRO']
    cd,ct=backtest(opens,dates,catalog,cash=True,strategies=secondary,allocator=TailAllocator)
    cd.to_csv(ROOT/'results'/'cash_daily.csv',index=False);ct.to_csv(ROOT/'results'/'cash_trades.csv',index=False)
    cm=[]
    for phase,mask in [('exploratory',cd.year<2025),('confirmation',cd.year==2025)]:
        m=metrics(cd[mask],ct[ct.year<2025] if phase=='exploratory' else ct[ct.year==2025]);m['phase']=phase;cm.append(m)
    pd.concat(cm).to_csv(ROOT/'results'/'cash_metrics.csv',index=False)
    print('Cash-enabled companion complete',flush=True)
    sensitivity(opens,dates,catalog,allocator=TailAllocator)
    meta=dict(data_start=str(dates.min().date()),data_end=str(dates.max().date()),
        first_signal=catalog[0]['date'],last_signal=catalog[-1]['date'],rebalances=len(catalog),
        first_return=str(dd.date.min()),last_return=str(dd.date.max()),primary_strategies=len(STRATEGIES))
    (ROOT/'results'/'portfolio_metadata.json').write_text(json.dumps(meta,indent=2))

def sensitivity(opens,dates,catalog,allocator=ConvexAllocator):
    sensitivities=[]
    # Re-run the full path so each setting carries its own historically feasible
    # holdings into 2025. Report 2025 only; never reset primary holdings at Jan 1.
    def confirm_metrics(dd,tt):return metrics(dd[dd.year==2025],tt[tt.year==2025])
    for eps in P['sensitivity_epsilon']:
        sd,st=backtest(opens,dates,catalog,strategies=['Fixed DRO'],epsilon=eps,allocator=allocator)
        m=confirm_metrics(sd,st);m['vary']='epsilon';m['setting']=eps;sensitivities.append(m)
    for lam in P['sensitivity_lambda']:
        sd,st=backtest(opens,dates,catalog,strategies=['Fixed DRO'],lam=lam,allocator=allocator)
        m=confirm_metrics(sd,st);m['vary']='lambda';m['setting']=lam;sensitivities.append(m)
    for fee in P['cost_sensitivity']:
        sd,st=backtest(opens,dates,catalog,strategies=['Residual Heston','Fixed DRO','Adaptive DRO'],cost=fee,allocator=allocator)
        m=confirm_metrics(sd,st);m['vary']='cost';m['setting']=fee;sensitivities.append(m)
    pd.concat(sensitivities).to_csv(ROOT/'results'/'portfolio_sensitivity.csv',index=False)
    print('Portfolio sensitivities complete',flush=True)

if __name__=='__main__':run()
