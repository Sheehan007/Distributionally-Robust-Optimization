"""Validation-gate audit and conditional allocation Monte Carlo sensitivity."""
import json
import numpy as np
import pandas as pd
import torch
from forecast import ROOT,P,FEATURES,VolatilityNet,data
from portfolio import scenario_catalog,ConvexAllocator
from tail_allocator import TailAllocator

def gates():
    rows=[]
    for ticker in P['tickers']:
        d=data(ticker);ids=np.arange(P['sequence_length']-1,len(d));raw=d[FEATURES].to_numpy(dtype='float32')
        x=np.stack([raw[i-19:i+1] for i in ids]);available=d.target_date.iloc[ids]
        target=d.target.iloc[ids].to_numpy();base=d.lv.iloc[ids].to_numpy()
        for year in P['evaluation_years']:
            mask=((available>=pd.Timestamp(year-1,1,1))&(available<pd.Timestamp(year,1,1))).to_numpy()
            predictions=[]
            for seed in P['seeds']:
                ck=torch.load(ROOT/'results'/f'{ticker}_residual_{year}_{seed}.pt',map_location='cpu',weights_only=True)
                model=VolatilityNet(True);model.load_state_dict(ck['state_dict']);model.eval()
                z=torch.from_numpy(((x[mask]-np.array(ck['feature_mean']))/np.array(ck['feature_std'])).astype('float32'))
                with torch.no_grad():predictions.append(model(z).numpy()*ck['target_std'])
            correction=np.mean(predictions,axis=0);delta=target[mask]-base[mask]
            gate=min(P['validation_gate_grid'],key=lambda g:np.mean((g*correction-delta)**2))
            baseline=float(np.mean(delta**2));selected=float(np.mean((gate*correction-delta)**2))
            assert selected<=baseline+1e-12
            saved=pd.read_csv(ROOT/'results'/f'{ticker}_forecasts.csv');saved=saved[saved.year==year]
            assert float(saved.gate.iloc[0])==gate
            rows.append(dict(ticker=ticker,year=year,gate=gate,baseline_validation_mse=baseline,selected_validation_mse=selected,
                ratio=selected/baseline,train_max_target=str(d.target_date[d.target_date<pd.Timestamp(year-1,1,1)].max().date()),
                validation_max_target=str(available[mask].max().date())))
    pd.DataFrame(rows).to_csv(ROOT/'results'/'gate_audit.csv',index=False)

def monte_carlo():
    t=pd.read_csv(ROOT/'results'/'paired_20000_seed0_trades.csv');t=t[t.strategy=='Residual Heston'].set_index('date')
    baselines={};rows=[];solvers={}
    for n in [2000,20000]:
        for offset in [0,10000,20000]:
            frames,dates,opens,catalog=scenario_catalog(n,offset,only_year=2025,save=False,engines=['residual'])
            solver=TailAllocator(n)
            for s in catalog:
                i=s['index'];prior_date=str(dates[i-5].date())
                pw=t.loc[prior_date,[f'w_{x}' for x in P['tickers']]].to_numpy(dtype=float)
                realized=opens[i+1]/opens[i-4]-1
                previous=pw*(1+realized)/(1+pw@realized)
                w,_=solver.solve(s['scenarios']['residual'],s['means'],previous,0.)
                a,_=solver.solve(s['scenarios']['residual'],s['means'],previous,s['epsilon'])
                if offset==0:baselines[s['date']]=(w,a)
                bw,ba=baselines[s['date']]
                rows.append(dict(date=s['date'],scenarios=n,seed_offset=offset,
                    nonrobust_l1_from_primary=float(np.abs(w-bw).sum()),adaptive_l1_from_primary=float(np.abs(a-ba).sum()),
                    robustness_tilt_l1=float(np.abs(a-w).sum()),effective_nonrobust=float(1/(w@w)),effective_adaptive=float(1/(a@a))))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'results'/'monte_carlo_allocations.csv',index=False)
    frame.groupby(['scenarios','seed_offset']).agg(mean_nominal_distance=('nonrobust_l1_from_primary','mean'),
        mean_adaptive_distance=('adaptive_l1_from_primary','mean'),p95_adaptive_distance=('adaptive_l1_from_primary',lambda x:x.quantile(.95)),
        mean_robustness_tilt=('robustness_tilt_l1','mean')).reset_index().to_csv(ROOT/'results'/'monte_carlo_summary.csv',index=False)

if __name__=='__main__':
    gates();print('Gate audit complete',flush=True);monte_carlo();print('Monte Carlo allocation checks complete',flush=True)
