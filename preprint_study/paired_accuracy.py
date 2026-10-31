"""Paired 20,000-scenario DRO experiments under three fixed seeds.

The first declared seed is the reporting path. Other seeds quantify simulator
sensitivity; no seed is selected by investment performance.
"""
import numpy as np
import pandas as pd
from portfolio import scenario_catalog,backtest,metrics,ConvexAllocator
from forecast import ROOT,P
from tail_allocator import TailAllocator

def run():
    all_metrics=[];checks=[];reference={};primary_trades=None
    n=P['paired_scenarios'];strategies=['Residual Heston','Fixed DRO','Adaptive DRO']
    for offset in P['paired_seed_offsets']:
        frames,dates,opens,catalog=scenario_catalog(n,offset,save=False,engines=['residual'])
        daily,trades=backtest(opens,dates,catalog,strategies=strategies,allocator=TailAllocator)
        daily.to_csv(ROOT/'results'/f'paired_{n}_seed{offset}_daily.csv',index=False)
        trades.to_csv(ROOT/'results'/f'paired_{n}_seed{offset}_trades.csv',index=False)
        for phase,mask in [('exploratory',daily.year<2025),('confirmation',daily.year==2025)]:
            tm=trades[trades.year<2025] if phase=='exploratory' else trades[trades.year==2025]
            m=metrics(daily[mask],tm);m['phase']=phase;m['seed_offset']=offset;m['scenarios']=n;all_metrics.append(m)
        if offset==P['paired_seed_offsets'][0]:primary_trades=trades[trades.strategy=='Residual Heston'].set_index('date')
        solver=TailAllocator(n)
        for k,s in enumerate(catalog):
            if s['year']!=2025:continue
            prev=catalog[k-1];pw=primary_trades.loc[prev['date'],[f'w_{t}' for t in P['tickers']]].to_numpy(float)
            change=opens[s['index']+1]/opens[prev['index']+1]-1
            previous=pw*(1+change)/(1+pw@change)
            nominal,_=solver.solve(s['scenarios']['residual'],s['means'],previous,0.)
            adaptive,_=solver.solve(s['scenarios']['residual'],s['means'],previous,s['epsilon'])
            if offset==P['paired_seed_offsets'][0]:reference[s['date']]=(nominal,adaptive)
            nr,ar=reference[s['date']]
            checks.append(dict(date=s['date'],scenarios=n,seed_offset=offset,
                nominal_distance=float(np.abs(nominal-nr).sum()),adaptive_distance=float(np.abs(adaptive-ar).sum()),
                robustness_tilt=float(np.abs(adaptive-nominal).sum())))
        print('Paired high-precision backtest complete:',n,offset,flush=True)
    pd.concat(all_metrics).to_csv(ROOT/'results'/'paired_metrics.csv',index=False)
    frame=pd.DataFrame(checks);frame.to_csv(ROOT/'results'/'paired_mc_allocations.csv',index=False)
    summary=frame.groupby(['scenarios','seed_offset']).agg(mean_nominal_distance=('nominal_distance','mean'),
        mean_adaptive_distance=('adaptive_distance','mean'),p95_adaptive_distance=('adaptive_distance',lambda x:x.quantile(.95)),
        mean_robustness_tilt=('robustness_tilt','mean')).reset_index()
    summary.to_csv(ROOT/'results'/'paired_mc_summary.csv',index=False)

if __name__=='__main__':run()
