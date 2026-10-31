"""Summarize paired high-scenario paths and numerical diagnostics."""
import json
import numpy as np
import pandas as pd
from forecast import ROOT,P
from portfolio import es
from analyze import bootstrap_indices,batch_es

def main():
    comparisons=[];solver=[]
    for offset in P['paired_seed_offsets']:
        daily=pd.read_csv(ROOT/'results'/f'paired_{P["paired_scenarios"]}_seed{offset}_daily.csv')
        trades=pd.read_csv(ROOT/'results'/f'paired_{P["paired_scenarios"]}_seed{offset}_trades.csv')
        solver.append(dict(seed_offset=offset,max_gap=float(trades.objective_gap.max()),
            max_violation=float(trades.primal_violation.max()),max_cuts=int(trades.tail_cuts.max()),
            inaccurate=int((trades.status=='optimal_inaccurate').sum()),solutions=len(trades)))
        for phase,mask in [('exploratory',daily.year<2025),('confirmation',daily.year==2025)]:
            pivot=daily[mask].pivot(index='date',columns='strategy',values='return_net')
            ids=bootstrap_indices(len(pivot),seed=89)
            nominal=pivot['Residual Heston'].to_numpy()
            for strategy in ['Fixed DRO','Adaptive DRO']:
                robust=pivot[strategy].to_numpy();a=robust[ids];b=nominal[ids]
                mean=(a-b).mean(axis=1)*252;tail=batch_es(a)-batch_es(b)
                comparisons.append(dict(seed_offset=offset,phase=phase,strategy=strategy,
                    mean_difference=float((robust-nominal).mean()*252),mean_low=float(np.quantile(mean,.025)),mean_high=float(np.quantile(mean,.975)),
                    es_difference=es(-robust)-es(-nominal),es_low=float(np.quantile(tail,.025)),es_high=float(np.quantile(tail,.975))))
    pd.DataFrame(comparisons).to_csv(ROOT/'results'/'paired_bootstrap.csv',index=False)
    pd.DataFrame(solver).to_csv(ROOT/'results'/'paired_solver_summary.csv',index=False)
    assets=[]
    for ticker in P['tickers']:
        d=pd.read_csv(ROOT/'data'/f'{ticker}.csv',index_col=0,parse_dates=True)
        assets.append(dict(ticker=ticker,year=2025,adjusted_close_return=float(d.loc['2025'].Close.iloc[-1]/d.loc[:'2024-12-31'].Close.iloc[-1]-1)))
    pd.DataFrame(assets).to_csv(ROOT/'results'/'asset_calendar_returns.csv',index=False)
    t=pd.read_csv(ROOT/'results'/'portfolio_trades.csv');g=t[(t.strategy=='Historical CVaR')&(t.year==2025)]
    concentration=dict(year=2025,strategy='Historical CVaR',mean_gold_weight=float(g.w_GLD.mean()),median_gold_weight=float(g.w_GLD.median()),
        mean_effective_assets=float(g.effective_assets.mean()),signals=len(g))
    (ROOT/'results'/'gold_concentration.json').write_text(json.dumps(concentration,indent=2))
    fits=[json.loads(p.read_text()) for p in (ROOT/'results').glob('*_calibration_20??.json')]
    report=dict(fits=len(fits),refinements_converged=sum(next(x for x in f['optimizer'] if x['stage']=='refine')['success'] for f in fits),
        fit_beats_fixed=sum(f['loglik_fit']>f['loglik_fixed'] for f in fits),boundary_fits=sum(f['boundary'] for f in fits),
        coarse_particles=P['particles_fit'],refine_particles=P['particles_refine'],check_particles=P['particles_check'],check_seeds=P['likelihood_check_seeds'],
        selected_original_start=sum(f['selected_candidate']==3 for f in fits),parameter_display_significant_figures=2)
    (ROOT/'results'/'calibration_accuracy.json').write_text(json.dumps(report,indent=2))
    print('Paired analysis complete',flush=True)

if __name__=='__main__':main()
