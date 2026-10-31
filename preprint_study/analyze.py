"""Saved-result analysis with separate exploratory and confirmation summaries."""
import json,hashlib
import numpy as np
import pandas as pd
from forecast import ROOT,P
from portfolio import es

def bootstrap_indices(n,draws=2000,block=20,seed=73):
    rng=np.random.default_rng(seed);ids=np.empty((draws,n),dtype=np.int32);ids[:,0]=rng.integers(n,size=draws)
    for j in range(1,n):
        restart=rng.random(draws)<1/block
        ids[:,j]=np.where(restart,rng.integers(n,size=draws),(ids[:,j-1]+1)%n)
    return ids

def batch_es(returns):
    loss=np.sort(-returns,axis=1);n=loss.shape[1]*.05;k=int(np.floor(n+1e-10));frac=n-k
    return (loss[:,-k:].sum(axis=1)+(frac*loss[:,-k-1] if frac>1e-9 else 0))/n

def main():
    forecasts=[];forecast_tests=[];params=[];scores=[];gates=[];training=[]
    for ticker in P['tickers']:
        f=pd.read_csv(ROOT/'results'/f'{ticker}_forecasts.csv')
        cal=pd.read_csv(ROOT/'results'/f'{ticker}_calibration.csv');cal['ticker']=ticker;params.append(cal)
        state=pd.read_csv(ROOT/'results'/f'{ticker}_filtered_states.csv')
        for year,group in f.groupby('year'):
            gates.append(dict(ticker=ticker,year=int(year),gate=float(group.gate.iloc[0]),validation_rmse=float(group.validation_rmse.iloc[0])))
        for phase,mask in [('exploratory',f.year<2025),('confirmation',f.year==2025)]:
            g=f[mask];ids=bootstrap_indices(len(g),seed=73)
            for model in ['persistence','update','har','ewma','level','residual_raw','residual']+[f'residual_seed_{x}' for x in P['seeds']]:
                e=(g[model]-g.target).to_numpy();ratio=np.exp(2*(g.target-g[model]).to_numpy())
                # Independently observed forward squared-return variation, sampled
                # every five dates so target windows do not overlap.
                independent=g.iloc[::5].dropna(subset=['forward_var5'])
                q=independent.forward_var5.to_numpy()/np.exp(2*independent[model].to_numpy())
                forecasts.append(dict(ticker=ticker,phase=phase,model=model,n=len(g),rmse=float(np.sqrt(np.mean(e*e))),
                    mae=float(np.abs(e).mean()),qlike_proxy=float(np.mean(ratio-np.log(ratio)-1)),
                    qlike_forward5=float(np.mean(q-np.log(q)-1)),n_forward5=len(q)))
            for comparator in ['persistence','update','har','level']:
                delta=((g.residual-g.target)**2-(g[comparator]-g.target)**2).to_numpy()
                boot=delta[ids].mean(axis=1)
                forecast_tests.append(dict(ticker=ticker,phase=phase,comparator=comparator,mse_difference=float(delta.mean()),
                    low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975))))
            sf=state[state.year<2025] if phase=='exploratory' else state[state.year==2025]
            delta=(sf.logscore-sf.fixed_logscore).to_numpy();bi=bootstrap_indices(len(delta),seed=87)
            boot=delta[bi].mean(axis=1)
            scores.append(dict(ticker=ticker,phase=phase,n=len(sf),fitted_logscore=float(sf.logscore.mean()),
                fixed_logscore=float(sf.fixed_logscore.mean()),difference=float(delta.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975))))
    pd.DataFrame(forecasts).to_csv(ROOT/'results'/'forecast_metrics.csv',index=False)
    pd.DataFrame(forecast_tests).to_csv(ROOT/'results'/'forecast_bootstrap.csv',index=False)
    pd.concat(params).to_csv(ROOT/'results'/'calibration_all.csv',index=False)
    pd.DataFrame(scores).to_csv(ROOT/'results'/'calibration_scores.csv',index=False)
    pd.DataFrame(gates).to_csv(ROOT/'results'/'validation_gates.csv',index=False)
    daily=pd.read_csv(ROOT/'results'/'portfolio_daily.csv');trades=pd.read_csv(ROOT/'results'/'portfolio_trades.csv')
    comparisons=[]
    for phase,mask in [('exploratory',daily.year<2025),('confirmation',daily.year==2025)]:
        pivot=daily[mask].pivot(index='date',columns='strategy',values='return_net');ids=bootstrap_indices(len(pivot),seed=89)
        a=pivot['Adaptive DRO'].to_numpy()
        for comparator in ['Residual Heston','Persistence Heston','Filtered Heston','Equal weight','Fixed DRO']:
            b=pivot[comparator].to_numpy();aa=a[ids];bb=b[ids]
            means=(aa-bb).mean(axis=1)*252;tails=batch_es(aa)-batch_es(bb)
            comparisons.append(dict(phase=phase,comparator=comparator,mean_difference=float(np.mean(a-b)*252),
                mean_low=float(np.quantile(means,.025)),mean_high=float(np.quantile(means,.975)),es_difference=es(-a)-es(-b),
                es_low=float(np.quantile(tails,.025)),es_high=float(np.quantile(tails,.975))))
    pd.DataFrame(comparisons).to_csv(ROOT/'results'/'portfolio_bootstrap.csv',index=False)
    breaches=[]
    for phase,mask in [('exploratory',trades.year<2025),('confirmation',trades.year==2025)]:
        for strategy,g in trades[mask].groupby('strategy'):
            if strategy not in ['Residual Heston','Fixed DRO','Adaptive DRO','Filtered Heston','Fixed parameter ablation']:continue
            b=(g.realized_gross<g.scenario_q05).astype(float).to_numpy()
            boot=b[bootstrap_indices(len(b),block=4)].mean(axis=1)
            breaches.append(dict(phase=phase,strategy=strategy,n=len(g),breaches=int(b.sum()),rate=float(b.mean()),
                low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975))))
    pd.DataFrame(breaches).to_csv(ROOT/'results'/'portfolio_quantiles.csv',index=False)
    pd.concat([pd.read_csv(ROOT/'results'/f'{t}_training.csv') for t in P['tickers']]).to_csv(ROOT/'results'/'training_all.csv',index=False)
    meta=json.loads((ROOT/'results'/'portfolio_metadata.json').read_text())
    meta.update(protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),model_checkpoints=len(list((ROOT/'results').glob('*.pt'))))
    (ROOT/'results'/'study_summary.json').write_text(json.dumps(meta,indent=2))
    print('Analysis complete')

if __name__=='__main__':main()
