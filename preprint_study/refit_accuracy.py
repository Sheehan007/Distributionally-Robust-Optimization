"""Refit all 28 historical windows, without using evaluation outcomes."""
import json,time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from calibrate import fit,calibrate_asset
from forecast import ROOT,P,data

def one(task):
    ticker,year=task;d=data(ticker)
    hist=d[d.index<pd.Timestamp(year,1,1)].tail(P['calibration_window']+P['calibration_warmup'])
    starts=json.loads((ROOT/'calibration_starts.json').read_text())
    result=fit(hist.r.to_numpy(),hist.rv.to_numpy(),1000+year,starts[f'{ticker}:{year}'])
    result.update(ticker=ticker,year=year,fit_start=str(hist.index.min().date()),fit_end=str(hist.index.max().date()))
    (ROOT/'results'/f'{ticker}_calibration_{year}.json').write_text(json.dumps(result,indent=2))
    return ticker,year,result['elapsed_seconds'],result['loglik_fit']-result['loglik_fixed']

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        for item in pool.map(one,[(t,y) for t in P['tickers'] for y in P['evaluation_years']]):print(item,flush=True)
    for t in P['tickers']:calibrate_asset(t)
    print('High-accuracy fits and causal filters complete',flush=True)
