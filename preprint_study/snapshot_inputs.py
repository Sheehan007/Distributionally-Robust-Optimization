"""Canonical hashes of the dated price and aligned VIX inputs actually used."""
from pathlib import Path
import hashlib
import pandas as pd

def fingerprints(root):
    root=Path(root);out={};spy=None
    for ticker in ['SPY','IWM','TLT','GLD']:
        d=pd.read_csv(root/'data'/f'{ticker}.csv',index_col=0,parse_dates=True).sort_index()
        d=d.loc['2012-01-01':'2025-12-31',['Open','Close']];d.index.name='Date'
        out[ticker]=hashlib.sha256(d.to_csv(float_format='%.17g',date_format='%Y-%m-%d').encode()).hexdigest()
        if ticker=='SPY':spy=d
    v=pd.read_csv(root/'data'/'VIX.csv');v.index=pd.to_datetime(v.DATE)
    used=pd.to_numeric(v.CLOSE,errors='coerce').reindex(spy.index).ffill().to_frame('VIX');used.index.name='Date'
    out['VIX_aligned']=hashlib.sha256(used.to_csv(float_format='%.17g',date_format='%Y-%m-%d').encode()).hexdigest()
    return out
