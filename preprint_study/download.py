"""Retrieve missing provider inputs locally after explicit terms acknowledgment.

No market observations or API responses are included in the public release.
The endpoint is unofficial; access and redistribution rights are not granted
by this script. Licensed local inputs in the documented format are also usable.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parent
TERMS={'Yahoo':'https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html',
       'Cboe':'https://www.cboe.com/terms'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge-terms',action='store_true',help='Confirm you reviewed provider terms and have appropriate access rights.')
    args=parser.parse_args()
    if not args.acknowledge_terms:parser.error('Review DATA_ACCESS.md and the provider terms before requesting downloads; then pass --acknowledge-terms if appropriate.')
    (ROOT/'data').mkdir(exist_ok=True);sources={};retrieval={}
    for ticker in ['SPY','IWM','TLT','GLD']:
        url=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1325376000&period2=1767225600&interval=1d'
        path=ROOT/'data'/f'{ticker}.csv';sources[path.name]=url
        if path.exists():continue
        response=requests.get(url,timeout=40,headers={'User-Agent':'Mozilla/5.0'});response.raise_for_status()
        (ROOT/'data'/f'{ticker}_raw.json').write_bytes(response.content)
        raw=response.json()['chart']['result'][0]
        dates=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        d=pd.DataFrame(raw['indicators']['quote'][0],index=dates);d.columns=[c.capitalize() for c in d.columns]
        adj=pd.Series(raw['indicators']['adjclose'][0]['adjclose'],index=d.index);factor=adj/d.Close
        for col in ['Open','High','Low','Close']:d[col]*=factor
        d=d.dropna();d.index.name='Date';assert len(d)>3400 and d.index.is_unique
        d.to_csv(path,float_format='%.12g');retrieval[path.name]=datetime.now(timezone.utc).isoformat()
        print(ticker,len(d),'observations saved locally',flush=True)
    vix=ROOT/'data'/'VIX.csv';sources[vix.name]='https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
    if not vix.exists():
        response=requests.get(sources[vix.name],timeout=40);response.raise_for_status();vix.write_bytes(response.content)
        retrieval[vix.name]=datetime.now(timezone.utc).isoformat()
    from snapshot_inputs import fingerprints
    files=[*(ROOT/'data').glob('*.csv'),*(ROOT/'data').glob('*_raw.json')]
    local=dict(retrieval_utc=datetime.now(timezone.utc).isoformat(),sources=sources,file_retrieval_utc=retrieval,
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},canonical_sha256=fingerprints(ROOT))
    (ROOT/'data'/'local_manifest.json').write_text(json.dumps(local,indent=2)+'\n')
    print('Saved local_manifest.json. Run verify_snapshot.py to compare the scientific inputs with the published snapshot hashes.')

if __name__=='__main__':main()
