"""Chronological residual and level LSTM comparison, with identity benchmark."""
from pathlib import Path
import copy,json,hashlib,math
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
torch.set_num_threads(1)
FEATURES=['r','lv','mean20','log_vix','outgoing_r','outgoing_sq']

def data(ticker):
    d=pd.read_csv(ROOT/'data'/f'{ticker}.csv',index_col=0,parse_dates=True)
    v=pd.read_csv(ROOT/'data'/'VIX.csv');v.index=pd.to_datetime(v.DATE)
    d['vix']=pd.to_numeric(v.CLOSE,errors='coerce').reindex(d.index).ffill()
    d['r']=np.log(d.Close).diff();d['rv']=d.r.rolling(21).std(ddof=0)*np.sqrt(252)
    d['lv']=np.log(d.rv.clip(lower=1e-6));d['target']=d.lv.shift(-1)
    d['target_date']=pd.Series(d.index,index=d.index).shift(-1)
    d['mean20']=d.r.rolling(20).mean();d['log_vix']=np.log(d.vix/100)
    d['outgoing_r']=d.r.shift(20);d['outgoing_sq']=d.outgoing_r**2
    d['har5']=d.lv.rolling(5).mean();d['har22']=d.lv.rolling(22).mean()
    d['ewma']=np.sqrt(d.r.pow(2).ewm(alpha=.06,adjust=False).mean()*252)
    # Exact expected update of the rolling population variance, conditional on
    # a next return with current-window mean and variance.
    n=21;s=d.r.rolling(n).sum()-d.outgoing_r
    a=d.r.pow(2).rolling(n).sum()-d.outgoing_sq
    m=d.r.rolling(n).mean();e2=d.rv.pow(2)/252+m*m
    nextvar=(a+e2)/n-(s*s+2*s*m+e2)/(n*n)
    d['update']=.5*np.log((252*nextvar).clip(lower=1e-10))
    d['forward_var5']=252/5*d.r.pow(2).rolling(5).sum().shift(-5)
    return d.dropna(subset=FEATURES+['target','target_date','har22','update','ewma']).copy()

class VolatilityNet(nn.Module):
    def __init__(self,residual=False):
        super().__init__();self.recurrent=nn.LSTM(len(FEATURES),P['hidden_size'],batch_first=True)
        self.head=nn.Linear(P['hidden_size'],1)
        if residual:
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        h,_=self.recurrent(x);return self.head(h[:,-1,:]).squeeze(-1)

def train_asset(ticker,reuse=False):
    d=data(ticker);length=P['sequence_length'];ids=np.arange(length-1,len(d))
    raw=d[FEATURES].to_numpy(dtype='float32')
    x=np.stack([raw[i-length+1:i+1] for i in ids])
    target=d.target.iloc[ids].to_numpy();base=d.lv.iloc[ids].to_numpy()
    available=d.target_date.iloc[ids];allrows=[];training=[]
    training_file=ROOT/'results'/f'{ticker}_training.csv'
    archived_training=pd.read_csv(training_file) if reuse and training_file.exists() else None
    for year in P['evaluation_years']:
        train=(available<pd.Timestamp(year-1,1,1)).to_numpy()
        valid=((available>=pd.Timestamp(year-1,1,1))&(available<pd.Timestamp(year,1,1))).to_numpy()
        test=d.index[ids].year==year
        tr=np.flatnonzero(train);va=np.flatnonzero(valid);te=np.flatnonzero(test)
        assert available[train].max()<available[valid].min()
        assert available[valid].max()<d.index[ids[te]].min()
        xm=x[train].mean(axis=(0,1));xs=x[train].std(axis=(0,1)).clip(1e-6)
        z=torch.from_numpy((x-xm)/xs)
        pred={};vpred={}
        for mode in ['residual','level']:
            y=target-base if mode=='residual' else target.copy()
            ym=0. if mode=='residual' else float(y[train].mean())
            ys=max(float(y[train].std()),1e-6)
            yy=torch.from_numpy(((y-ym)/ys).astype('float32'))
            pred[mode]=[];vpred[mode]=[]
            for seed in P['seeds']:
                torch.manual_seed(seed);rng=np.random.default_rng(seed)
                model=VolatilityNet(residual=mode=='residual')
                if reuse:
                    ck=torch.load(ROOT/'results'/f'{ticker}_{mode}_{year}_{seed}.pt',map_location='cpu',weights_only=True)
                    assert ck['features']==FEATURES and ck['mode']==mode
                    model.load_state_dict(ck['state_dict']);model.eval()
                    scale_mean=np.asarray(ck['feature_mean'],dtype='float32');scale_std=np.asarray(ck['feature_std'],dtype='float32')
                    zz=torch.from_numpy((x-scale_mean)/scale_std)
                    with torch.no_grad():
                        vp=model(zz[va]).numpy()*ck['target_std']+ck['target_mean']
                        tp=model(zz[te]).numpy()*ck['target_std']+ck['target_mean']
                    vpred[mode].append(vp);pred[mode].append(tp)
                    if archived_training is not None:
                        row=archived_training[(archived_training.year==year)&(archived_training['mode']==mode)&(archived_training.seed==seed)].iloc[0].to_dict()
                        training.append(row)
                    continue
                opt=torch.optim.Adam(model.parameters(),lr=P['learning_rate'])
                model.eval()
                with torch.no_grad():best=float(((model(z[va])-yy[va])**2).mean())
                state=copy.deepcopy(model.state_dict());best_epoch=0;wait=0
                for epoch in range(P['epochs']):
                    model.train();order=rng.permutation(tr)
                    for start in range(0,len(order),P['batch_size']):
                        b=order[start:start+P['batch_size']]
                        opt.zero_grad();loss=((model(z[b])-yy[b])**2).mean()
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                    model.eval()
                    with torch.no_grad():vl=float(((model(z[va])-yy[va])**2).mean())
                    if vl<best-1e-6:
                        best=vl;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;wait=0
                    else:wait+=1
                    if wait>=P['patience']:break
                model.load_state_dict(state);model.eval()
                with torch.no_grad():
                    vp=model(z[va]).numpy()*ys+ym;tp=model(z[te]).numpy()*ys+ym
                vpred[mode].append(vp);pred[mode].append(tp)
                training.append(dict(ticker=ticker,year=year,mode=mode,seed=seed,best_epoch=best_epoch,
                    train_n=len(tr),validation_n=len(va),test_n=len(te),validation_rmse=float(np.sqrt(np.mean((vp-y[va])**2)))))
                torch.save(dict(state_dict=state,features=FEATURES,feature_mean=xm.tolist(),feature_std=xs.tolist(),
                    target_mean=ym,target_std=ys,mode=mode),ROOT/'results'/f'{ticker}_{mode}_{year}_{seed}.pt')
        correction=np.mean(vpred['residual'],axis=0)
        val_delta=target[va]-base[va]
        gate=min(P['validation_gate_grid'],key=lambda g:np.mean((g*correction-val_delta)**2))
        validation_rmse=float(np.sqrt(np.mean((gate*correction-val_delta)**2)))
        residual=np.mean(pred['residual'],axis=0);level=np.mean(pred['level'],axis=0)
        h=np.column_stack([np.ones(len(d)),d[['lv','har5','har22']].to_numpy()])
        beta=np.linalg.lstsq(h[ids[tr]],target[tr],rcond=None)[0]
        for k,j in enumerate(te):
            row=d.iloc[ids[j]]
            rec=dict(date=str(d.index[ids[j]].date()),position=int(ids[j]),year=year,
                target=float(target[j]),persistence=float(base[j]),update=float(row['update']),ewma=float(np.log(row.ewma)),
                har=float(h[ids[j]]@beta),level=float(level[k]),residual_raw=float(base[j]+residual[k]),
                residual=float(base[j]+gate*residual[k]),gate=float(gate),validation_rmse=validation_rmse,
                forward_var5=float(row.forward_var5),seed_spread=float(np.std(np.asarray(pred['residual'])[:,k])))
            for n,seed in enumerate(P['seeds']):rec[f'residual_seed_{seed}']=float(base[j]+pred['residual'][n][k])
            allrows.append(rec)
        print(ticker,year,'forecast fits complete',flush=True)
    pd.DataFrame(allrows).to_csv(ROOT/'results'/f'{ticker}_forecasts.csv',index=False)
    if training:pd.DataFrame(training).to_csv(ROOT/'results'/f'{ticker}_training.csv',index=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-checkpoints',action='store_true')
    args=parser.parse_args()
    for ticker in P['tickers']:
        if args.reuse_checkpoints or not (ROOT/'results'/f'{ticker}_forecasts.csv').exists():train_asset(ticker,reuse=args.reuse_checkpoints)
