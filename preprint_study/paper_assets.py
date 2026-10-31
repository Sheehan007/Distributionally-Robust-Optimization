"""Generate paper assets from public summaries and available local path caches."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast import ROOT,P
FIG=ROOT/'paper'/'figures';TAB=ROOT/'paper'/'tables'
FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'axes.labelsize':9,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.2})
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');fig.savefig(FIG/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)
def table(name,rows):
    # Last alignment row is terminated by the parent document after input.
    (TAB/f'study_{name}.tex').write_text(' \\\\\n'.join(' & '.join(str(x) for x in row) for row in rows)+'\n')
def pct(x):return f'{100*x:.2f}'
def sig2(x):
    if x==0:return '0'
    decimals=max(0,1-int(np.floor(np.log10(abs(x)))))
    return f'{x:.{decimals}f}'
fm=pd.read_csv(ROOT/'results'/'forecast_metrics.csv')
labels={'persistence':'Persistence','update':'Window update','har':'HAR-type OLS','ewma':'EWMA','level':'Level LSTM',
    'residual_raw':'Raw residual LSTM','residual':'Gated residual LSTM'}
rows=[]
for model,label in labels.items():
    row=[label]
    for phase in ['exploratory','confirmation']:
        for ticker in P['tickers']:
            r=fm[(fm.phase==phase)&(fm.ticker==ticker)&(fm.model==model)].iloc[0];row.append(f'{r.rmse:.4f}')
    rows.append(row)
table('forecast',rows)

fig,ax=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
for a,phase,title in zip(ax,['exploratory','confirmation'],['2019–2024 exploratory','2025 confirmation']):
    for k,model in enumerate(['level','residual','update','har']):
        ratios=[]
        for t in P['tickers']:
            g=fm[(fm.phase==phase)&(fm.ticker==t)].set_index('model')
            ratios.append(g.loc[model,'rmse']/g.loc['persistence','rmse'])
        a.bar(np.arange(4)+(k-1.5)*.18,ratios,width=.17,label=labels[model])
    a.axhline(1,color='black',lw=.8,ls='--');a.set_xticks(range(4),P['tickers']);a.set_title(title)
ax[0].set_ylabel('RMSE relative to persistence')
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.05),ncol=4,frameon=False)
fig.tight_layout(rect=(0,.08,1,1));save(fig,'study_forecast')

cal=pd.read_csv(ROOT/'results'/'calibration_all.csv')
rows=[]
for t in P['tickers']:
    r=cal[(cal.ticker==t)&(cal.year==2025)].iloc[0]
    rows.append([t,*[sig2(r[k]) for k in ['kappa','theta','xi','rho','mu']],sig2(r.loglik_fit_seed_sd)])
table('calibration',rows)
score=pd.read_csv(ROOT/'results'/'calibration_scores.csv');rows=[]
for _,r in score.iterrows():
    rows.append([r.ticker,'2019--24' if r.phase=='exploratory' else '2025',f'{r.fitted_logscore:.4f}',f'{r.fixed_logscore:.4f}',
        f'{r.difference:.4f}',f'[{r.low:.4f}, {r.high:.4f}]'])
table('logscores',rows)
fig,axes=plt.subplots(2,2,figsize=(7,3.8),sharex=True)
for a,key,label in zip(axes.flat,['kappa','theta','xi','rho'],[r'$\kappa$',r'$\theta$',r'$\xi$',r'$\rho$']):
    for t in P['tickers']:
        g=cal[cal.ticker==t];a.plot(g.year,g[key],'o-',markersize=3,lw=1,label=t)
    a.set_ylabel(label)
axes[0,0].legend(ncol=2,frameon=False);fig.tight_layout();save(fig,'study_calibration')

metrics=pd.read_csv(ROOT/'results'/'portfolio_metrics.csv');rows=[]
names={'Buy and hold':'Buy and hold','Equal weight':'Equal weight','Minimum variance':'Minimum variance',
    'Historical CVaR':'Historical mean--CVaR','Filtered Heston':'Fitted Heston / filter','Persistence Heston':'Fitted Heston / persistence',
    'Residual Heston':'Fitted Heston / residual','Fixed parameter ablation':'Fixed parameters / residual',
    'Residual Gaussian':'Gaussian / residual','Fixed DRO':'Fixed-radius DRO','Adaptive DRO':'Adaptive-radius DRO'}
for s in names:
    row=[names[s]]
    for phase in ['exploratory','confirmation']:
        r=metrics[(metrics.strategy==s)&(metrics.phase==phase)].iloc[0]
        row += [pct(r.cagr),f'{r.sharpe:.3f}',pct(r.es95),pct(-r.max_drawdown),f'{r.mean_effective_assets:.2f}']
    rows.append(row)
table('portfolio',rows)
colors={'Equal weight':'#555555','Minimum variance':'#999999',
    'Residual Heston':'#d07a24','Fixed DRO':'#756bb1','Adaptive DRO':'#a52a38'}
if (ROOT/'results'/'portfolio_daily.csv').exists() and (ROOT/'results'/'paired_20000_seed0_daily.csv').exists():
    dd=pd.read_csv(ROOT/'results'/'portfolio_daily.csv')
    paired_path=pd.read_csv(ROOT/'results'/'paired_20000_seed0_daily.csv')
    dd=pd.concat([dd[dd.strategy.isin(['Equal weight','Minimum variance'])],paired_path]);dd['date']=pd.to_datetime(dd.date)
    fig,axes=plt.subplots(2,1,figsize=(7,3.8),sharex=True)
    for s,color in colors.items():
        g=dd[dd.strategy==s];nav=np.cumprod(1+g.return_net.to_numpy())
        axes[0].plot(g.date,nav,color=color,lw=1,label=names[s]);axes[1].plot(g.date,nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1,color=color,lw=.9)
    for a in axes:a.axvline(pd.Timestamp('2025-01-01'),color='black',ls='--',lw=.7)
    axes[0].set_ylabel('Net wealth from 1');axes[1].set_ylabel('Drawdown')
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
    fig.tight_layout(rect=(0,.09,1,1));save(fig,'study_wealth')
    
    trades=pd.read_csv(ROOT/'results'/'paired_20000_seed0_trades.csv');trades['date']=pd.to_datetime(trades.date)
    fig,axes=plt.subplots(2,1,figsize=(7,3.4),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    g=trades[trades.strategy=='Adaptive DRO'];axes[0].stackplot(g.date,*[g[f'w_{t}'] for t in P['tickers']],labels=P['tickers'],alpha=.9)
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Target portfolio weights');axes[0].legend(ncol=4,frameon=False,loc='upper left')
    for s in ['Residual Heston','Fixed DRO','Adaptive DRO']:
        g=trades[trades.strategy==s];axes[1].plot(g.date,g.effective_assets,color=colors[s],lw=1,label=names[s])
    axes[1].set_ylabel('Effective assets');axes[1].legend(ncol=3,frameon=False,fontsize=7,loc='upper left');axes[1].set_ylim(1,4.2)
    for a in axes:a.axvline(pd.Timestamp('2025-01-01'),color='black',ls='--',lw=.7)
    fig.tight_layout();save(fig,'study_weights')
rows=[]
boot=pd.read_csv(ROOT/'results'/'paired_bootstrap.csv')
for _,r in boot[boot.seed_offset==0].iterrows():
    rows.append(['2019--24' if r.phase=='exploratory' else '2025',names[r.strategy],f'{100*r.mean_difference:.2f}',
        f'[{100*r.mean_low:.2f}, {100*r.mean_high:.2f}]',f'{10000*r.es_difference:.2f}',f'[{10000*r.es_low:.2f}, {10000*r.es_high:.2f}]'])
table('bootstrap',rows)
cash=pd.read_csv(ROOT/'results'/'cash_metrics.csv');rows=[]
for s in ['Persistence Heston','Residual Heston','Fixed DRO','Adaptive DRO']:
    row=[names[s]]
    for phase in ['exploratory','confirmation']:
        r=cash[(cash.strategy==s)&(cash.phase==phase)].iloc[0];row += [pct(r.cagr),pct(r.es95),pct(r.mean_gross)]
    rows.append(row)
table('cash',rows)
rows=[]
for _,r in pd.read_csv(ROOT/'results'/'portfolio_quantiles.csv').iterrows():
    if r.strategy not in ['Residual Heston','Adaptive DRO','Fixed parameter ablation']:continue
    rows.append(['2019--24' if r.phase=='exploratory' else '2025',names[r.strategy],int(r.n),int(r.breaches),pct(r.rate),f'[{100*r.low:.2f}, {100*r.high:.2f}]'])
table('quantiles',rows)

rows=[]
for ticker in P['tickers']:
    for phase in ['exploratory','confirmation']:
        row=[ticker,'2019--24' if phase=='exploratory' else '2025']
        g=fm[(fm.ticker==ticker)&(fm.phase==phase)].set_index('model')
        for m in ['persistence','update','residual']:row += [f'{g.loc[m,"mae"]:.4f}',f'{g.loc[m,"qlike_forward5"]:.4f}']
        rows.append(row)
table('additional_forecast',rows)
rows=[]
for _,r in cal.iterrows():
    rows.append([r.ticker,int(r.year),*[sig2(r[k]) for k in ['kappa','theta','xi','rho']],f'{r.loglik_fit-r.loglik_fixed:.2f}',sig2(r.loglik_fit_seed_sd)])
table('all_parameters',rows)
sens=pd.read_csv(ROOT/'results'/'portfolio_sensitivity.csv');rows=[]
for _,r in sens.iterrows():
    rows.append([r['vary'],f'{r.setting:g}',names[r.strategy],pct(r.cagr),pct(r.es95),f'{r.mean_effective_assets:.2f}'])
table('sensitivity',rows)
mcpath=ROOT/'results'/'monte_carlo_summary.csv'
if mcpath.exists():
    mc=pd.read_csv(mcpath);rows=[]
    for _,g in mc.iterrows():
        rows.append([int(g.scenarios),int(g.seed_offset),f'{g.mean_adaptive_distance:.4f}',f'{g.p95_adaptive_distance:.4f}',f'{g.mean_robustness_tilt:.4f}'])
    table('monte_carlo',rows)
paired=pd.read_csv(ROOT/'results'/'paired_metrics.csv');rows=[]
for strategy in ['Residual Heston','Fixed DRO','Adaptive DRO']:
    row=[names[strategy]]
    for phase in ['exploratory','confirmation']:
        r=paired[(paired.strategy==strategy)&(paired.phase==phase)&(paired.seed_offset==0)].iloc[0]
        row += [pct(r.cagr),f'{r.sharpe:.3f}',pct(r.es95),pct(-r.max_drawdown),f'{r.mean_effective_assets:.2f}']
    rows.append(row)
table('paired',rows)
rows=[]
for offset in P['paired_seed_offsets']:
    for phase in ['exploratory','confirmation']:
        g=paired[(paired.seed_offset==offset)&(paired.phase==phase)].set_index('strategy')
        b=boot[(boot.seed_offset==offset)&(boot.phase==phase)&(boot.strategy=='Adaptive DRO')].iloc[0]
        rows.append([offset,'2019--24' if phase=='exploratory' else '2025',pct(g.loc['Residual Heston','cagr']),
            pct(g.loc['Adaptive DRO','cagr']),f'{100*b.mean_difference:.3f}',f'{10000*b.es_difference:.2f}'])
table('paired_seeds',rows)
print('Study paper assets generated')
