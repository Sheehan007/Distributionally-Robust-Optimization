"""Compare tail separation against the full 20,000-row conic epigraph.

Four fixed schedule positions are chosen independently of numerical differences.
"""
import numpy as np
import pandas as pd
from portfolio import scenario_catalog,ConvexAllocator
from tail_allocator import TailAllocator
from forecast import ROOT,P

def main():
    frames,dates,opens,catalog=scenario_catalog(20000,save=False,engines=['residual'],signal_positions=[0,60,302,330])
    primary=pd.read_csv(ROOT/'results'/'paired_20000_seed0_trades.csv')
    primary=primary[primary.strategy=='Residual Heston'].set_index('date');rows=[]
    for item in catalog:
        i=item['index'];prior_date=str(dates[i-5].date())
        if prior_date in primary.index:
            pw=primary.loc[prior_date,[f'w_{t}' for t in P['tickers']]].to_numpy(float)
            change=opens[i+1]/opens[i-4]-1;previous=pw*(1+change)/(1+pw@change)
        else:previous=np.zeros(4)
        full=ConvexAllocator(20000);cut=TailAllocator(20000)
        for epsilon in [0.,item['epsilon']]:
            a,da=full.solve(item['scenarios']['residual'],item['means'],previous,epsilon)
            b,db=cut.solve(item['scenarios']['residual'],item['means'],previous,epsilon)
            delta=abs(da['objective']-db['objective']);distance=float(np.abs(a-b).sum())
            assert delta<5e-8,(item['date'],delta)
            assert distance<.004,(item['date'],distance)
            rows.append(dict(date=item['date'],epsilon=epsilon,objective_difference=delta,weight_l1_difference=distance,
                full_status=da['status'],cut_status=db['status'],separation_gap=db['objective_gap']))
            print(item['date'],epsilon,delta,distance,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results'/'full_epigraph_validation.csv',index=False)

if __name__=='__main__':main()
