"""CVaR outer approximation, evaluating all scenarios at each separation step.

Each cut is a valid capped-simplex tail average. The master supplies an objective
upper bound and the full empirical CVaR supplies a feasible lower bound. This is
an implementation of standard convex cutting planes, not a new risk method.
"""
import numpy as np
import cvxpy as cp
from forecast import P
from portfolio import D,Q

def tail_gradient(scenarios,w):
    loss=-scenarios@w;order=np.argsort(loss)[::-1]
    mass=Q*len(loss);whole=int(np.floor(mass+1e-10));fraction=mass-whole
    a=-scenarios[order[:whole]].sum(axis=0)
    if fraction>1e-9:a-=fraction*scenarios[order[whole]]
    return a/mass

class TailAllocator:
    def __init__(self,n,cash=False,gamma=None,maxcuts=180):
        self.n=n;self.cash=cash;self.gamma=P['gamma'] if gamma is None else gamma;self.maxcuts=maxcuts
        self.w=cp.Variable(D);self.tail=cp.Variable();self.turn=cp.Variable(D,nonneg=True)
        self.A=cp.Parameter((maxcuts,D));self.mean=cp.Parameter(D);self.root=cp.Parameter((D,D))
        self.prev=cp.Parameter(D);self.eps=cp.Parameter(nonneg=True);self.cost=cp.Parameter(nonneg=True)
        rf=(1+P['cash_annual'])**(P['horizon_days']/252)-1
        self.income=(1-cp.sum(self.w))*rf if cash else 0
        # Express the tail in percentage units and utility in per-thousand units
        # to avoid imposing return-scale absolute tolerances on a tiny objective.
        constraints=[self.w>=0,self.tail>=self.A@self.w-100*self.income,
            self.turn>=self.w-self.prev,self.turn>=self.prev-self.w]
        constraints.append(cp.sum(self.w)<=1 if cash else cp.sum(self.w)==1)
        risk=10*self.tail+self.eps*cp.norm(self.w,2)/Q
        objective=self.mean@self.w+1000*self.income-self.gamma*risk-.5*cp.sum_squares(self.root@self.w)-self.cost*cp.sum(self.turn)
        self.problem=cp.Problem(cp.Maximize(objective),constraints)
        assert self.problem.is_dcp() and self.problem.is_dpp()

    def solve(self,scenarios,mean,previous,epsilon=0.,cost=None,lam=None):
        cost=P['transaction_cost'] if cost is None else cost;lam=P['quadratic_risk_lambda'] if lam is None else lam
        cov=np.cov(scenarios,rowvar=False,ddof=0);cov+=.05*np.trace(cov)/D*np.eye(D)
        self.mean.value=1000*np.asarray(mean);self.prev.value=previous
        self.root.value=np.linalg.cholesky(1000*lam*cov).T if lam>0 else np.zeros((D,D))
        self.eps.value=1000*max(float(epsilon),0);self.cost.value=1000*float(cost)
        starts=[np.ones(D)/D,*np.eye(D),np.asarray(previous)]
        cuts=[tail_gradient(scenarios,w) for w in starts]
        values=np.tile(cuts[0],(self.maxcuts,1))
        iterations=0;gap=np.inf
        for iteration in range(self.maxcuts-len(cuts)):
            values[:len(cuts)]=cuts;self.A.value=100*values
            self.problem.solve(solver='CLARABEL',warm_start=True,max_iter=150,
                tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
            if self.problem.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(self.problem.status)
            w=np.asarray(self.w.value).reshape(-1);gradient=tail_gradient(scenarios,w)
            income=(1-w.sum())*((1+P['cash_annual'])**(P['horizon_days']/252)-1) if self.cash else 0.
            exact_tail=float(gradient@w-income)
            gap=max(0.,self.gamma*(exact_tail-float(self.tail.value)/100))
            iterations+=int(self.problem.solver_stats.num_iters)
            if gap<=1e-10:break
            cuts.append(gradient)
        else:raise RuntimeError(f'Tail cut limit reached: objective gap {gap}')
        violation=max(float(np.maximum(-w,0).max()),max(0,float(w.sum()-1)) if self.cash else abs(float(w.sum()-1)))
        assert violation<2e-6,(self.problem.status,w)
        w=np.maximum(w,0)
        if not self.cash:w/=w.sum()
        elif w.sum()>1:w/=w.sum()
        exact_tail=float(tail_gradient(scenarios,w)@w-income)
        objective=float(mean@w+income-self.gamma*(exact_tail+epsilon*np.linalg.norm(w)/Q)-.5*lam*w@cov@w-cost*np.abs(w-previous).sum())
        return w,dict(status=self.problem.status,primal_violation=violation,iterations=iterations,
            objective=objective,risk_quadratic=float(.5*lam*w@cov@w),tail_cuts=len(cuts),objective_gap=gap)
