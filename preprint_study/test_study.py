"""Independent mathematical and temporal checks for revision two."""
import unittest,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm
import torch
from forecast import ROOT,P,data,VolatilityNet
from calibrate import particle_filter,draws
from portfolio import ConvexAllocator,es

class StudyTests(unittest.TestCase):
    def test_residual_epoch_zero_is_exact_persistence(self):
        torch.manual_seed(19);model=VolatilityNet(True)
        x=torch.randn(15,20,6)
        np.testing.assert_array_equal(model(x).detach().numpy(),np.zeros(15))

    def test_empirical_es_fractional_tail_mass(self):
        rng=np.random.default_rng(3)
        for n in [7,21,101]:
            losses=rng.normal(size=n);c=np.r_[1,np.ones(n)/(n*.05)]
            lp=linprog(c,A_ub=np.column_stack([-np.ones(n),-np.eye(n)]),b_ub=-losses,
                bounds=[(None,None)]+[(0,None)]*n,method='highs')
            self.assertTrue(lp.success);self.assertAlmostEqual(es(losses),lp.fun,places=9)

    def test_multivariate_wasserstein_tail_transport(self):
        rng=np.random.default_rng(17);r=rng.normal(0,.02,(20,4));w=np.array([.1,.2,.3,.4]);eps=.0001
        shifted=r.copy();shifted[np.argmin(r@w)]-=eps/.05*w/np.linalg.norm(w)
        self.assertAlmostEqual(np.linalg.norm(shifted-r,axis=1).mean(),eps)
        self.assertAlmostEqual(es(-shifted@w),es(-r@w)+eps*np.linalg.norm(w)/.05,places=10)

    def test_quadratic_optimizer_has_analytic_interior_solution(self):
        r=np.r_[np.eye(4),-np.eye(4)]*.02;mean=np.array([.001,.00105,.00110,.00115])
        solver=ConvexAllocator(len(r),gamma=0)
        w,diag=solver.solve(r,mean,np.zeros(4),epsilon=0,cost=0,lam=4)
        diagonal=.0001*1.05
        expected=.25+(mean-mean.mean())/(4*diagonal)
        np.testing.assert_allclose(w,expected,atol=3e-5)
        self.assertTrue(np.all(w>.05) and np.all(w<.5))

    def test_cvar_allocation_matches_independent_linear_program(self):
        rng=np.random.default_rng(120);r=rng.normal(.002,.02,(41,4));mean=r.mean(axis=0);gamma=.05;n=len(r)
        solver=ConvexAllocator(n,gamma=gamma)
        w,diag=solver.solve(r,mean,np.zeros(4),cost=0,lam=0)
        c=np.r_[-mean,gamma,np.ones(n)*gamma/(n*.05)]
        a=np.column_stack([-r,-np.ones(n),-np.eye(n)])
        lp=linprog(c,A_ub=a,b_ub=np.zeros(n),A_eq=np.r_[np.ones(4),np.zeros(n+1)][None,:],b_eq=[1],
            bounds=[(0,None)]*4+[(None,None)]+[(0,None)]*n,method='highs')
        self.assertTrue(lp.success)
        self.assertAlmostEqual(float(-mean@w+gamma*es(-r@w)),lp.fun,places=7)

    def test_particle_filter_reduces_to_known_gaussian_density(self):
        r=np.array([.01,-.005,.002,.001,-.01]);theta=.04;mu=.06
        z,u=draws(len(r),128,91)
        ll,v,ess=particle_filter(r,np.array([3.,theta,0.,-.6]),mu,theta,z,u)
        exact=norm.logpdf(r,loc=(mu-.5*theta)/252,scale=np.sqrt(theta/252))
        np.testing.assert_allclose(ll,exact,atol=1e-11)
        np.testing.assert_allclose(v,theta,atol=1e-11)
        np.testing.assert_allclose(ess,128,atol=1e-10)

    def test_leverage_conditioned_state_transition(self):
        n=2000;z=np.r_[np.linspace(-2,2,n//2),-np.linspace(-2,2,n//2)][None,:];u=np.array([.3])
        r=np.array([-.02]);k,theta,xi,rho=3.,.04,.25,-.6;mu=.06;dt=1/252
        ll,v,ess=particle_filter(r,np.array([k,theta,xi,rho]),mu,theta,z,u)
        shock=(r[0]-(mu-.5*theta)*dt)/np.sqrt(theta*dt)
        expected=theta+xi*np.sqrt(theta*dt)*rho*shock
        self.assertAlmostEqual(float(v[0]),expected,places=10)

    def test_calibration_and_training_dates_precede_evaluation(self):
        for ticker in P['tickers']:
            cal=pd.read_csv(ROOT/'results'/f'{ticker}_calibration.csv')
            for _,r in cal.iterrows():
                self.assertLess(pd.Timestamp(r.fit_end),pd.Timestamp(int(r.year),1,1))
                self.assertGreater(r.feller_gap,0)
            d=data(ticker);self.assertTrue(d.index.is_unique and d.index.is_monotonic_increasing)
            for year in P['evaluation_years']:
                val=d[(d.target_date>=pd.Timestamp(year-1,1,1))&(d.target_date<pd.Timestamp(year,1,1))]
                self.assertLess(val.target_date.max(),d[d.index.year==year].index.min())

    def test_portfolio_constraints_and_shared_daily_intervals(self):
        path=ROOT/'results'/'portfolio_trades.csv'
        if not path.exists():self.skipTest('Portfolio computation still running')
        t=pd.read_csv(path);w=t[[f'w_{x}' for x in P['tickers']]].to_numpy()
        np.testing.assert_allclose(w.sum(axis=1),1,atol=2e-6);self.assertGreater(w.min(),-1e-8)
        self.assertLess(t.primal_violation.max(),2e-6)
        d=pd.read_csv(ROOT/'results'/'portfolio_daily.csv');reference=None
        for _,g in d.groupby('strategy'):
            if reference is None:reference=g.date.tolist()
            self.assertEqual(reference,g.date.tolist())
            self.assertEqual(len(g),len(set(g.date)))

if __name__=='__main__':unittest.main(verbosity=2)
