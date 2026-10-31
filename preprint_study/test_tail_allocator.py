"""Independent full-epigraph comparison, including fractional tail mass."""
import unittest
import numpy as np
from portfolio import ConvexAllocator
from tail_allocator import TailAllocator,tail_gradient

class TailTests(unittest.TestCase):
    def test_outer_approximation_matches_full_empirical_program(self):
        rng=np.random.default_rng(317)
        r=rng.normal(0,.025,(347,4));r[:,1]+=.3*r[:,0]
        mean=np.array([.002,.0015,.001,.0018]);previous=np.array([.3,.1,.4,.2])
        for cash in [False,True]:
            for eps in [0.,.0001]:
                a,da=ConvexAllocator(len(r),cash).solve(r,mean,previous,eps,cost=.0002)
                b,db=TailAllocator(len(r),cash).solve(r,mean,previous,eps,cost=.0002)
                self.assertLess(abs(da['objective']-db['objective']),2e-8)
                self.assertLess(np.abs(a-b).sum(),.002)
                self.assertLessEqual(db['objective_gap'],1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
