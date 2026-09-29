import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from D_mrhgat.synthetic import make_flows

@pytest.mark.parametrize('scenario',['S0','S1','S1s','S2','S3'])
def test_generation_conservation_and_mechanism_support(scenario):
    rng=np.random.default_rng(12)
    n=60
    x=rng.normal(size=(n,9))
    K=np.full((n,n),2)
    K[np.arange(n),(np.arange(n)+1)%n]=1
    np.fill_diagonal(K,0)
    F,t=make_flows(x,K,np.ones((n,n))/n,np.full(n,800),scenario,.3,2)
    F2,t2=make_flows(x,K,np.ones((n,n))/n,np.full(n,800),scenario,.3,2)
    np.testing.assert_array_equal(F,F2)
    np.testing.assert_allclose(t['expected'].sum((1,2)),F.sum(1))
    assert (t['expected']>=0).all()
    assert (t['expected'][:,1,:][t['delta']<=0]==0).all()
    assert (t['expected'][:,2,:][t['delta']>0]==0).all()
    assert (t['expected'][:,1:,:].diagonal(axis1=0,axis2=2)==0).all()
    if scenario=='S0':
        assert (t['expected'][:,1:,:]==0).all()
    if scenario in ('S1','S1s'):
        assert (t['expected'][:,2,:]==0).all()
