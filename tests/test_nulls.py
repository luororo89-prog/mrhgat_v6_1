import sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from D_mrhgat.batched_train import fit_batch,baseline_batch
from D_mrhgat.train import fit,fit_baseline
from D_mrhgat.nulls import null_probability,holm,decide

torch.set_num_threads(3)

def fixture():
    torch.manual_seed(10)
    n=15
    x=torch.randn(n,9)
    K=torch.full((n,n),2,dtype=torch.long)
    K.fill_diagonal_(0)
    F=torch.poisson(torch.ones(2,n,n)*8)
    return x,K,F

@pytest.mark.parametrize('variant',['none','independent','nested'])
def test_batched_refits_match_separate_adam_trajectories(variant):
    x,K,F=fixture()
    base={'beta':torch.zeros(2,9),'topo':torch.zeros(2,3)}
    out,_=fit_batch(x,K,F,variant,[0,1],epochs=4,base_states=base)
    for index in range(2):
        _,single,_=fit({'x':x,'K':K,'F':F[index]},variant,index,
            {k:v[index] for k,v in base.items()},epochs=4)
        torch.testing.assert_close(out['pi'][index],single['pi'],rtol=2e-4,atol=2e-6)

def test_fit_batch_parallel_workers_match_sequential():
    x,K,F=fixture()
    base={'beta':torch.zeros(2,9),'topo':torch.zeros(2,3)}
    out_seq,_=fit_batch(x,K,F,'none',[0,1],epochs=4,base_states=base,n_workers=1)
    out_par,_=fit_batch(x,K,F,'none',[0,1],epochs=4,base_states=base,n_workers=2)
    torch.testing.assert_close(out_seq['pi'],out_par['pi'],rtol=0,atol=0)

def test_batched_baselines_match_separate_fits():
    x,K,F=fixture()
    batch=baseline_batch(x,K,F,epochs=30)
    for i in range(2):
        single,_=fit_baseline({'x':x,'K':K,'F':F[i]},epochs=30)
        torch.testing.assert_close(batch['beta'][i],single.beta,rtol=2e-4,atol=2e-6)
        torch.testing.assert_close(batch['topo'][i],single.topo,rtol=2e-4,atol=2e-6)

def test_nulls_preserve_probabilities_and_direction_decision():
    rho=np.array([[.6,.2,.2],[.55,.2,.25],[.7,.2,.1],[.65,.2,.15]])
    p=np.full((4,3,4),.25)
    for kind in ['NC','ND']:
        result=null_probability(rho,np.log(p),np.arange(4.),kind,np.random.default_rng(0))
        np.testing.assert_allclose(result.sum(1),1)
        assert (result>=0).all()
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    rows=decide({'a':-.5},[{'a':.01*i} for i in range(20)],['a'],[1])
    assert not rows[0]['detected']
