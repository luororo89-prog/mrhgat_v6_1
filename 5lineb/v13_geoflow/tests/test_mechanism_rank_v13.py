import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism_rank_v13 import MechanismReference, ranking_loss


def fixture():
    rng=np.random.default_rng(5);n=30
    x=rng.normal(size=(n,9));raw=np.exp(x)
    cols=[f'{m}_{t}' for t in ('TH','SH','CHC') for m in ('RA','SA','TC')]
    W=rng.integers(1,20,size=(n,n)).astype(float)
    visible=rng.random((n,n))<.6;np.fill_diagonal(visible,False)
    context=W*visible
    g=dict(x=x,raw=raw,columns=cols,W=W)
    d=np.zeros((n,n));top=np.ones((n,n),dtype=int);np.fill_diagonal(top,0)
    return g,context,d,top


def test_loss_only_updates_corresponding_gate():
    q=torch.tensor([[.2,.8],[.8,.3]],requires_grad=True)
    ranking_loss(q,(np.array([0]),np.array([1])),0).backward()
    assert q.grad[0,0]<0 and q.grad[1,0]>0
    assert torch.equal(q.grad[:,1],torch.zeros(2))


def test_masked_counts_do_not_change_rules_or_references():
    g,c,d,t=fixture();ref=MechanismReference(g,d,t,c)
    changed={**g,'W':g['W'].copy()}
    hidden=(c==0)&~np.eye(len(c),dtype=bool)
    changed['W'][hidden]=1e8
    other=MechanismReference(changed,d,t,c)
    oi,di=np.nonzero(c)
    np.testing.assert_allclose(ref.references(c,oi,di),other.references(c,oi,di))
    assert ref.state()==other.state()


def test_origin_pressure_and_missing_service():
    g,c,d,t=fixture();ref=MechanismReference(g,d,t,c)
    p=ref.pressure(c)
    c2=c.copy();c2[2,0]+=100
    p2=ref.pressure(c2)
    assert p2[0]>p[0]
    np.testing.assert_allclose(p2[1:],p[1:])
    g['raw'][0,[g['columns'].index('RA_'+v) for v in ('TH','SH','CHC')]]=0
    ref=MechanismReference(g,d,t,c)
    assert np.isnan(ref.references(c,np.array([0]),np.array([1]))[0,1])


def test_training_pairs_are_visible_and_matched():
    g,c,d,t=fixture();ref=MechanismReference(g,d,t,c)
    oi,di=np.nonzero(~np.eye(len(c),dtype=bool))
    r,p=ref.pairs(c,oi,di)
    for k,(a,b) in enumerate(p):
        assert len(a)>0
        assert (c[oi[a],di[a]]>0).all() and (c[oi[b],di[b]]>0).all()
        endpoint=oi if k==0 else di
        assert (endpoint[a]==endpoint[b]).all()
        assert (r[a,k]>r[b,k]+ref.margin*ref.scales[k]).all()


def test_supply_gap_needs_origin_deficit_and_destination_advantage():
    g,c,d,t=fixture();g['x'][0]=1.;g['x'][1]=2.;g['x'][2]=-2.;g['x'][3]=-3.
    ref=MechanismReference(g,d,t,c)
    r=ref.references(c,np.array([0,2,2]),np.array([1,3,1]))
    assert r[0,0]==0 and r[1,0]==0 and r[2,0]>0


def test_rank_wrapper_preserves_forward_and_checkpoint(tmp_path):
    from test_od_mechanism_v13 import example, run
    from mechanism_rank_v13 import ranked_forward
    model,x,d,t,c,internal,raw,cols=example()
    g=dict(x=x.numpy(),raw=raw,columns=cols,W=c+np.diag(internal))
    ref=MechanismReference(g,d.numpy(),t.numpy(),c)
    def original(model,variant,g_,tensors,context,tau):
        return run(model,x,d,t,context,internal,raw,cols,train=True)
    baseline=original(model,None,g,None,c,.5)
    wrapped=ranked_forward(original,ref)(model,None,g,None,c,.5)
    torch.testing.assert_close(wrapped['score'],baseline['score'],rtol=0,atol=0)
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    fresh,*_=example();fresh.load_state_dict(torch.load(path,weights_only=True))
    replay=original(fresh,None,g,None,c,.5)
    torch.testing.assert_close(replay['score'],baseline['score'],rtol=0,atol=0)
