import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from D_mrhgat.model import MRHGAT, cross_entropy
from D_mrhgat.data import shares, validate_keys, masks, load_graph
from D_mrhgat.train import seed_all, fit_baseline

torch.set_num_threads(3)

def fixture(n=24):
    seed_all(0)
    x = torch.randn(n,9)
    F = torch.poisson(torch.ones(n,n)*3)
    K = torch.full((n,n),2,dtype=torch.long)
    K[torch.arange(n), (torch.arange(n)+1)%n] = 1
    K[torch.arange(n), (torch.arange(n)-1)%n] = 1
    K.fill_diagonal_(0)
    return {'x':x,'F':F,'K':K, 'fold':torch.arange(n)%5, 'ids':torch.arange(n)}

@pytest.mark.parametrize('variant', ['nested','independent','none'])
def test_normalization_mass_and_self(variant):
    d = fixture()
    out = MRHGAT(int_channel=variant)(d['x'], d['F'], d['K'])
    torch.testing.assert_close(out['pi'].sum(1), torch.ones(24))
    torch.testing.assert_close(out['r'].sum(1), torch.ones(24,24))
    M = d['F'][:,None,:] * out['r']
    torch.testing.assert_close(M.sum(1), d['F'])
    assert (out['r'][:,1:3,:].diagonal(dim1=0, dim2=2) == 0).all()
    if variant != 'none':
        assert (out['r'][:,3,:][d['K'] != 0] == 0).all()

@pytest.mark.parametrize('variant', ['nested','independent','none'])
def test_zero_inflow_and_gradients_finite(variant):
    d = fixture()
    d['F'][:, :3] = 0
    m = MRHGAT(int_channel=variant)
    out = m(d['x'], d['F'], d['K'])
    assert torch.isfinite(out['logpi']).all()
    cross_entropy(out['logpi'], shares(d['F']), d['F'].sum(1)>0).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

def test_heldout_and_buffer_shares_cannot_change_loss_or_gradient():
    d = fixture(30)
    # A contiguous held-out set permits a nonempty training area.
    d['fold'] = torch.arange(30)//6
    train, evaluation = masks(d, 0)
    target = shares(d['F'])
    altered = target.clone()
    altered[~train] = torch.softmax(torch.randn_like(altered[~train]),1)
    m = MRHGAT()
    before = d['F'].clone()
    gradients, losses = [], []
    for share in (target, altered):
        m.zero_grad()
        loss = cross_entropy(m(d['x'],d['F'],d['K'])['logpi'],share,train)
        loss.backward()
        gradients.append([p.grad.clone() if p.grad is not None else None for p in m.parameters()])
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
    for a,b in zip(*gradients):
        if a is not None:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(before,d['F'],rtol=0,atol=0)

def test_input_contract_excludes_capacity_time_and_posthoc():
    expected = {'x','F','K','ids','fold'}
    validate_keys(expected)
    for key in ('S','delta','Pi','PiL','I','C','T','logC','capacity','tier'):
        with pytest.raises(ValueError):
            validate_keys(expected | {key})
    graph = load_graph()
    assert graph['x'].shape == (666,9)
    assert graph['F'].sum().item() == 330831
    assert int((graph['K']==1).sum()) == 3998

def test_citywide_scaling_does_not_apply_local_capacity():
    d = fixture()
    m = MRHGAT()
    a = m(d['x'],d['F'],d['K'])
    b = m(d['x'],10*d['F'],d['K'])
    torch.testing.assert_close(a['pi'],b['pi'])

def test_baseline_recovers_supply_and_topology_on_expected_counts():
    d = fixture(70)
    beta = torch.tensor([.2,-.15,.3,-.25,.15,.2,-.15,.25,-.2])
    topo = torch.tensor([1.5,.7,0.])
    p = torch.softmax((d['x']@beta)[None,:]+topo[d['K']],1)
    d['F'] = p * 100000
    m,_ = fit_baseline(d,epochs=400)
    torch.testing.assert_close(m.beta,beta,rtol=.1,atol=.005)
    torch.testing.assert_close(m.topo-m.topo[2],topo,rtol=.1,atol=.01)
