import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_od_relation_v13 import ODRelationNet
from model_v13 import MechNet
from run_od_relation_v13 import destination_ce, make_model, pressure_context


def example(variant='od_relation'):
    torch.manual_seed(4)
    n = 5
    topology = torch.ones(n, n, dtype=torch.long)
    topology.fill_diagonal_(0)
    cfg = dict(hid_dim=12, iters=7, routing_layers=5, dropout=0., attention=True)
    model = make_model(variant, cfg, n, topology)
    x = torch.randn(n, 9)
    distance = torch.rand(n, n)
    context = torch.tensor([[0,2,0,3,0],[0,0,4,0,1],[2,0,0,2,0],[0,3,1,0,0],[1,0,0,2,0.]])
    share = context / context.sum(1, keepdim=True)
    return model, (x, distance, topology, torch.ones(n,n,dtype=torch.bool), share, context>0, .5)


def test_shared_architecture_initial_parameters_and_rng():
    reference, _ = example('node_reference')
    reference_rng = torch.get_rng_state()
    model, _ = example()
    assert ODRelationNet.forward is MechNet.forward
    assert ODRelationNet.od_pair_scores is MechNet.od_pair_scores
    assert model.od_pair_head and reference.od_pair_head
    assert model.routing_layers == reference.routing_layers == 5
    assert model.iters == reference.iters == 7
    assert model.K == reference.K == 2 and model.d == reference.d == 6
    assert torch.equal(reference_rng, torch.get_rng_state())
    reference_state = reference.state_dict()
    state = model.state_dict()
    assert all(torch.equal(value, state[key]) for key,value in reference_state.items())
    assert set(state) == set(reference_state)


def test_destination_only_pair_reproduces_entire_reference():
    reference, inputs = example('node_reference')
    model, _ = example()
    with torch.no_grad():
        reference.Bt.copy_(torch.tensor([[.1,-.2,.3],[-.1,.2,-.3]]))
        model.Bt.copy_(reference.Bt)
        model.pair_origin.zero_()
        model.pair_destination.copy_(torch.eye(model.d).repeat(model.K,1,1))
        reference.pair_origin.copy_(model.pair_origin)
        reference.pair_destination.copy_(model.pair_destination)
    reference.eval()
    model.eval()
    actual, expected = model(*inputs), reference(*inputs)
    for key in ('p','zr','c_route','alpha','c','score','logpi'):
        assert torch.allclose(actual[key],expected[key],atol=3e-6,rtol=3e-6),key


def test_five_layers_probabilities_and_shared_aggregation():
    model, inputs = example()
    with patch.object(model,'_centres',wraps=model._centres) as centres:
        result = model(*inputs)
        assert centres.call_count == 5
    n = len(inputs[0])
    assert result['c'].shape == (n,2,6)
    assert torch.allclose(result['p'].sum(1),torch.ones(n,n))
    assert torch.allclose(result['alpha'].sum(2),torch.ones(n,2))
    assert torch.allclose(result['logpi'].exp().sum(1),torch.ones(n))
    mass = inputs[4][:,None,:]*inputs[5][:,None,:]*result['p']
    aggregate = mass.sum(2)[:,:,None]*torch.einsum('ukv,vkd->ukd',result['alpha'],result['zr'])
    expected = torch.nn.functional.normalize(result['zr']+aggregate,dim=-1)
    assert torch.allclose(result['c'],expected)


def test_cross_entropy_reaches_pair_routing_pressure_and_topology():
    model, inputs = example()
    with torch.no_grad():
        model.V.normal_(std=.1)
        model.inflow.normal_()
    result = model(*inputs)
    result['c_route'].retain_grad()
    result['p'].retain_grad()
    target = torch.eye(5)*4+torch.ones(5,5)
    destination_ce(result['score'],target,torch.ones(5,5,dtype=torch.bool)).backward()
    for gradient in (result['c_route'].grad,result['p'].grad,model.pair_origin.grad,
                     model.pair_destination.grad,model.V.grad,model.Bt.grad,
                     model.cs_u.grad[:,9:],model.cs_tau.grad,model.B0.grad,model.beta.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.norm()>0


def test_pair_head_uses_supply_pair_origin_channels_and_od_corrections():
    model, inputs = example()
    with torch.no_grad():
        model.inflow.normal_()
        model.cs_u.normal_(std=.2)
        model.cs_tau.copy_(torch.tensor([-.1,.3]))
        model.B0.copy_(torch.tensor([.4,-.2,.1]))
        model.beta.fill_(.7)
    result = model(*inputs)
    z, h, p = result['z'], result['c'], result['p']
    normalize = torch.nn.functional.normalize
    expected = torch.empty_like(result['score'])
    for i in range(len(z)):
        for j in range(len(z)):
            rule = []
            for k in range(model.K):
                pair = normalize(normalize(z[i,k],dim=0) @ model.pair_origin[k]
                                 + normalize(z[j,k],dim=0) @ model.pair_destination[k],dim=0)
                rule.append((pair @ normalize(h[i,k],dim=0))/model.cs_tau[k].exp()
                            + model.inflow[i] @ model.cs_u[k,9:])
            expected[i,j] = p[i,:,j] @ torch.stack(rule) + model.B0[inputs[2][i,j]] \
                            - torch.nn.functional.softplus(model.beta)*inputs[1][i,j]
    assert torch.allclose(result['score'],expected,atol=1e-6)
    altered_h = h.detach().clone()
    altered_h[1] = torch.randn_like(altered_h[1])
    changed = model.od_pair_scores(z,altered_h,p,inputs[1],inputs[2])
    assert torch.allclose(changed[0],result['score'][0])
    assert not torch.allclose(changed[1],result['score'][1])
    altered_z = z.detach().clone()
    altered_z[1] = torch.randn_like(altered_z[1])
    changed = model.od_pair_scores(altered_z,h,p,inputs[1],inputs[2])
    assert not torch.allclose(changed[0,1],result['score'][0,1])


def test_pressure_changes_destinations_through_existing_channel_probabilities():
    model, inputs = example()
    result = model(*inputs)
    z, h = result['z'], result['c']
    p = torch.zeros_like(result['p'])
    p[:,0,:] = torch.linspace(.1,.9,len(z))[None,:]
    p[:,1,:] = 1-p[:,0,:]
    before = model.od_pair_scores(z,h,p,inputs[1],inputs[2])
    with torch.no_grad():
        model.inflow[:,0].fill_(1)
        model.cs_u[0,9].fill_(2)
    after = model.od_pair_scores(z,h,p,inputs[1],inputs[2])
    assert torch.allclose(after-before,2*p[:,0,:],atol=1e-6)
    assert not torch.allclose(before.softmax(1),after.softmax(1))


def test_self_option_uses_the_same_prediction_head():
    model, inputs = example()
    model.eval()
    before = torch.diag(model(*inputs)['logpi'].exp())
    with torch.no_grad():
        model.B0[0].add_(4)
    after = torch.diag(model(*inputs)['logpi'].exp())
    assert torch.all(after>before)


def test_empty_messages_preserve_finite_local_representation():
    model, inputs = example()
    inputs = list(inputs)
    inputs[4] = torch.zeros_like(inputs[4])
    inputs[5] = torch.zeros_like(inputs[5])
    result = model(*inputs)
    assert torch.isfinite(result['score']).all()
    assert torch.count_nonzero(result['alpha']) == 0
    assert torch.allclose(result['c'],result['zr'],atol=1e-6)


def test_cross_entropy_is_origin_normalized_and_ignores_empty_origins():
    score = torch.tensor([[2.,0.,-1.],[0.,1.,3.],[1.,2.,3.]])
    target = torch.tensor([[7.,3.,0.],[0.,2.,2.],[0.,0.,0.]])
    candidates = torch.tensor([[True,True,False],[False,True,True],[True,True,True]])
    loss = destination_ce(score,target,candidates)
    logpi = torch.log_softmax(score.masked_fill(~candidates,-1e9),1)
    expected = -((target[:2]/target[:2].sum(1,keepdim=True))*logpi[:2]).sum(1).mean()
    assert torch.allclose(loss,expected)
    assert torch.allclose(loss,destination_ce(score,target*100,candidates))


def test_pressure_excludes_within_community_targets():
    context = np.array([[0.,2.,3.],[4.,0.,1.],[1.,2.,0.]])
    raw = np.arange(27).reshape(3,9)+1.
    columns = [f'{indicator}_{tier}' for tier in ('CHC','SH','TH') for indicator in ('RA','SA','TC')]
    expected = pressure_context(context,raw,columns)
    changed = context.copy()
    np.fill_diagonal(changed,10000)
    assert np.allclose(expected,pressure_context(changed,raw,columns))


def test_sparse_pair_centres_and_gradients_match_dense_update():
    model,_ = example()
    torch.manual_seed(9)
    edge = torch.randn(5,5,2,6,requires_grad=True)
    anchor = torch.randn(5,2,6,requires_grad=True)
    weight = torch.tensor([[0.,.4,.6,0,0],[.3,0,0,.7,0],[0,0,0,0,0],[0,0,1.,0,0],[.5,0,.5,0,0]])
    bias = torch.randn(5,5,2)
    oi,di = torch.nonzero(weight>0,as_tuple=True)
    sparse = model._centres(edge[oi,di],anchor,weight[oi,di],oi,bias[oi,di],.5)
    dense = anchor
    for _ in range(7):
        p = torch.softmax(torch.einsum('ijkd,ikd->ijk',edge,dense)/.5+bias,-1)
        dense = torch.nn.functional.normalize(anchor+torch.einsum('ijk,ijkd->ikd',weight[:,:,None]*p,edge),dim=-1)
    assert torch.allclose(sparse,dense,atol=2e-6,rtol=2e-6)
    direction = torch.randn_like(sparse)
    gs = torch.autograd.grad((sparse*direction).sum(),(edge,anchor),retain_graph=True)
    gd = torch.autograd.grad((dense*direction).sum(),(edge,anchor))
    assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(gs,gd))


if __name__ == '__main__':
    import unittest
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name,fn in sorted(globals().items())
                               if name.startswith('test_') and callable(fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
