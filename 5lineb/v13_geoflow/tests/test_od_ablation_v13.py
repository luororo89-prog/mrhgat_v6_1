import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_od_ablation_v13 import make_ablation_model, VARIANTS
from run_od_relation_v13 import make_model, destination_ce


def example(variant='original', old=False):
    torch.manual_seed(4)
    n = 5
    topology = torch.ones(n, n, dtype=torch.long)
    topology.fill_diagonal_(0)
    cfg = dict(hid_dim=12, iters=7, routing_layers=5, dropout=0.2, attention=True)
    model = (make_model('od_relation', cfg, n, topology) if old else
             make_ablation_model(variant, cfg, n, topology))
    state = torch.get_rng_state().clone()
    x, distance = torch.randn(n, 9), torch.rand(n, n)
    context = torch.tensor([[0,2,0,3,0],[0,0,4,0,1],[2,0,0,2,0],[0,3,1,0,0],[1,0,0,2,0.]])
    share = context / context.sum(1, keepdim=True)
    inputs = (x, distance, topology, torch.ones(n,n,dtype=torch.bool), share, context>0, .5)
    return model, inputs, state


def _check_common_initialization_rng_and_routing(variant):
    reference, inputs, expected_rng = example(old=True)
    model, _, actual_rng = example(variant)
    assert torch.equal(expected_rng, actual_rng)
    assert model.iters == 7 and model.routing_layers == 5 and model.K == 2
    state = model.state_dict()
    for key, value in reference.state_dict().items():
        assert torch.equal(state[key], value), key
    reference.train(); model.train()
    torch.manual_seed(89); expected = reference(*inputs)
    torch.manual_seed(89); actual = model(*inputs)
    for key in ('p', 'zr', 'c_route', 'alpha', 'c'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(actual['score'], expected['score'], rtol=0, atol=0)


def test_common_initialization_rng_and_routing_are_matched():
    for variant in VARIANTS:
        _check_common_initialization_rng_and_routing(variant)


def test_original_optimizer_step_exactly_reproduces_reference():
    reference, inputs, _ = example(old=True)
    model, _, _ = example()
    for net in (reference, model):
        opt = torch.optim.Adam(net.parameters(), lr=.02, weight_decay=.003)
        torch.manual_seed(10)
        output = net(*inputs)
        target = torch.eye(5)*4 + torch.ones(5,5)
        destination_ce(output['score'], target, torch.ones(5,5,dtype=torch.bool)).backward()
        opt.step()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def _check_independent_layers_and_ce_gradients(variant):
    model, inputs, _ = example(variant)
    model.eval()
    with torch.no_grad():
        model.V.normal_(std=.1); model.inflow.normal_()
    mappings = [model.pair_projections(layer) for layer in range(5)]
    assert len({v.data_ptr() for pair in mappings for v in pair}) == 10
    with patch.object(model, '_centres', wraps=model._centres) as centres:
        output = model(*inputs)
        assert centres.call_count == 5
    output['p'].retain_grad()
    target = torch.eye(5)*4 + torch.ones(5,5)
    destination_ce(output['score'], target, torch.ones(5,5,dtype=torch.bool)).backward()
    gradients = [v.grad for pair in mappings for v in pair]
    gradients += [model.V.grad, model.Bt.grad, output['p'].grad]
    for gradient in gradients:
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_independent_layers_have_distinct_storage_and_receive_ce_gradients():
    _check_independent_layers_and_ce_gradients('fix1')


if __name__ == '__main__':
    torch.set_num_threads(2)
    tests = [(name, value) for name, value in globals().copy().items()
             if name.startswith('test_') and callable(value)]
    for name, test in tests:
        test()
        print('PASS', name, flush=True)
    print(f'{len(tests)} test functions passed (including all variant combinations).')
