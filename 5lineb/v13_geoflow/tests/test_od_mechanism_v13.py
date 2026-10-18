import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_od_mechanism_v13 import make_mechanism_model, MECHANISMS
from run_od_relation_v13 import destination_ce, pressure_context

N = 6
CFG = dict(hid_dim=12, iters=7, dropout=0.0, gravity_mlp=8)


def example(variant='mech_full', seed=3):
    torch.manual_seed(seed)
    model = make_mechanism_model(variant, CFG, N)
    rng = np.random.default_rng(seed)
    x = torch.randn(N, 9)
    distance = torch.rand(N, N)
    topology = torch.as_tensor(rng.integers(0, 3, (N, N)))
    context = np.array([[0, 3, 0, 2, 0, 1], [1, 0, 4, 0, 0, 0], [2, 0, 0, 2, 0, 3],
                        [0, 3, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 5, 0]], dtype=float)
    internal = np.array([5., 0., 2., 7., 1., 3.])
    raw = np.abs(rng.normal(size=(N, 9)))
    columns = [f'{m}_{t}' for m in ('RA', 'SA', 'TC') for t in ('CHC', 'SH', 'TH')]
    return model, x, distance, topology, context, internal, raw, columns


def run(model, x, distance, topology, context, internal, raw, columns, train=False):
    model.inflow.copy_(torch.as_tensor(pressure_context(context, raw, columns)))
    model.set_context(context, internal)
    model.train(train)
    c = torch.as_tensor(context, dtype=torch.float32)
    share = c / c.sum(1, keepdim=True).clamp_min(1e-12)
    return model(x, distance, topology, torch.ones(N, N, dtype=torch.bool), share, c > 0, 0.5)


def test_shapes_and_independent_sigmoid_gates():
    model, *data = example()
    out = run(model, *data)
    m = int((data[3] > 0).sum())
    assert out['score'].shape == (N, N) and out['p'].shape == (N, 2, N)
    assert out['c'].shape == (N, 3 * model.d) and out['s'].shape == (N, 2)
    assert out['q_edge'].shape == (m, 2) and out['a_edge'].shape == (m, 2)
    q = out['q_edge']
    assert ((q > 0) & (q < 1)).all()
    assert not torch.allclose(q.sum(1), torch.ones(m), atol=1e-3)          # not a two-class softmax
    assert set(out['channels']) == {'base', 'comp', 'compete'}


def test_both_mechanisms_can_be_active_on_one_od():
    model, *data = example()
    with torch.no_grad():
        for c in MECHANISMS:
            model.gate[c][2].bias.fill_(20.0)
    q = run(model, *data)['q_edge']
    assert (q > 0.99).all()


def test_base_channel_isolated_from_neighbours_and_no_self_messages():
    model, x, distance, topology, context, internal, raw, columns = example()
    out = run(model, x, distance, topology, context, internal, raw, columns)
    oi, di = out['edge_index']
    assert (oi != di).all()
    x2 = x.clone()
    x2[1] += 5.0                                                     # destination of origin 0 changes
    out2 = run(model, x2, distance, topology, context, internal, raw, columns)
    torch.testing.assert_close(out2['channels']['base'][0], out['channels']['base'][0])
    assert not torch.allclose(out2['channels']['comp'][0], out['channels']['comp'][0])


def test_masked_edges_do_not_enter_aggregation_or_attention_denominator():
    model, x, distance, topology, context, internal, raw, columns = example()
    out = run(model, x, distance, topology, context, internal, raw, columns)
    oi = out['edge_index'][0]
    for k in range(2):
        sums = torch.zeros(N).index_add(0, oi, out['a_edge'][:, k])
        has = torch.as_tensor((context > 0).any(1))
        torch.testing.assert_close(sums[has], torch.ones(int(has.sum())))
    x2 = x.clone()
    x2[2] += 5.0                                                     # 2 is no message destination of origin 0
    out2 = run(model, x2, distance, topology, context, internal, raw, columns)
    for c in MECHANISMS:
        torch.testing.assert_close(out2['channels'][c][0], out['channels'][c][0])
    assert torch.equal(out['channels']['comp'][4], torch.zeros(model.d))   # origin without messages


def test_destination_loss_reaches_gates_attention_and_channel_maps():
    model, *data = example()
    out = run(model, *data, train=True)
    target = torch.rand(N, N)
    destination_ce(out['score'], target, torch.ones(N, N, dtype=torch.bool)).backward()
    params = dict(model.named_parameters())
    for c in MECHANISMS:
        for name in (f'gate.{c}.0.weight', f'att_s.{c}.weight', f'att_a.{c}', f'value.{c}.weight', f'propensity.{c}.weight'):
            assert params[name].grad is not None and params[name].grad.abs().sum() > 0, name


def test_channel_ablations_remove_external_vectors():
    for variant, kept in (('mech_base_only', ()), ('mech_base_comp', ('comp',)), ('mech_base_compete', ('compete',))):
        model, *data = example(variant)
        out = run(model, *data)
        for c in MECHANISMS:
            zero = torch.equal(out['channels'][c], torch.zeros_like(out['channels'][c]))
            assert zero == (c not in kept), (variant, c)
    model, *data = example()
    model.zero_external = True
    out = run(model, *data)
    for c in MECHANISMS:
        assert torch.equal(out['channels'][c], torch.zeros_like(out['channels'][c]))


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print('PASS', t.__name__)
