import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from D_mrhgat.model import MRHGAT
from D_mrhgat.stability import hsic, factor_consensus, SeparatedMRHGAT, fit_baselines, fit_group
from D_mrhgat.mechanism_judgment import classify_evidence, judge_anonymous_factors

def test_matching_undoes_permutation_without_truth_and_conserves_flow():
    rng=np.random.default_rng(51)
    F=rng.integers(1,30,(12,12))
    x=rng.normal(size=(12,9))
    r=rng.dirichlet([2,2,2],size=(12,12)).transpose(0,2,1)
    stack=np.stack([r,r[:,[0,2,1]],r,r[:,[0,2,1]],r])
    mean,aligned,info=factor_consensus(stack,F)
    np.testing.assert_allclose(aligned, np.broadcast_to(mean,aligned.shape),atol=1e-12)
    np.testing.assert_allclose((mean*F[:,None,:]).sum(1),F,atol=1e-10)
    assert info['mean_pairwise_matched_cosine'] > .999999
    assert info['truth_used_for_matching'] is False

def test_regularized_model_has_finite_gradients_preserves_probability_and_uses_direction():
    torch.manual_seed(42)
    x=torch.randn(15,9)
    F=torch.randint(0,50,(15,15)).float()
    K=torch.ones(15,15,dtype=torch.long)
    K.fill_diagonal_(0)
    model=MRHGAT(int_channel='none')
    out=model(x,F,K)
    assert not torch.allclose(out['sigma'],model(x,F.T,K)['sigma'])
    torch.testing.assert_close(out['r'].sum(1),torch.ones_like(F))
    loss=-out['logpi'].mean()+.1*hsic(out['factor_embeddings'],torch.ones(15,dtype=torch.bool))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

def test_dependence_uses_only_training_rows():
    z=torch.randn(20,2,6)
    rows=torch.arange(20)<10
    altered=z.clone()
    altered[10:]*=100
    torch.testing.assert_close(hsic(z,rows),hsic(altered,rows))

def test_legacy_encoder_unchanged():
    model=MRHGAT(int_channel='none')
    x=torch.randn(12,9)
    K=torch.ones(12,12,dtype=torch.long)
    for c,z in model.encode(x,K).items():
        expected=torch.nn.functional.normalize(torch.relu(model.projection[c](x)),dim=1)
        torch.testing.assert_close(z,expected)

def test_auxiliary_classifier_does_not_change_md_forward_probabilities():
    torch.manual_seed(8)
    original=MRHGAT(int_channel='none')
    separated=SeparatedMRHGAT(int_channel='none')
    separated.load_state_dict(original.state_dict(),strict=False)
    x=torch.randn(10,9)
    F=torch.randint(0,15,(10,10)).float()
    K=torch.ones(10,10,dtype=torch.long)
    K.fill_diagonal_(0)
    out=separated(x,F,K)
    torch.testing.assert_close(out['r'],original(x,F,K)['r'])
    out['factor_discrimination'].backward()
    assert separated.P['comp'].grad.abs().sum()>0
    assert separated.P['compet'].grad.abs().sum()>0

def test_group_fits_are_independent_per_seed_and_parallel_matches_sequential():
    torch.manual_seed(3)
    n=15
    x=torch.randn(n,9)
    K=torch.randint(0,3,(n,n))
    K.fill_diagonal_(0)
    F=torch.poisson(torch.ones(2,n,n)*5)
    fold=torch.arange(n)%3
    train_rows=torch.stack([fold!=0,fold!=1])
    eval_rows=torch.stack([fold==0,fold==1])
    base=fit_baselines(x,K,F,train_rows,eval_rows,epochs=8)
    assert base['beta'].shape==(2,9) and base['topo'].shape==(2,3)
    for profile in ('plain','hsic','factor_hsic'):
        out,states,info=fit_group(x,K,F,[0,1],base,train_rows,eval_rows,profile=profile,epochs=12)
        assert torch.isfinite(out['pi']).all()
        torch.testing.assert_close(out['pi'].sum(-1),torch.ones(2,n))
        assert len(info['best_epoch'])==2 and len(info['ce'])==2
    out,_,info=fit_group(x,K,F,[0,1],base,train_rows,eval_rows=None,profile='hsic',fixed_epochs=[5,9])
    assert info['best_epoch']==[5,9]
    out_seq,_,_=fit_group(x,K,F,[0,1],base,train_rows,eval_rows,profile='hsic',epochs=10,n_workers=1)
    out_par,_,_=fit_group(x,K,F,[0,1],base,train_rows,eval_rows,profile='hsic',epochs=10,n_workers=2)
    torch.testing.assert_close(out_seq['pi'],out_par['pi'])

def test_mechanism_naming_requires_significance_and_rejects_mixed_factor():
    positive={'beta':.4,'z':3.,'holm_p':.01}
    direction_only={'beta':.4}
    assert classify_evidence([direction_only],[])=='unresolved'
    assert classify_evidence([positive],[])=='compensation_pattern'
    assert classify_evidence([],[positive])=='competition_pattern'
    assert classify_evidence([positive],[positive])=='mixed_unresolved'

def test_anonymous_search_corrects_all_tests_and_names_either_index():
    rows=[]
    for channel in (1,2):
        for hypothesis in ('compensation','competition'):
            match=(hypothesis=='competition') if channel==1 else (hypothesis=='compensation')
            for topology in ('ADJ','CRS'):
                rows.append(dict(factor_index=channel,hypothesis=hypothesis,topology=topology,
                                 beta=.4 if match else -.4,z=4 if match else -4,
                                 p=.0001 if match else .9999,valid=True))
    result=judge_anonymous_factors(rows)
    assert [r['mechanism_label'] for r in result]==['competition_pattern','compensation_pattern']
    assert result[0]['tests'][2]['p_holm']==.0008
    rows[0]['valid']=False
    assert judge_anonymous_factors(rows)[0]['mechanism_label']=='unresolved'
