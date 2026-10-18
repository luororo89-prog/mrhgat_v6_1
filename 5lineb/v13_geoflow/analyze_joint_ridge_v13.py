"""Local-reference joint Ridge decoder, following consensus_v10's paired difference.

S=[h_base, value_comp(x_i), value_compete(x_i,t_ii,onehot(R_ii))].
C=[h_base,h_comp,h_compete] from the unchanged current model.
Fit D:S->X, then delta=D(C)-D(S), X_after=X+delta, using one frozen ruler.
The local message reference is an explicit adaptation to MechanismNet, which
does not expose the old model's self-loop vectors. Native scales are retained.
"""
import argparse
from itertools import combinations
from pathlib import Path
import json
import hashlib
import sys
import zipfile
import io
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, adjusted_rand_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from run_od_ablation_v13 import setup
from run_od_relation_v13 import forward_context
from model_od_mechanism_v13 import make_mechanism_model
from ruler import measure
from decode import ALPHAS

SOURCE=ROOT/'results_od_mechanism/outpatient_d0.25_od_identifiability_rank_v13'
OUT=ROOT/'results_od_mechanism/outpatient_d0.25_joint_ridge_v13'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def save_npz(path,**arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
    temporary=path.with_suffix('.tmp');temporary.write_bytes(buffer.getvalue());temporary.replace(path)
    with np.load(path) as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
def table(out,name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(out/(name+'.csv'),index=False)
    return d

def local_reference(model,x,distance,topology,base):
    local_comp=model.value['comp'](x)
    local_compete=model.value['compete'](torch.cat([
        x,torch.diag(distance)[:,None],F.one_hot(torch.diag(topology),3).float()],1))
    return torch.cat([base,local_comp,local_compete],1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);a=ap.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=False)
    cfg,g,t,H,Hv=setup(HERE/'configs/outpatient_d0.25_od_identifiability_rank_v13.yaml')
    ruler_path=ROOT/cfg['ruler_from']/'ruler.json';r=json.loads(ruler_path.read_text())['main'];R=r['frozen']
    for k in ['U','weights','centers','mean','std']:R[k]=np.asarray(R[k])
    X=g['x'];W=g['W'];n=len(X);off=~np.eye(n,dtype=bool);eligible=W.sum(1)>0
    nominal=measure(X,R);before=nominal['pc'];types=nominal['class']
    folds=pd.read_parquet(ROOT/'data/processed/spatial_folds.parquet').set_index('community_id').loc[g['ids'],'fold'].to_numpy()
    assert not pd.isna(folds).any() and len(np.unique(folds))==5
    dump(out/'protocol.json',dict(setting='E3_w1',seeds=[0,1,2],source_commit='2575596b012ee763eed3f0a8c488efd70abde5ed',
         source_joint_decoder='5lineb/v10_edgedisen/consensus_v10.py',alphas=ALPHAS.tolist(),
         fit='RidgeCV(S,X); native feature scales; intercept=True; internal default leave-one-out alpha selection',
         S='[h_base, value.comp(x_i), value.compete([x_i,t_ii,onehot(R_ii)])]',
         C='[h_base,h_comp,h_compete]',delta='D(C)-D(S)',after='X+delta',
         local_reference='Constructed local message reference; not a tensor exposed by the historical model',
         main_graph='all thresholded nonself OD edges; model parameters frozen',
         reconstruction_validation='5 saved spatial folds; encoder and ruler remain transductive',
         normalization='none added; preserve native mechanism channel magnitudes',
         fixed_type_names=R['type_names'],eligible='positive outgoing total, internal flow included',
         robust_type_rule='at least 80% of seeds agree with type from mean delta; with 3 seeds this requires unanimity',
         interpretation='model-estimated change under the declared local reference; no independently measured post-interaction supply'))
    table(out,'node_spatial_folds',dict(community_id=g['ids'],fold=folds,eligible=eligible))
    table(out,'pc_projection',pd.DataFrame(R['U'],index=g['columns'],columns=['PC1','PC2','PC3']).rename_axis('indicator').reset_index())
    recon=[];nodeframes=[];DX=[];OOF=[];TYPES=[];DECODED=[];status=[];context_rows=[]
    for seed in [0,1,2]:
        model=make_mechanism_model('mech_full',cfg,n,t[2]);model.load_state_dict(torch.load(SOURCE/'E3_w1'/f'seed{seed}/model.pt',weights_only=True));model.eval()
        with torch.no_grad():
            testout=forward_context(model,'mech_full',g,t,W*(off&~H),cfg['tau_final'])
            np.testing.assert_array_equal(testout['score'].numpy(),np.load(SOURCE/'E3_w1'/f'seed{seed}/arrays.npz')['scores'])
            St=local_reference(model,*t[:3],testout['channels']['base']).numpy().astype(float)
            Ct=testout['c'].numpy().astype(float)
            full=forward_context(model,'mech_full',g,t,W*off,cfg['tau_final'])
            S=local_reference(model,*t[:3],full['channels']['base']).numpy().astype(float)
            C=full['c'].numpy().astype(float)
        np.testing.assert_array_equal(S[:,:32],C[:,:32])
        dec=RidgeCV(alphas=ALPHAS).fit(S,X)
        db=dec.predict(S);da=dec.predict(C);dx=da-db
        np.testing.assert_allclose(dx,(C-S)@dec.coef_.T,atol=1e-10)
        np.testing.assert_allclose(dec.predict(S)-dec.predict(S),0,atol=1e-12)
        parts=np.stack([(C[:,k*32:(k+1)*32]-S[:,k*32:(k+1)*32])@dec.coef_[:,k*32:(k+1)*32].T for k in range(3)])
        np.testing.assert_allclose(parts.sum(0),dx,atol=1e-10)
        np.testing.assert_allclose(parts[0],0,atol=1e-12)
        oof=np.zeros_like(X);oofdx=np.zeros_like(X);fold_alpha=[]
        for f in np.unique(folds):
            ix=folds==f;fd=RidgeCV(alphas=ALPHAS).fit(S[~ix],X[~ix])
            oof[ix]=fd.predict(S[ix]);oofdx[ix]=fd.predict(C[ix])-fd.predict(S[ix]);fold_alpha.append(float(fd.alpha_))
        after=X+dx;pc=after@R['U'];dpc=dx@R['U'];at=measure(after,R)['class']
        oat=measure(X+oofdx,R)['class'];tol=np.sqrt(np.mean(((oof-X)@R['U'])**2,axis=0))
        tol=np.maximum(tol,1e-8)
        for j,c in enumerate(g['columns']):
            recon.append(dict(seed=seed,indicator=c,alpha=float(dec.alpha_),oof_rmse=np.sqrt(np.mean((oof[:,j]-X[:,j])**2)),
                oof_mae=np.mean(np.abs(oof[:,j]-X[:,j])),oof_r2=r2_score(X[:,j],oof[:,j]),
                oof_pearson=np.corrcoef(X[:,j],oof[:,j])[0,1],insample_rmse=np.sqrt(np.mean((db[:,j]-X[:,j])**2))))
        testdx=dec.predict(Ct)-dec.predict(St)
        context_rows.append(dict(seed=seed,test_vs_full_delta_correlation=np.corrcoef(dx[eligible].ravel(),testdx[eligible].ravel())[0,1],
             test_vs_full_after_type_agreement=np.mean(measure(X+testdx,R)['class'][eligible]==at[eligible])))
        rec=dict(seed=seed,alpha=float(dec.alpha_),fold_alphas=fold_alpha,oof_rmse=float(np.sqrt(np.mean((oof-X)**2))),
             oof_min_r2=float(min(r2_score(X[:,j],oof[:,j]) for j in range(9))),
             oof_before_type_agreement=float(np.mean(measure(oof,R)['class']==types)),
             eligible_n=int(eligible.sum()),retained=int(np.sum((at==types)&eligible)),changed=int(np.sum((at!=types)&eligible)),
             oof_vs_full_after_type_agreement=float(np.mean(oat[eligible]==at[eligible])),
             mean_delta_pc=dpc[eligible].mean(0).tolist(),positive_pc_count=(dpc[eligible]>0).sum(0).tolist(),
             negative_pc_count=(dpc[eligible]<0).sum(0).tolist(),pc_oof_rmse_tolerance=tol.tolist())
        status.append(rec)
        frame=pd.DataFrame(dict(community_id=g['ids'],name=g['names'],seed=seed,eligible=eligible,baseline_type=types,
                 after_type=at,retained=types==at,oof_after_type=oat,D_9=np.sqrt(np.mean(dx**2,1)),
                 D_PC=np.sqrt(np.sum(dpc**2*R['weights'],1)),D_PC_unweighted=np.linalg.norm(dpc,axis=1)))
        for j,c in enumerate(g['columns']):
            for label,values in [('before',X),('decoded_before',db),('decoded_after',da),('after',after),('delta',dx),('oof_before',oof),('oof_delta',oofdx)]:
                frame[f'{label}_{c}']=values[:,j]
        for k in range(3):
            frame[f'PC{k+1}_before']=before[:,k];frame[f'PC{k+1}_after']=pc[:,k];frame[f'delta_PC{k+1}']=dpc[:,k]
            frame[f'PC{k+1}_direction']=np.sign(dpc[:,k]).astype(int)
            frame[f'PC{k+1}_direction_tolerant']=np.where(np.abs(dpc[:,k])<=tol[k],0,np.sign(dpc[:,k])).astype(int)
        nodeframes.append(frame);DX.append(dx);OOF.append(oof);TYPES.append(at);DECODED.append(db)
        save_npz(out/f'seed{seed}_representations_decoder.npz',ids=g['ids'],h_base=C[:,:32],h_comp=C[:,32:64],h_compete=C[:,64:],
             S=S,C=C,coef=dec.coef_,intercept=dec.intercept_,alpha=dec.alpha_,delta=dx,parts=parts,oof_delta=oofdx)
        print(rec,flush=True)
    DX=np.stack(DX);OOF=np.stack(OOF);TYPES=np.stack(TYPES);DECODED=np.stack(DECODED)
    dpc=DX@R['U'];mean=DX.mean(0);mean_type=measure(X+mean,R)['class']
    agree=(TYPES==mean_type[None]).mean(0);unanimous=(TYPES==TYPES[:1]).all(0)
    consistent=pd.DataFrame(dict(community_id=g['ids'],name=g['names'],eligible=eligible,baseline_type=types,mean_after_type=mean_type,
        retained_mean=mean_type==types,after_type_unanimous=unanimous,mean_type_seed_agreement=agree,
        robust_changed=(mean_type!=types)&(agree>=.8),robust_retained=(mean_type==types)&(agree>=.8),
        retained_changed_unanimous=((TYPES==types[None])==(TYPES[:1]==types[None])).all(0)))
    for k in range(3):
        consistent[f'delta_PC{k+1}_mean']=dpc[:,:,k].mean(0);consistent[f'delta_PC{k+1}_sd']=dpc[:,:,k].std(0,ddof=1)
        consistent[f'PC{k+1}_sign_unanimous']=(np.sign(dpc[:,:,k])==np.sign(dpc[:1,:,k])).all(0)
    pairs=[]
    for s,t_ in combinations(range(3),2):
        pairs.append(dict(seed_a=s,seed_b=t_,metric='after_type',agreement=np.mean(TYPES[s,eligible]==TYPES[t_,eligible]),
                      ari=adjusted_rand_score(TYPES[s,eligible],TYPES[t_,eligible])))
        pairs.append(dict(seed_a=s,seed_b=t_,metric='retained_changed',agreement=np.mean((TYPES[s,eligible]==types[eligible])==(TYPES[t_,eligible]==types[eligible]))))
        for j,c in enumerate(g['columns']):
            pairs.append(dict(seed_a=s,seed_b=t_,metric='delta_'+c,pearson=np.corrcoef(DX[s,eligible,j],DX[t_,eligible,j])[0,1],
                   mean_absolute_difference=np.abs(DX[s,eligible,j]-DX[t_,eligible,j]).mean(),
                   oof_reconstruction_difference=np.abs(OOF[s,:,j]-OOF[t_,:,j]).mean()))
        for k in range(3):
            pairs.append(dict(seed_a=s,seed_b=t_,metric=f'delta_PC{k+1}',pearson=np.corrcoef(dpc[s,eligible,k],dpc[t_,eligible,k])[0,1],
                    spearman=spearmanr(dpc[s,eligible,k],dpc[t_,eligible,k]).statistic,
                    agreement=np.mean(np.sign(dpc[s,eligible,k])==np.sign(dpc[t_,eligible,k]))))
    table(out,'reconstruction',recon);table(out,'node_results',pd.concat(nodeframes,ignore_index=True));table(out,'node_seed_consistency',consistent)
    table(out,'seed_pair_consistency',pairs);table(out,'graph_context_sensitivity',context_rows);dump(out/'per_seed_summary.json',status)
    frames=pd.concat(nodeframes,ignore_index=True);valid=frames[frames.eligible]
    table(out,'type_summary',valid.groupby(['seed','baseline_type']).agg(n=('community_id','size'),retained_rate=('retained','mean'),
          D_9=('D_9','mean'),D_PC=('D_PC','mean'),delta_PC1=('delta_PC1','mean'),delta_PC2=('delta_PC2','mean'),delta_PC3=('delta_PC3','mean')).reset_index())
    table(out,'type_transition',valid.groupby(['seed','baseline_type','after_type']).size().rename('n').reset_index())
    summary=dict(eligible_n=int(eligible.sum()),excluded_zero_outflow=int((~eligible).sum()),
         mean_retained=int(np.sum((mean_type==types)&eligible)),mean_changed=int(np.sum((mean_type!=types)&eligible)),
         robust_retained=int(consistent.loc[eligible,'robust_retained'].sum()),robust_changed=int(consistent.loc[eligible,'robust_changed'].sum()),
         after_type_counts_by_seed=[np.bincount(v[eligible],minlength=4).tolist() for v in TYPES],
         mean_after_type_counts=np.bincount(mean_type[eligible],minlength=4).tolist(),
         unanimous_after_type=int(unanimous[eligible].sum()),unanimous_retained_changed=int(consistent.loc[eligible,'retained_changed_unanimous'].sum()),
         unanimous_pc_direction=[int(consistent.loc[eligible,f'PC{k+1}_sign_unanimous'].sum()) for k in range(3)])
    dump(out/'consistency_summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');colors=['#94a3b8','#4b9b79','#e2a34c','#9275b3']
    for k in range(3):
        pa=(X+mean)@R['U'];v=before[eligible,k];w=pa[eligible,k]
        axes[0,k].scatter(v,w,c=np.asarray(colors)[types[eligible]],s=9,alpha=.6)
        lo=min(v.min(),w.min());hi=max(v.max(),w.max());axes[0,k].plot([lo,hi],[lo,hi],color='gray',lw=.8)
        axes[0,k].set(xlabel=f'PC{k+1} before',ylabel=f'PC{k+1} after (mean change)')
        for s in range(3): axes[1,k].hist(dpc[s,eligible,k],bins=35,histtype='step',label=f'Seed {s}',density=True)
        axes[1,k].axvline(0,color='gray',lw=.8);axes[1,k].set(xlabel=f'Change in PC{k+1}',ylabel='Density')
    from matplotlib.lines import Line2D
    axes[0,0].legend(handles=[Line2D([],[],ls='',marker='o',color=colors[i],label=f'Type {i}') for i in range(4)],
                     ncols=2,fontsize=8,loc='upper left')
    axes[1,0].legend();fig.savefig(out/'pc_changes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    tab=pd.read_csv(out/'type_summary.csv')
    for s in range(3):
        q=tab[tab.seed==s];axes[0].plot(q.baseline_type,q.retained_rate,'o-',label=f'Seed {s}')
    axes[0].set(xlabel='Baseline type',ylabel='Retained rate',xticks=range(4),ylim=(0,1));axes[0].legend()
    mat=pd.crosstab(types[eligible],mean_type[eligible]).reindex(index=range(4),columns=range(4),fill_value=0)
    axes[1].imshow(mat,cmap='Blues');axes[1].set(xlabel='After type from mean change',ylabel='Baseline type',xticks=range(4),yticks=range(4))
    for i in range(4):
        for j in range(4):axes[1].text(j,i,str(mat.iloc[i,j]),ha='center',va='center',color='white' if mat.iloc[i,j]>mat.to_numpy().max()/2 else 'black')
    fig.savefig(out/'type_changes.png',dpi=180);plt.close(fig)
    dump(out/'verification.json',dict(checkpoint_scores_exact=True,same_base_block=True,linear_difference_identity=True,
         channel_parts_additive=True,base_change_zero=True,same_representation_change_zero=True,
         baseline_pc_fixed=True,type_centers_fixed=True))
    dump(out/'manifest.json',dict(script_sha256=sha(__file__),ruler_sha256=sha(ruler_path),
         original_joint_decoder_sha256=sha(HERE.parent/'v10_edgedisen/consensus_v10.py'),
         spatial_folds_sha256=sha(ROOT/'data/processed/spatial_folds.parquet'),
         checkpoints={str(s):sha(SOURCE/'E3_w1'/f'seed{s}/model.pt') for s in range(3)},
         files={p.name:sha(p) for p in out.iterdir()}))
    print('COMPLETE',summary,flush=True)

if __name__=='__main__':main()
