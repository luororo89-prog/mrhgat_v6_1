"""Frozen-ruler decoder audit and directed functional-community description.

Run from repository root. Existing checkpoints and inputs are read only.
Node holdouts concern the decoder only; the pretrained graph is transductive.
Decoded differences are reconstruction diagnostics until independently validated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import importlib.metadata
import warnings

import numpy as np
import pandas as pd
import torch
import igraph as ig
import leidenalg as la
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_od_ablation_v13 import setup
from model_od_mechanism_v13 import make_mechanism_model
from run_od_relation_v13 import forward_context, evaluate_scores
from ruler import measure

SOURCE = ROOT / 'results_od_mechanism/outpatient_d0.25_od_identifiability_rank_v13'
DEFAULT_OUT = ROOT / 'results_od_mechanism/outpatient_d0.25_supply_followup_v13'
SOURCE_COMMIT = '3463723e8fe425dc40e094a1ab0f5493032bbf3f'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(out, name, rows):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(out / (name + '.csv'), index=False)
    return df


def block_ci(values, groups, repetitions=2000):
    """Equal node estimand; resample whole directed communities."""
    rng = np.random.default_rng(731)
    blocks = [np.asarray(values)[groups == k] for k in np.unique(groups)]
    means = [np.concatenate([blocks[k] for k in rng.integers(len(blocks), size=len(blocks))]).mean()
             for _ in range(repetitions)]
    return np.quantile(means, [.025, .975]).tolist()


def partition(W, resolution, seed):
    i, j = np.nonzero(W)
    graph = ig.Graph(n=len(W), edges=list(zip(i.tolist(), j.tolist())), directed=True)
    p = la.find_partition(graph, la.RBConfigurationVertexPartition,
                          weights=W[i, j].tolist(), resolution_parameter=resolution,
                          seed=seed, n_iterations=-1)
    # Stable display numbering by lowest source node index, without changing membership.
    labels = np.asarray(p.membership)
    order = sorted(np.unique(labels), key=lambda k: np.flatnonzero(labels == k)[0])
    labels = np.array([{k: v for v, k in enumerate(order)}[k] for k in labels])
    return labels, float(p.quality())


def community_analysis(g, X, pc, types, out):
    W = g['W']; n = len(W)
    main, quality = partition(W, 1., 0)
    rows, assignments = [], []
    for gamma in (.75, 1., 1.25):
        for seed in range(5):
            labels, q = partition(W, gamma, seed)
            rows.append(dict(resolution=gamma, seed=seed, communities=len(np.unique(labels)),
                             active_communities=len(np.unique(labels[(W.sum(0)+W.sum(1))>0])),
                             threshold=g['theta'],quality=q, ari_main=adjusted_rand_score(main, labels)))
            assignments.extend(dict(community_id=int(g['ids'][i]), resolution=gamma, seed=seed,
                                    functional_community=int(labels[i])) for i in range(n))
    for threshold in (1, 3):
        wt = g['W_raw'] * (g['W_raw'] >= threshold)
        labels, q = partition(wt, 1., 0)
        rows.append(dict(resolution=1., seed=0, threshold=threshold, communities=len(np.unique(labels)),
                         active_communities=len(np.unique(labels[(wt.sum(0)+wt.sum(1))>0])),
                         quality=q, ari_main=adjusted_rand_score(main, labels)))
    table(out, 'community_sensitivity', rows)
    table(out, 'community_assignments_sensitivity', assignments)
    groups = np.unique(main)
    between = np.array([[W[np.ix_(main == a, main == b)].sum() for b in groups] for a in groups])
    table(out, 'functional_community_flows', [dict(origin=int(a), destination=int(b), flow=between[a,b])
                                             for a in groups for b in groups])
    population = pd.read_parquet(ROOT/'data/processed/communities_candidate.parquet').set_index('community_id').loc[g['ids'], '联通人'].to_numpy()
    stats = []
    for k in groups:
        mask = main == k
        counts = np.bincount(types[mask], minlength=4); p = counts / mask.sum()
        external = between[k].copy(); external[k] = 0
        incoming = between[:, k].copy(); incoming[k] = 0
        ep = external / external.sum() if external.sum() else external
        ip = incoming / incoming.sum() if incoming.sum() else incoming
        total = W[mask].sum(); internal = between[k,k]
        row = dict(functional_community=int(k), n=int(mask.sum()), population=population[mask].sum(),
                   shannon=-np.sum(p[p>0]*np.log(p[p>0])), simpson=1-np.sum(p*p),
                   richness=int((counts>0).sum()), type_concentration=np.sum(p*p),
                   internal_flow=internal, internal_share=internal/total if total else np.nan,
                   cross_share=external.sum()/total if total else np.nan,
                   external_outflow=external.sum(), external_inflow=incoming.sum(),
                   external_destination_hhi=np.sum(ep*ep), external_origin_hhi=np.sum(ip*ip),
                   top_external_share=ep.max())
        for t in range(4): row.update({f'type{t}_n':counts[t], f'type{t}_share':p[t]})
        for j in range(3): row.update({f'PC{j+1}_mean':pc[mask,j].mean(),f'PC{j+1}_sd':pc[mask,j].std()})
        for j,c in enumerate(g['columns']): row.update({c+'_mean_z':X[mask,j].mean(),c+'_sd_z':X[mask,j].std()})
        stats.append(row)
    stats = table(out, 'functional_communities', stats)
    node = pd.DataFrame(dict(community_id=g['ids'], name=g['names'], lon=g['lon'], lat=g['lat'],
                             population=population, baseline_type=types, functional_community=main,
                             outflow=W.sum(1), inflow=W.sum(0),
                             within_share=np.divide(np.diag(W),W.sum(1),out=np.zeros(n),where=W.sum(1)>0),
                             external_community_share=np.array([W[i,main!=main[i]].sum()/W[i].sum() if W[i].sum() else np.nan for i in range(n)])))
    table(out, 'node_network_baseline', node)
    return main, stats, node


def fit_decoder(h, X, train, val, use_mlp=False):
    candidates=[]; trials=[]
    for alpha in (1e-4, .001, .01, .1, 1., 10., 100.):
        model=make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(h[train],X[train]); error=np.mean((model.predict(h[val])-X[val])**2)
        candidates.append((error, 'ridge', alpha, model))
        trials.append(dict(family='ridge',alpha=alpha,val_mse=float(error),converged=True))
    if use_mlp:
        for alpha in (.001, .1):
            model=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), activation='relu',
                                solver='lbfgs',alpha=alpha,max_iter=5000,max_fun=100000,random_state=19))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(h[train],X[train])
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            error=np.mean((model.predict(h[val])-X[val])**2)
            trials.append(dict(family='mlp',alpha=alpha,val_mse=float(error),converged=converged))
            if converged: candidates.append((error,'mlp',alpha,model))
    return min(candidates,key=lambda z:z[0]), trials


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args(); out=args.output
    out.mkdir(parents=True,exist_ok=False)
    cfg,g,tensors,H,Hv=setup(HERE/'configs/outpatient_d0.25_od_identifiability_rank_v13.yaml')
    ruler_path=ROOT/cfg['ruler_from']/'ruler.json'
    report=json.loads(ruler_path.read_text()); frozen=report['main']['frozen']
    for k in ('U','weights','centers','mean','std'): frozen[k]=np.asarray(frozen[k])
    X=g['x']; W=g['W']; n=len(W); off=~np.eye(n,dtype=bool)
    saved=np.load(SOURCE/'fixed_split.npz')
    for key,arr in dict(ids=g['ids'],W=W,H=H,Hv=Hv).items(): np.testing.assert_array_equal(saved[key],arr)
    before=measure(X,frozen); pc=before['pc']; types=before['class']
    assert pc.shape==(n,3) and len(frozen['centers'])==4
    protocol=dict(source_commit=SOURCE_COMMIT,setting='E3_w1',model_seeds=[0,1,2],decoder_split_seed=731,
                  node_split=[.6,.2,.2],decoder_validation='transductive node holdout, not unseen-node GNN validation',
                  adequate_validation_rmse_max=.5,adequate_validation_each_r2_min=0.,
                  incremental_validation_mse_gain_min=.01,incremental_test_block_ci_lower_min=0.,
                  thresholds_status='exploratory operational criteria, declared before decoding',
                  community_algorithm='Leiden directed RBConfiguration',resolution=1.,community_seed=0,
                  self_flows='included in directed partition and internal flow; excluded from external messages',
                  type_names=frozen['type_names'],supply_units='frozen transformed standardized indicators',
                  interpretation='Reconstruction targets are baseline indicators, not measured post-interaction supply. Passing this audit is necessary but insufficient for substantive supply-change inference.')
    dump(out/'protocol.json',protocol)
    main_groups,stats,node=community_analysis(g,X,pc,types,out)
    train,other=train_test_split(np.arange(n),test_size=.4,random_state=731,stratify=types)
    val,test=train_test_split(other,test_size=.5,random_state=732,stratify=types[other])
    split=np.full(n,'train',dtype=object);split[val]='validation';split[test]='test'
    table(out,'decoder_node_split',dict(community_id=g['ids'],split=split,baseline_type=types))
    table(out,'pc_loadings',pd.DataFrame(dict(indicator=g['columns'],**{f'PC{k+1}_projection':frozen['U'][:,k] for k in range(3)},
                        **{f'PC{k+1}_loading':[report['main']['loadings'][f'PC{k+1}'][c] for c in g['columns']] for k in range(3)})))
    performance=[];selection=[];replay=[];audit=[];nodes=[];matched=[]
    for seed in (0,1,2):
        path=SOURCE/'E3_w1'/f'seed{seed}'
        model=make_mechanism_model('mech_full',cfg,n,tensors[2])
        model.load_state_dict(torch.load(path/'model.pt',weights_only=True));model.eval()
        with torch.no_grad():
            result=forward_context(model,'mech_full',g,tensors,W*(off & ~H),cfg['tau_final'])
        actual=result['score'].numpy(); expected=np.load(path/'arrays.npz')
        np.testing.assert_array_equal(actual,expected['scores'])
        np.testing.assert_array_equal(result['c'].numpy(),expected['h'])
        metrics=evaluate_scores(result['score'],W,H,(W>0)&off&~H)
        old=json.loads((path/'metrics.json').read_text())
        for key in ('ce','cpc','nrmse'): assert abs(metrics[key]-old[key])<1e-12
        replay.append(dict(seed=seed,scores_exact=True,embeddings_exact=True,checkpoint_sha256=sha(path/'model.pt'),
                           **{k:metrics[k] for k in ('ce','cpc','nrmse')}))
        h=result['c'].numpy().astype(float);base=result['channels']['base'].numpy().astype(float)
        models={};pred={};val_mse={}
        linear={name:fit_decoder(features,X,train,val) for name,features in [('full',h),('base',base)]}
        compare_mlp=any(np.sqrt(v[0][0])>.5 or np.min(r2_score(X[val],v[0][3].predict(features[val]),multioutput='raw_values'))<=0
                        for name,features in [('full',h),('base',base)] for v in [linear[name]])
        for name,features in [('full',h),('base',base)]:
            lb=linear[name][0]; ly=lb[3].predict(features)
            matched.append(dict(seed=seed,representation=name,family='ridge',alpha=lb[2],
                                val_rmse=np.sqrt(lb[0]),test_rmse=np.sqrt(np.mean((ly[test]-X[test])**2))))
        for name,features in [('full',h),('base',base)]:
            best,trials=linear[name]
            y=best[3].predict(features)
            if compare_mlp:
                best,trials=fit_decoder(features,X,train,val,use_mlp=True);y=best[3].predict(features)
            selection.extend(dict(seed=seed,representation=name,**r) for r in trials)
            models[name]=best[3];pred[name]=y;val_mse[name]=best[0]
            for label,ix in [('validation',val),('test',test)]:
                for j,c in enumerate(g['columns']):
                    err=y[ix,j]-X[ix,j]
                    corr=np.corrcoef(y[ix,j],X[ix,j])[0,1] if y[ix,j].std()>0 and X[ix,j].std()>0 else np.nan
                    performance.append(dict(seed=seed,representation=name,split=label,indicator=c,
                                            family=best[1],alpha=best[2],mae=np.abs(err).mean(),rmse=np.sqrt((err**2).mean()),
                                            r2=r2_score(X[ix,j],y[ix,j]),pearson=corr))
            if best[1]=='ridge':
                scaler,reg=best[3].steps[0][1],best[3].steps[1][1]
                np.savez_compressed(out/f'decoder_seed{seed}_{name}.npz',mean=scaler.mean_,scale=scaler.scale_,
                                    coef=reg.coef_,intercept=reg.intercept_,alpha=best[2])
            else:
                scaler,reg=best[3].steps[0][1],best[3].steps[1][1]
                np.savez_compressed(out/f'decoder_seed{seed}_{name}.npz',mean=scaler.mean_,scale=scaler.scale_,
                    family='mlp',activation=reg.activation,n_iter=reg.n_iter_,alpha=best[2],
                    **{f'coef_{k}':v for k,v in enumerate(reg.coefs_)},
                    **{f'intercept_{k}':v for k,v in enumerate(reg.intercepts_)})
        after=pred['full']; hz=h.copy();hz[:,base.shape[1]:]=0
        local_same=models['full'].predict(hz)
        delta=after-X; delta_pc=delta@frozen['U']
        val_r2=r2_score(X[val],after[val],multioutput='raw_values')
        gain=1-val_mse['full']/val_mse['base']
        mse_gain=((pred['base'][test]-X[test])**2).mean(1)-((after[test]-X[test])**2).mean(1)
        lo,hi=block_ci(mse_gain,main_groups[test])
        adequate=bool(np.sqrt(val_mse['full'])<=.5 and val_r2.min()>0)
        increment=bool(gain>.01 and lo>0)
        audit.append(dict(seed=seed,val_rmse_full=float(np.sqrt(val_mse['full'])),val_rmse_base=float(np.sqrt(val_mse['base'])),
                          test_rmse_full=float(np.sqrt(np.mean((after[test]-X[test])**2))),
                          test_rmse_base=float(np.sqrt(np.mean((pred['base'][test]-X[test])**2))),
                          val_min_r2=float(val_r2.min()),val_relative_mse_gain=float(gain),
                          test_mse_gain=float(mse_gain.mean()),test_gain_ci_low=lo,test_gain_ci_high=hi,
                          reconstruction_adequate=adequate,external_increment=increment,
                          substantive_supply_change_supported=False))
        after_type=measure(after,frozen)['class'];retained=types==after_type
        dpc=np.sqrt(np.sum(delta_pc**2*frozen['weights'],1));d9=np.sqrt(np.mean(delta**2,1))
        frame=node.copy();frame['seed']=seed;frame['split']=split
        frame['after_type_diagnostic']=after_type;frame['retained_diagnostic']=retained
        frame['D_PC_diagnostic']=dpc;frame['D_PC_unweighted_diagnostic']=np.linalg.norm(delta_pc,axis=1)
        frame['D_9_diagnostic']=d9;frame['R_stability_diagnostic']=-dpc
        for j,c in enumerate(g['columns']):
            frame[c+'_before_z']=X[:,j];frame[c+'_after_diagnostic_z']=after[:,j]
            frame[c+'_same_decoder_base_z']=local_same[:,j];frame[c+'_delta_diagnostic_z']=delta[:,j]
        tolerance=np.sqrt(np.mean(delta_pc[val]**2,0))
        for j in range(3):
            frame[f'PC{j+1}_before']=pc[:,j];frame[f'PC{j+1}_after_diagnostic']=after@frozen['U'][:,j]
            frame[f'delta_PC{j+1}_diagnostic']=delta_pc[:,j]
            frame[f'PC{j+1}_validation_rmse_tolerance']=tolerance[j]
            frame[f'PC{j+1}_direction_diagnostic']=np.where(np.abs(delta_pc[:,j])<=tolerance[j],0,np.sign(delta_pc[:,j]))
        nodes.append(frame)
        print('seed',seed,audit[-1],flush=True)
    dump(out/'checkpoint_replay.json',replay);dump(out/'decoder_audit.json',audit)
    table(out,'decoder_performance',performance);table(out,'decoder_selection',selection)
    table(out,'decoder_matched_linear',matched)
    allnodes=pd.concat(nodes,ignore_index=True);table(out,'node_supply_diagnostics',allnodes)
    # Diagnostics remain explicitly labelled; they are not reported as measured supply change.
    grouped=allnodes.groupby(['seed','baseline_type','retained_diagnostic']).agg(n=('community_id','size'),
                D_PC=('D_PC_diagnostic','mean'),D_9=('D_9_diagnostic','mean')).reset_index()
    table(out,'type_retention_diagnostics',grouped)
    trans=allnodes.groupby(['seed','baseline_type','after_type_diagnostic']).size().rename('n').reset_index()
    trans['row_share']=trans.n/trans.groupby(['seed','baseline_type']).n.transform('sum')
    table(out,'type_transition_diagnostics',trans)
    table(out,'functional_community_diagnostics',allnodes.groupby(['seed','functional_community']).agg(
        n=('community_id','size'),retained_rate=('retained_diagnostic','mean'),D_PC=('D_PC_diagnostic','mean'),
        D_PC_median=('D_PC_diagnostic','median'),D_PC_p90=('D_PC_diagnostic',lambda x:x.quantile(.9)),
        D_9=('D_9_diagnostic','mean')).reset_index())
    table(out,'community_type_retention_diagnostics',allnodes.groupby(['seed','functional_community','baseline_type']).agg(
        n=('community_id','size'),retained_rate=('retained_diagnostic','mean'),D_PC=('D_PC_diagnostic','mean')).reset_index())
    summary=allnodes.groupby('community_id').agg(D_PC_mean=('D_PC_diagnostic','mean'),D_PC_sd=('D_PC_diagnostic','std'),
        D_9_mean=('D_9_diagnostic','mean'),retained_rate_across_seeds=('retained_diagnostic','mean'),
        after_type_unique=('after_type_diagnostic','nunique')).reset_index()
    table(out,'node_seed_stability_diagnostics',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,color in [('base','#64748b'),('full','#0f766e')]:
        axs[0].plot([0,1,2],[r['test_rmse_'+name] for r in audit],'o-',label=name,color=color)
    axs[0].set(xlabel='Model seed',ylabel='Held-out reconstruction RMSE',xticks=[0,1,2]);axs[0].legend()
    axs[1].errorbar([0,1,2],[a['test_mse_gain'] for a in audit],
        yerr=np.array([[a['test_mse_gain']-a['test_gain_ci_low'] for a in audit],
                       [a['test_gain_ci_high']-a['test_mse_gain'] for a in audit]]),fmt='o',capsize=4,color='#0f766e')
    axs[1].axhline(0,color='gray',lw=1);axs[1].set(xlabel='Model seed',ylabel='Base MSE minus full MSE',xticks=[0,1,2])
    fig.suptitle('Decoder audit · community block bootstrap 95% intervals');fig.savefig(out/'decoder_audit.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained'); bottom=np.zeros(len(stats))
    colors=['#94a3b8','#4b9b79','#e2a34c','#9275b3']
    for t in range(4):
        v=stats[f'type{t}_share'].to_numpy();axs[0].bar(stats.functional_community,v,bottom=bottom,label=f'Type {t}',color=colors[t]);bottom+=v
    axs[0].set(xlabel='Directed functional community',ylabel='Baseline type share');axs[0].legend(ncols=4,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False)
    axs[1].scatter(stats.cross_share,stats.shannon,s=stats.n*2,c=stats.functional_community,cmap='tab10',vmin=0,vmax=9)
    for _,r in stats.iterrows(): axs[1].annotate(str(int(r.functional_community)),(r.cross_share,r.shannon),xytext=(4,4),textcoords='offset points')
    axs[1].set(xlabel='Cross-community flow share',ylabel='Shannon diversity')
    fig.savefig(out/'community_composition.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,label,cmap in [(axs[0],main_groups,matplotlib.colors.ListedColormap(plt.get_cmap('tab10').colors[:len(stats)])),(axs[1],types,matplotlib.colors.ListedColormap(colors))]:
        # Directed strongest outgoing backbone per origin, solely for visualization.
        for i in range(n):
            row=W[i].copy();row[i]=0;j=int(row.argmax())
            if row[j]>0:
                ax.annotate('',xy=(g['lon'][j],g['lat'][j]),xytext=(g['lon'][i],g['lat'][i]),
                            arrowprops=dict(arrowstyle='->',lw=.25,alpha=.16,color='#475569'))
        ax.scatter(g['lon'],g['lat'],c=label,cmap=cmap,s=7,zorder=3)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([],[],marker='o',lw=0,color=cmap.colors[k],label=str(k)) for k in range(len(cmap.colors))],
                  title='Community ID' if ax is axs[0] else 'Baseline type',ncols=4,loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False)
        ax.set(xlabel='Longitude',ylabel='Latitude');ax.set_aspect(1/np.cos(np.deg2rad(np.mean(g['lat']))))
    axs[0].set_title('Directed functional communities');axs[1].set_title('Frozen baseline types')
    fig.savefig(out/'directed_community_map.png',dpi=180);plt.close(fig)
    manifest=dict(source_commit=SOURCE_COMMIT,inputs={str(p.relative_to(ROOT)):sha(p) for p in
        [ruler_path,Path(cfg['supply_file']),Path(cfg['flow_file']),Path(cfg['adjacency_file']),SOURCE/'fixed_split.npz']},
        software={m:importlib.metadata.version(m) for m in ['numpy','pandas','torch','scikit-learn','igraph','leidenalg']},
        script_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    dump(out/'manifest.json',manifest)
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
