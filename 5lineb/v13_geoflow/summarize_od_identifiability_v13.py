"""Read-only collection, fixed semantic labels and validation-only selection."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def summarize(root):
    metrics=[];diagnostics=[];groups=[];stability=[];losses=[]
    for setting in sorted(root.glob('E*_w*')):
        seeds={}
        for folder in sorted(setting.glob('seed*')):
            if not (folder/'replay.json').exists():continue
            seed=int(folder.name[4:]);m=json.loads((folder/'metrics.json').read_text())
            m.update(setting=setting.name,seed=seed);metrics.append(m)
            for filename,target in [('mechanism_diagnostics.csv',diagnostics),('quantile_response.csv',groups)]:
                df=pd.read_csv(folder/filename);df['setting']=setting.name;df['seed']=seed;target.append(df)
            h=pd.read_csv(folder/'history.csv')
            losses.append(dict(setting=setting.name,seed=seed,**{c:float(h[c].mean()) for c in ['ce','comp_rank','compete_rank']}))
            seeds[seed]=folder
        for sa,sb in combinations(seeds,2):
            for population in ['validation','test','visible']:
                a=np.load(seeds[sa]/f'gates_{population}.npz');b=np.load(seeds[sb]/f'gates_{population}.npz')
                np.testing.assert_array_equal(a['origin'],b['origin']);np.testing.assert_array_equal(a['destination'],b['destination'])
                for k,name in enumerate(['comp','compete']):
                    qa,qb=a['q'][:,k],b['q'][:,k]
                    ga=pd.read_csv(seeds[sa]/'quantile_response.csv');gb=pd.read_csv(seeds[sb]/'quantile_response.csv')
                    ga=ga[(ga.split==population)&(ga.mechanism==name)]
                    gb=gb[(gb.split==population)&(gb.mechanism==name)]
                    paired=ga.merge(gb,on=['feature','bin'],suffixes=('_a','_b')).dropna(subset=['gate_mean_a','gate_mean_b'])
                    group_mad=float((paired.gate_mean_a-paired.gate_mean_b).abs().mean())
                    rng=np.random.default_rng(456);ia=rng.integers(len(qa),size=32768);ib=rng.integers(len(qa),size=32768)
                    da,db=qa[ia]-qa[ib],qb[ia]-qb[ib]
                    stability.append(dict(setting=setting.name,seed_a=sa,seed_b=sb,population=population,mechanism=name,
                        pearson=float(np.corrcoef(qa,qb)[0,1]),spearman=float(spearmanr(qa,qb).statistic),
                        mad=float(np.abs(qa-qb).mean()),mse=float(((qa-qb)**2).mean()),
                        quantile_response_mad=group_mad,
                        ordering_agreement=float((np.sign(da)==np.sign(db)).mean()),
                        threshold_agreement=float(((qa>.5)==(qb>.5)).mean())))
    if not metrics:return
    M=pd.DataFrame(metrics);D=pd.concat(diagnostics);G=pd.concat(groups);S=pd.DataFrame(stability)
    M.to_csv(root/'metrics_per_seed.csv',index=False);D.to_csv(root/'mechanism_diagnostics.csv',index=False)
    G.to_csv(root/'quantile_response.csv',index=False);S.to_csv(root/'stability_fixed_semantics.csv',index=False)
    pd.DataFrame(losses).to_csv(root/'loss_scales.csv',index=False)
    summary=M.groupby('setting')[['ce','cpc','nrmse','best_val_ce']].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(root/'prediction_summary.csv')
    # Exploratory acceptance thresholds fixed before completed E3 results: direction >.55, own-v-cross
    # ranking advantage >.02, variance >1e-4, each tail saturation <.95,
    # mean validation CE no more than E0 +.05. All three seeds must pass.
    verdict=[];gate_verdict=[];v=D[D.split=='validation']
    for setting,df in v.groupby('setting'):
        ok=len(df)==6
        for seed,ss in df.groupby('seed'):
            if len(ss)!=2:ok=False;continue
            for k,name in enumerate(['comp','compete']):
                own=ss[ss.mechanism==name].iloc[0];other=ss[ss.mechanism!=name].iloc[0]
                alternate='compete' if name=='comp' else 'comp'
                ok &= bool(own['rank_'+name]>.55 and own['rank_'+name]-own['rank_'+alternate]>.02
                           and own['rank_'+name]-other['rank_'+name]>.02
                           and own['variance']>1e-4 and own['low']<.95 and own['high']<.95)
                gate_verdict.append(dict(setting=setting,seed=seed,mechanism=name,
                    direction=bool(own['rank_'+name]>.55),
                    own_vs_cross=bool(own['rank_'+name]-own['rank_'+alternate]>.02),
                    own_vs_other_gate=bool(own['rank_'+name]-other['rank_'+name]>.02),
                    noncollapsed=bool(own['variance']>1e-4 and own['low']<.95 and own['high']<.95)))
            for mech,feature in [('comp','deficit'),('comp','advantage'),('compete','pressure'),('compete','vulnerability')]:
                response=G[(G.setting==setting)&(G.seed==seed)&(G.split=='validation')&
                           (G.mechanism==mech)&(G.feature==feature)].dropna(subset=['gate_mean']).sort_values('bin')
                ok &= bool(len(response)>1 and response.gate_mean.iloc[-1]-response.gate_mean.iloc[0]>.02)
                for row in gate_verdict:
                    if row['setting']==setting and row['seed']==seed and row['mechanism']==mech:
                        row[feature+'_response']=bool(len(response)>1 and response.gate_mean.iloc[-1]-response.gate_mean.iloc[0]>.02)
        base=summary.loc['E0_w0','best_val_ce_mean'] if 'E0_w0' in summary.index else np.nan
        prediction=bool(summary.loc[setting,'best_val_ce_mean']<=base+.05)
        verdict.append(dict(setting=setting,mechanism_validation_pass=bool(ok),validation_prediction_pass=prediction,
                            eligible_E4=bool(ok and prediction and setting.startswith('E3_'))))
    V=pd.DataFrame(verdict);V.to_csv(root/'validation_acceptance.csv',index=False)
    pd.DataFrame(gate_verdict).to_csv(root/'gate_acceptance_per_seed.csv',index=False)
    eligible=V[V.eligible_E4].setting.tolist()
    chosen=min(eligible,key=lambda s:summary.loc[s,'best_val_ce_mean']) if eligible else None
    (root/'selection.json').write_text(json.dumps(dict(E4_source=chosen,
        E4_status='eligible' if chosen else 'not run: E3 has not passed all validation requirements',
        selection_basis='all three seeds, validation only; fixed labels; no test tuning'),indent=2))
    print(summary.to_string());print(V.to_string(index=False))
    plot(root,D,G,S)


def plot(root,D,G,S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings=sorted(D.setting.unique())
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,(mech,feature) in zip(axes.flat,[('comp','deficit'),('comp','advantage'),('compete','pressure'),('compete','vulnerability')]):
        for setting in [s for s in settings if s.startswith('E0_') or s.endswith('_w1')]:
            a=G[(G.split=='validation')&(G.mechanism==mech)&(G.feature==feature)&(G.setting==setting)]
            a=a.groupby('bin').gate_mean.agg(['mean','std'])
            ax.plot(a.index+1,a['mean'],marker='o',label=setting,linewidth=1)
        ax.set(title=mech+' gate vs '+feature,xlabel='Training-defined quantile group',ylabel='Mean gate')
        ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.savefig(root/'mechanism_response.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,name in zip(axes,['comp','compete']):
        a=S[(S.population=='test')&(S.mechanism==name)]
        for idx,setting in enumerate(settings):
            values=a[a.setting==setting].pearson
            ax.scatter(np.full(len(values),idx),values,s=30)
        ax.set_xticks(range(len(settings)),settings,rotation=60,ha='right')
        ax.axhline(0,color='grey',linewidth=1);ax.set(title=name,ylabel='Pearson across seed pairs',ylim=(-1.05,1.05))
    fig.savefig(root/'stability_fixed_semantics.png',dpi=170);plt.close(fig)
    selected=[s for s in settings if s.startswith(('E0_','E3_'))]
    fig,axes=plt.subplots(2,len(selected),figsize=(4*len(selected),6),squeeze=False,layout='constrained')
    for j,setting in enumerate(selected):
        for path in sorted((root/setting).glob('seed*/gates_validation.npz')):
            q=np.load(path)['q']
            for k,name in enumerate(['comp','compete']):
                axes[k,j].hist(q[:,k],bins=np.linspace(0,1,21),density=True,histtype='step',label=path.parent.name)
                axes[k,j].set(title=setting+' '+name,xlabel='Gate value',ylabel='Density',xlim=(0,1))
                axes[k,j].legend(fontsize=8)
    fig.savefig(root/'gate_distributions.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();summarize(args.directory)
