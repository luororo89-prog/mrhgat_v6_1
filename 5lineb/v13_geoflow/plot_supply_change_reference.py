"""Reference-style alluvial and bubble figure from three-seed mean changes."""
from pathlib import Path
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.colors import to_rgb
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results_od_mechanism/outpatient_d0.25_joint_ridge_v13'
parser=argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
args=parser.parse_args()
suffix='' if args.seed is None else f'_seed{args.seed}'
if args.seed is None:
    d=pd.read_csv(OUT/'node_seed_consistency.csv')
else:
    d=pd.read_csv(OUT/'node_results.csv')
    d=d.loc[d.seed.eq(args.seed)].copy()
    d=d.rename(columns={'retained':'retained_mean', **{f'delta_PC{i}':f'delta_PC{i}_mean' for i in (1,2,3)}})
d=d.loc[d.eligible].copy()
assert len(d)==664 and d.community_id.is_unique
colors=['#397bb5','#9464af','#399952','#df4640']
names=['Scarcity','CHC-dominant','TH–SH','TH–CHC']
plt.rcParams.update({'font.family':'Nimbus Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(14,6.1),facecolor='white')
a=fig.add_axes([.025,.13,.46,.76]); b=fig.add_axes([.64,.24,.235,.51]); top=fig.add_axes([.64,.75,.235,.11])
fig.text(.018,.95,'a',weight='bold',fontsize=16); fig.text(.515,.95,'b',weight='bold',fontsize=16)
# Fixed-PC signs; zero changes would need a separate state and must not be silently classified.
pcs=['delta_PC%d_mean'%i for i in (1,2,3)]
assert not (d[pcs]==0).any().any()
states=np.column_stack([d.baseline_type.to_numpy(),*(np.where(d[p]<0,0,1) for p in pcs)])
N=len(d); gap=25; xs=[0,1.3,2.6,3.9]; width=.22
positions=[]
for j in range(4):
    vals=range(4) if j==0 else range(2)
    heights={v:int((states[:,j]==v).sum()) for v in vals}
    total=sum(heights.values())+gap*(len(heights)-1)
    cursor=(N+3*gap-total)/2
    pos={}
    for v in vals:
        pos[v]=(cursor,heights[v]); cursor+=heights[v]+gap
    positions.append(pos)
# Each adjacent layer is split by original configuration, retaining its colour.
for j in range(3):
    left={k:v[0] for k,v in positions[j].items()}; right={k:v[0] for k,v in positions[j+1].items()}
    for t in range(4):
        for u in positions[j]:
            for v in positions[j+1]:
                n=int(((states[:,0]==t)&(states[:,j]==u)&(states[:,j+1]==v)).sum())
                if not n: continue
                y0,y1=left[u],right[v]; left[u]+=n; right[v]+=n
                x0,x1=xs[j]+width,xs[j+1]; dx=(x1-x0)*.46
                verts=[(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),(x1,y1+n),(x1-dx,y1+n),(x0+dx,y0+n),(x0,y0+n),(x0,y0)]
                a.add_patch(PathPatch(MPath(verts,[1,4,4,4,2,4,4,4,79]),facecolor=colors[t],alpha=.44,edgecolor='none'))
for j in range(4):
    for v,(y,n) in positions[j].items():
        a.add_patch(Rectangle((xs[j],y),width,n,facecolor=colors[v] if j==0 else ['#bdbdbd','#4c4c4c'][v],edgecolor='none'))
        if j==0: a.text(-.09,y+n/2,names[v],ha='right',va='center')
        else: a.text(xs[j]+width/2,y+n/2,['Decline','Rise'][v],rotation=90,ha='center',va='center',color='#333333' if v==0 else 'white',fontsize=10)
for j,label in enumerate(['Basic supply\nconfiguration','PC1','PC2','PC3']): a.text(xs[j]+width/2,N+3*gap+33,label,ha='center',va='top')
a.plot([xs[1],xs[1],xs[3]+width,xs[3]+width],[-12,-20,-20,-12],color='black',lw=.8)
a.text((xs[1]+xs[3]+width)/2,-30,'Component change',ha='center',va='bottom',fontsize=12)
a.set_xlim(-1.65,4.25); a.set_ylim(N+3*gap+100,-60); a.axis('off')
patterns=[(1,1,0),(1,1,1),(1,0,1),(1,0,0),(0,1,0),(0,1,1),(0,0,0),(0,0,1)]
b.set(xlim=(-.5,3.5),ylim=(7.5,-.5),xticks=range(4),yticks=range(8))
b.set_xticklabels(names,rotation=31,ha='right'); b.set_yticklabels([])
b.tick_params(length=0,pad=9)
for x in np.arange(-.5,4,1): b.axvline(x,color='#c6c6c6',lw=.8,zorder=0)
for y in np.arange(-.5,8,1): b.axhline(y,color='#c6c6c6',lw=.8,zorder=0)
for sp in b.spines.values(): sp.set_linewidth(1.2)
scale=3.5
rows=[]
for t in range(4):
    g=d[d.baseline_type==t]; retained=int(g.retained_mean.sum()); changed=len(g)-retained
    light=tuple(.55+.45*np.array(to_rgb(colors[t])))
    for k,p in enumerate(patterns):
        mask=np.all(states[:,1:]==p,axis=1)&(states[:,0]==t)
        n=int(mask.sum()); c=int((mask&~d.retained_mean.to_numpy()).sum())
        if n: b.scatter(t,k,s=n*scale,color=light,edgecolors='none',zorder=2)
        if c: b.scatter(t,k,s=c*scale,color=colors[t],edgecolors='none',zorder=3)
        rows.append(dict(baseline_type=t,PC1=p[0],PC2=p[1],PC3=p[2],retained=n-c,changed=c,total=n))
    for dx,count,col in [(-.18,retained,light),(.18,changed,colors[t])]:
        pct=100*count/len(g); top.bar(t+dx,pct,width=.29,color=col)
        label=f'{pct:.2f}%' if 0<pct<100 else f'{pct:.0f}%'
        top.text(t+dx,pct+4,label,ha='center',va='bottom',fontsize=9)
for j in range(3):
    x=-1.65+j*.4
    b.text(x,-.85,f'PC{j+1}',ha='center',va='center',fontsize=10,clip_on=False)
    for k,p in enumerate(patterns): b.scatter(x,k,marker='^' if p[j] else 'v',s=65,color='#202020' if p[j] else '#b5b5b5',clip_on=False)
b.text(-2.07,3.5,'Component change',rotation=90,va='center',ha='center',clip_on=False)
b.set_xlabel('Basic supply configuration',labelpad=10)
top.set(xlim=(-.5,3.5),ylim=(0,125)); top.axis('off')
# Legend disks use area proportional to counts. Dark disks encode changed counts;
# the remaining light area encodes retained counts.
leg=fig.add_axes([.89,.24,.105,.53]); leg.axis('off'); leg.set(xlim=(0,1),ylim=(0,1))
leg.text(0,.98,'Communities',va='top')
for y,n in zip([.86,.75,.62],[10,40,80]): leg.scatter(.12,y,s=n*scale,c='#c7c7c7',edgecolors='none'); leg.text(.37,y,str(n),va='center')
leg.text(0,.49,'Effective type')
for y,c,l in [(.42,'#c7c7c7','Retained'),(.35,'#555555','Changed')]: leg.scatter(.12,y,s=90,c=c);leg.text(.28,y,l,va='center')
leg.text(0,.23,'Component')
for y,m,c,l in [(.16,'^','#202020','Rise'),(.09,'v','#b5b5b5','Decline')]: leg.scatter(.12,y,s=70,c=c,marker=m);leg.text(.28,y,l,va='center')
fig.text(.5,.025,('Three-seed mean supply change' if args.seed is None else f'Seed {args.seed} supply change')+' · Fixed PC projection · 664 communities',ha='center',color='#555555',fontsize=10)
pd.DataFrame(rows).to_csv(OUT/f'supply_change_reference{suffix}_counts.csv',index=False)
for ext in ['png','svg','pdf']: fig.savefig(OUT/f'supply_change_reference{suffix}.{ext}',dpi=300,facecolor='white')
print('Saved reference-style figure. Counts:',d.groupby('baseline_type').retained_mean.agg(['size','sum']).to_dict())
