"""Weak mechanism references and matched ranking, without changing the network.

All nine frozen-ruler coordinates are benefit indicators. Zero is the saved
city reference, not a clinical adequacy threshold. References are not labels.
Competition means origin pressure associated with outward substitution.
"""
import numpy as np
import torch
from scipy.special import expit
from torch.nn import functional as F


class MechanismReference:
    def __init__(self, g, distance, topology, training_context, margin=0.1, bins=10):
        self.x = np.asarray(g['x'])
        self.raw = np.asarray(g['raw'])
        self.columns = g['columns']
        self.internal = np.diag(g['W']).copy()
        self.distance = np.asarray(distance)
        self.topology = np.asarray(topology)
        self.margin = margin
        self.n = len(self.x)
        self.ra = self.raw[:, [self.columns.index('RA_'+t) for t in ('CHC','SH','TH')]]
        self.has = self.ra > 0
        visible = training_context > 0
        self.cutpoints = np.unique(np.quantile(self.distance[visible], np.linspace(0,1,bins+1)[1:-1]))
        self.distance_bin = np.digitize(self.distance, self.cutpoints)
        # Distances are the unchanged standardized road distances. Logistic
        # attenuation needs no fitted distance scale and remains monotone.
        self.access = expit(-self.distance)
        self.deficit = np.maximum(-self.x, 0)
        self.deficit /= 1 + self.deficit
        self.vulnerability = expit(-self.x).mean(1)
        self.destination_supply = expit(self.x).mean(1)
        p = self.pressure(training_context)
        self.pressure_scale = np.nanmedian(p[p > 0]) if (p > 0).any() else 1.0
        oi, di = np.nonzero(visible)
        refs = self.references(training_context, oi, di)
        self.scales = np.array([max(np.nanstd(refs[:, k]), 1e-12) for k in range(2)])

    def pressure(self, context):
        c = np.array(context, copy=True)
        np.fill_diagonal(c, 0)
        incoming = c.sum(0)
        ratios = np.divide(incoming[:, None], self.ra, out=np.zeros_like(self.ra), where=self.has)
        values = np.log1p(ratios)
        return np.divide((values*self.has).sum(1), self.has.sum(1),
                         out=np.full(self.n, np.nan), where=self.has.any(1))

    def components(self, context, oi, di):
        advantage = np.maximum(self.x[di]-self.x[oi], 0)
        advantage /= 1 + advantage
        gap = (self.deficit[oi] * advantage).mean(1)
        p = self.pressure(context)
        pressure = p / (p + self.pressure_scale)
        c = np.array(context, copy=True)
        np.fill_diagonal(c, 0)
        outgoing = c.sum(1)
        total = outgoing + self.internal
        outward = np.divide(outgoing, total, out=np.full(self.n,np.nan), where=total>0)
        alternative = self.destination_supply[di] * self.access[oi,di]
        return dict(deficit=self.deficit[oi].mean(1), advantage=advantage.mean(1),
                    pressure=pressure[oi], vulnerability=self.vulnerability[oi],
                    outward=outward[oi], alternative=alternative,
                    comp=gap*self.access[oi,di],
                    compete=pressure[oi]*self.vulnerability[oi]*outward[oi]*alternative)

    def references(self, context, oi, di):
        c = self.components(context, oi, di)
        return np.column_stack([c['comp'], c['compete']])

    def pairs(self, context, oi, di, seed=0, max_pairs=8192, visible_only=True):
        """Comp: same origin; compete: same destination (origin pressure varies).
        Both additionally match topology and frozen training distance decile.
        Four deterministic random within-stratum matchings avoid all-pairs cost.
        """
        refs = self.references(context, oi, di)
        rng = np.random.default_rng(seed)
        result = []
        for k in range(2):
            valid = np.isfinite(refs[:,k]) & (oi!=di)
            if visible_only:
                valid &= context[oi,di]>0
            idx = np.flatnonzero(valid)
            group = ((oi if k == 0 else di)*3+self.topology[oi,di])*(len(self.cutpoints)+1)+self.distance_bin[oi,di]
            aa, bb = [], []
            for _ in range(4):
                order = idx[np.lexsort((rng.random(len(idx)), group[idx]))]
                a, b = order[:-1], order[1:]
                keep = (group[a]==group[b]) & (np.abs(refs[a,k]-refs[b,k])>self.margin*self.scales[k])
                a,b = a[keep],b[keep]
                high = refs[a,k]>refs[b,k]
                aa.append(np.where(high,a,b));bb.append(np.where(high,b,a))
            a,b = np.concatenate(aa),np.concatenate(bb)
            if len(a)>max_pairs:
                take=rng.choice(len(a),max_pairs,replace=False);a,b=a[take],b[take]
            result.append((a,b))
        return refs,result

    def state(self):
        return dict(distance_cutpoints=self.cutpoints.tolist(), reference_sd=self.scales.tolist(),
                    pressure_scale=float(self.pressure_scale), margin_sd=self.margin,
                    competition_endpoint='origin', matching=['origin_topology_distance','destination_topology_distance'],
                    missing_pressure='exclude no-service origins; never fill as absence',
                    supply_reference='frozen standardized nine benefit indicators; relative, not clinical adequacy')


def ranking_loss(q, pairs, channel):
    a,b=pairs
    if not len(a):
        return q[:,channel].sum()*0
    # Logistic ranking directly on gate probabilities. No mutual exclusion,
    # balance, calibration or synthetic labels; gradients reach only this gate.
    return F.softplus(-(q[a,channel]-q[b,channel])).mean()


def ranked_forward(original, reference):
    counter = [0]
    def forward(model, variant, g, tensors, context, tau):
        out = original(model, variant, g, tensors, context, tau)
        if model.training:
            oi,di=out['edge_index'].detach().numpy()
            _,pairs=reference.pairs(context,oi,di,seed=20000+counter[0])
            counter[0]+=1
            out['comp_rank']=ranking_loss(out['q_edge'],pairs[0],0)
            out['compete_rank']=ranking_loss(out['q_edge'],pairs[1],1)
            out['rank_diagnostics']={c+'_pairs':len(pairs[k][0]) for k,c in enumerate(('comp','compete'))}
        return out
    return forward
