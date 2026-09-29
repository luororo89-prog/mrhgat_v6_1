"""Independent aggregate generator with explicit components; truth is never an input."""
import numpy as np
from scipy.special import expit

def row_normalize(a):
    total = a.sum(1,keepdims=True)
    return np.divide(a,total,out=np.zeros_like(a),where=total>0)

def make_flows(x, K, baseline_probability, row_totals, scenario, phi, seed):
    if scenario not in ('S0','S1','S1s','S2','S3'):
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    n = len(x)
    s = x.mean(1)
    delta = s[None,:] - s[:,None]
    ext = K != 0
    low = s <= np.quantile(s,1/3)
    high = s >= np.quantile(s,2/3)
    components = np.zeros((n,3,n),dtype=float)
    components[:,0,:] = row_normalize(baseline_probability.astype(float))
    if scenario != 'S0':
        mass = phi * low.astype(float) if scenario != 'S1s' else phi * expit(-(s-np.quantile(s,1/3))/.3)
        pc = row_normalize(np.maximum(delta,0) * np.where(K==1,.4,1.6) * ext)
        mass *= pc.sum(1)>0
        components[:,0,:] *= (1-mass[:,None])
        components[:,1,:] = mass[:,None]*pc
    preliminary = components.sum(1) * row_totals[:,None]
    low_in = (preliminary * (delta>0) * ext).sum(0)
    displacers = np.zeros(n,dtype=bool)
    if scenario in ('S2','S3'):
        pool = high & (row_totals>0)
        displacers = pool & (low_in > np.median(low_in[pool]))
        if scenario == 'S3':
            count = displacers.sum()
            displacers[:] = False
            displacers[rng.choice(np.flatnonzero(pool),count,replace=False)] = True
            mass = phi*displacers
        else:
            mass = np.minimum(.8, phi*low_in/max(low_in[displacers].mean(),1e-12))*displacers
        pd = row_normalize((delta<=0)*ext*np.where(K==1,1.6,.4))
        mass *= pd.sum(1)>0
        components *= (1-mass)[:,None,None]
        components[:,2,:] = mass[:,None]*pd
    probability = components.sum(1)
    np.testing.assert_allclose(probability.sum(1),1,atol=1e-7)
    totals = rng.poisson(row_totals)
    F = np.stack([rng.multinomial(int(total), p/p.sum()) for total,p in zip(totals,probability)])
    truth = components * totals[:,None,None]
    return F.astype(np.float32), {'expected':truth,'low':low,'high':high,'displacers':displacers,
        'delta':delta,'low_in':low_in, 'probability':probability}
