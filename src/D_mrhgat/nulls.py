"""Capacity-free NC/ND generation and multiple-test decision rules."""
import numpy as np
from scipy.stats import norm

def null_probability(rho,logp,s,kind,rng):
    if rho.shape[1]!=3:
        raise ValueError('This NC/ND implementation is for the selected none configuration only')
    r=rho.copy()
    if kind=='NC':
        r[:,0]+=r[:,1]
        r[:,1]=0
    elif kind=='ND':
        strata=np.searchsorted(np.quantile(s,[1/3,2/3]),s,side='right')
        accepted=False
        for _ in range(10000):
            shuffled=rho[:,2].copy()
            for level in np.unique(strata):
                ids=np.flatnonzero(strata==level)
                shuffled[ids]=rng.permutation(rho[ids,2])
            base=rho[:,0]+rho[:,2]-shuffled
            if (base>=0).all():
                r[:,0],r[:,2]=base,shuffled
                accepted=True
                break
        if not accepted:
            raise ValueError('No feasible within-stratum displacement permutation; do not clip negative base mass')
    else:
        raise ValueError(kind)
    np.testing.assert_allclose(r.sum(1),1,atol=1e-6)
    assert (r>=0).all()
    pi=(r[:,:,None]*np.exp(logp)).sum(1)
    return pi/pi.sum(1,keepdims=True)

def holm(p):
    p=np.asarray(p,dtype=float)
    order=np.argsort(p)
    corrected=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result=np.empty_like(p)
    result[order]=np.minimum(corrected,1)
    return result

def decide(observed,nulls,keys,signs):
    rows=[]
    for key,sign in zip(keys,signs):
        values=np.asarray([r[key] for r in nulls],dtype=float)
        beta=observed[key]
        valid=np.isfinite(beta) and np.isfinite(values).all() and values.std(ddof=1)>1e-12
        z=float(sign*(beta-values.mean())/values.std(ddof=1)) if valid else float('nan')
        p=float(norm.sf(z)) if valid else 1.
        rows.append({'coefficient':key,'beta':beta,'expected_sign':sign,'null_mean':float(values.mean()),
            'null_sd':float(values.std(ddof=1)),'z':z,'p':p,'valid':bool(valid)})
    corrected=holm([r['p'] for r in rows])
    for r,p in zip(rows,corrected):
        r['p_holm']=float(p)
        r['detected']=bool(r['valid'] and r['beta']*r['expected_sign']>0 and r['z']>1.96 and p<.05)
    return rows
