"""Post-training mechanism judgment for anonymous factors.

No model or training module imports this file. Directional diagnostics are not
significance tests. Final names require the MD null z and Holm-adjusted p evidence.
"""
import numpy as np
from .tests_posthoc import directional_prerequisites
from .nulls import holm


def classify_evidence(compensation_tests, competition_tests):
    def passed(tests):
        return any(t.get('valid',True) and t.get('beta',0)>0 and t.get('z',0)>1.96
                   and t.get('p_holm',t.get('holm_p',1))<.05 for t in tests)
    comp,compet=passed(compensation_tests),passed(competition_tests)
    if comp and compet:
        return 'mixed_unresolved'
    if comp:
        return 'compensation_pattern'
    if compet:
        return 'competition_pattern'
    return 'unresolved'


def screen_anonymous_factors(x,F,K,r):
    """Candidate directions only. Labels remain unresolved until null evidence."""
    result=[]
    for channel in range(1,r.shape[1]):
        # Evaluate BOTH hypotheses for each same factor, not a fixed channel role.
        probe=np.stack([r[:,0,:],r[:,channel,:],r[:,channel,:]],axis=1)
        values=directional_prerequisites(x,F,K,probe,'none')
        result.append({'factor_index':channel,
            'compensation_direction':{k:values['comp_'+k+'_beta'] for k in ('ADJ','CRS')},
            'competition_direction':{k:values['compet_'+k+'_beta'] for k in ('ADJ','CRS')},
            'null_tests_completed':False,'mechanism_label':'unresolved'})
    return result


def judge_anonymous_factors(tests, factor_indices=(1,2)):
    """Name factors from completed null tests, correcting the entire search.

    Each row must identify factor_index, hypothesis, topology, beta, z, p and
    valid. All factors, two hypotheses and two topologies form one Holm family.
    Incomplete or invalid evidence cannot produce a mechanism name.
    """
    expected={(c,h,k) for c in factor_indices for h in ('compensation','competition')
              for k in ('ADJ','CRS')}
    keys=[(t['factor_index'],t['hypothesis'],t['topology']) for t in tests]
    if set(keys)!=expected or len(keys)!=len(expected):
        raise ValueError('Complete unique null tests for every anonymous factor are required')
    rows=[dict(t) for t in tests]
    for row in rows:
        row['valid']=bool(row.get('valid',False) and
                          all(np.isfinite(row.get(k,np.nan)) for k in ('beta','z','p')) and
                          0<=row['p']<=1)
    adjusted=holm([r['p'] if r['valid'] else 1. for r in rows])
    for row,p in zip(rows,adjusted):
        row['p_holm']=float(p)
    result=[]
    for channel in factor_indices:
        evidence=[r for r in rows if r['factor_index']==channel]
        valid=all(r['valid'] for r in evidence)
        comp=[r for r in evidence if r['hypothesis']=='compensation']
        compet=[r for r in evidence if r['hypothesis']=='competition']
        result.append({'factor_index':channel,'null_tests_completed':True,
                       'evidence_valid':valid,'mechanism_label':
                       classify_evidence(comp,compet) if valid else 'unresolved',
                       'tests':evidence})
    return result
