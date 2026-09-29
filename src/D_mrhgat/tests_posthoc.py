"""Post-hoc directional prerequisites. Never imported by the model or loss."""
import numpy as np

def standardized_coefficient(y, predictors):
    y=np.asarray(y,dtype=float)
    X=np.asarray(predictors,dtype=float)
    valid=np.isfinite(y)&np.isfinite(X).all(1)
    y,X=y[valid],X[valid]
    if len(y)<=X.shape[1]+2 or y.std()<=1e-12 or (X.std(0)<=1e-12).any():
        return float('nan')
    X=(X-X.mean(0))/X.std(0)
    y=(y-y.mean())/y.std()
    return float(np.linalg.lstsq(np.column_stack([np.ones(len(y)),X]),y,rcond=None)[0][1])

def directional_prerequisites(x,F,K,r,variant):
    s=x.mean(1)
    delta=s[None,:]-s[:,None]
    ext=K!=0
    inflow=(F*ext).sum(0)
    low_in=(F*ext*(delta>0)).sum(0)
    outflow=F.sum(1)
    valid=outflow>0
    disp=(F*r[:,2,:]).sum(1)/np.maximum(outflow,1)
    self_share=np.diag(F)/np.maximum(outflow,1)
    result={}
    for k,name in ((1,'ADJ'),(2,'CRS')):
        i,j=np.where((K==k)&(F>0)&(delta>0))
        result['comp_'+name+'_beta']=standardized_coefficient(r[i,1,j],np.column_stack([delta[i,j],inflow[i]]))
        i,j=np.where((K==k)&(F>0)&(delta<=0))
        result['compet_'+name+'_beta']=standardized_coefficient(r[i,2,j],np.column_stack([np.log1p(low_in[i]),delta[i,j]]))
    X=np.column_stack([np.log1p(low_in[valid]),s[valid]])
    result['self_low_inflow_beta']=standardized_coefficient(self_share[valid],X)
    result['disp_low_inflow_beta']=standardized_coefficient(disp[valid],X)
    if variant=='independent':
        result['required_compet_direction']=(result['self_low_inflow_beta']<0 and result['disp_low_inflow_beta']>0)
    else:
        # Any accepted edge topology must first have a positive coefficient.
        result['required_compet_direction']=any(result['compet_'+k+'_beta']>0 for k in ('ADJ','CRS'))
    return result
