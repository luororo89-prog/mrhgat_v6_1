import numpy as np
from sklearn.metrics import roc_auc_score

def auc(labels, scores):
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        return float('nan')
    return float(roc_auc_score(labels,scores))

def evaluate(F, K, out, truth, baseline_probability):
    r = out['r'].detach().cpu().numpy()
    M = r*F[:,None,:]
    expected = truth['expected']
    total_expected = expected.sum(1)
    fractions = np.divide(expected,total_expected[:,None,:],out=np.zeros_like(expected),where=total_expected[:,None,:]>0)
    positive = F>0
    delta = truth['delta']
    comp_edges = positive & (K!=0) & (delta>0)
    compet_edges = positive & (K!=0) & (delta<=0)
    valid = F.sum(1)>0
    node = np.divide(M.sum(2),F.sum(1)[:,None],out=np.zeros((len(F),r.shape[1])),where=F.sum(1)[:,None]>0)
    residual = np.log((F+1e-4)/(F.sum(1)[:,None]*baseline_probability+1e-4))
    high_valid = valid & truth['high']
    metrics = {'comp_edge_auc':auc(fractions[:,1,:][comp_edges]>.5,r[:,1,:][comp_edges]),
        'comp_residual_auc':auc(fractions[:,1,:][comp_edges]>.5,residual[comp_edges]),
        'comp_node_auc':auc(truth['low'][valid],node[valid,1]),
        'compet_edge_auc':auc(fractions[:,2,:][compet_edges]>.5,r[:,2,:][compet_edges]),
        'compet_node_auc':auc(truth['displacers'][high_valid],node[high_valid,2]),
        'comp_edge_positive':int((fractions[:,1,:][comp_edges]>.5).sum()),
        'compet_edge_positive':int((fractions[:,2,:][compet_edges]>.5).sum()),
        'comp_share_predicted':float(M[:,1,:].sum()/F.sum()),
        'compet_share_predicted':float(M[:,2,:].sum()/F.sum()),
        'comp_share_expected':float(expected[:,1,:].sum()/expected.sum()),
        'compet_share_expected':float(expected[:,2,:].sum()/expected.sum())}
    for k,name in ((1,'adj'),(2,'crs')):
        mask = K==k
        metrics['topo_'+name+'_pred'] = float(M[:,1,:][mask].sum()/max(F[mask].sum(),1))
        metrics['topo_'+name+'_truth'] = float(expected[:,1,:][mask].sum()/max(total_expected[mask].sum(),1))
    return metrics
