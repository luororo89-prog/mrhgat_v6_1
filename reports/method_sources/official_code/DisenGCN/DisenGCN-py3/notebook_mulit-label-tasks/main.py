#!/usr/bin/env python3


import argparse
import os
import random
import sys
import tempfile
import time
import urllib.request

import gc
import networkx as nx
import numpy as np
import scipy.io
import scipy.sparse
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import sklearn.preprocessing
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim


class DataReader:
    def __init__(self, data_name, data_dir, use_val, trn_ratio):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        assert data_name in ["blogcatalog", "flickr", "youtube",
                             "PPI", "POS", "SYN"]
        mat_file = os.path.join(data_dir, data_name + '.mat')
        adj, targ = DataReader._load_tang09_mat(mat_file)
        graph = DataReader._mat_to_nx(adj)
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        targ = targ.toarray().astype(np.int64)
        trn = DataReader._sample_trn(targ, trn_ratio)
        val_tst = list(set(range(n)) - set(trn))
        if use_val:
            np.random.shuffle(val_tst)
            val = val_tst[:len(val_tst) // 2]
            tst = val_tst[len(val_tst) // 2:]
        else:
            val, tst = [], val_tst
        trn = [x for x in trn if targ[x].sum() > 0]
        print('#instance (trn, val, tst) ~ #links ~ #class = '
              '%d (%d, %d, %d) ~ %d ~ %d' % (
                  n, len(trn), len(val), len(tst), m, targ.shape[1]),
              file=sys.stderr)
        self.trn_idx = np.array(sorted(trn), np.int64)
        self.val_idx = np.array(sorted(val), np.int64)
        self.tst_idx = np.array(sorted(tst), np.int64)
        self.graph = graph
        self.feat = scipy.sparse.eye(n) + adj  # = scipy.sparse.eye(n)
        self.targ = targ

    @staticmethod
    def _sample_trn(targ, trn_ratio):
        n, m = targ.shape
        cnt_all = targ.sum(axis=0)
        cnt_trn = np.zeros_like(cnt_all)
        trn_idx = []
        for i in np.random.permutation(n):
            to_keep = False
            for j in np.random.permutation(m):
                if targ[i, j] > 0 and cnt_trn[j] < cnt_all[j] * trn_ratio:
                    to_keep = True
                    break
            if to_keep:
                trn_idx.append(i)
                for j in range(m):
                    if targ[i, j] > 0:
                        cnt_trn[j] += 1
        return trn_idx

    @staticmethod
    def _load_tang09_mat(mat_data_path):
        _, data_file = os.path.split(mat_data_path)
        assert data_file in ["blogcatalog.mat", "flickr.mat",
                             "youtube.mat", "PPI.mat", "POS.mat", "SYN.mat"]
        if not os.path.isfile(mat_data_path):
            if data_file in ["PPI.mat", "POS.mat"]:
                origin = "http://snap.stanford.edu/node2vec/"
            else:
                origin = "http://leitang.net/code/social-dimension/data/"
            if data_file == "PPI.mat":
                data_file = "Homo_sapiens.mat"
            url = origin + data_file
            urllib.request.urlretrieve(url, mat_data_path)
        mat = scipy.io.loadmat(mat_data_path)
        return mat["network"], mat["group"]

    @staticmethod
    def _mat_to_nx(adj_mat):
        g = nx.Graph()
        g.add_nodes_from(range(adj_mat.shape[0]))
        coo = scipy.sparse.coo_matrix(adj_mat)
        for u, v, _ in zip(coo.row, coo.col, coo.data):
            g.add_edge(u, v)
        assert g.number_of_nodes() == adj_mat.shape[0]
        return g

    def get_split(self):
        # *val_idx* contains unlabeled samples for semi-supervised training.
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ


class MultiLabelEval:
    @staticmethod
    def eval_emb(x, y, trn_idx, tst_idx):
        classifier = sklearn.multiclass.OneVsRestClassifier(
            sklearn.linear_model.LogisticRegression(solver='lbfgs'),
            n_jobs=-1)
        classifier.fit(x[trn_idx], y[trn_idx])
        score = classifier.predict_proba(x[tst_idx])
        return MultiLabelEval.eval(score, y[tst_idx], print_detail=True)

    @staticmethod
    def eval(pred_score, y, print_detail):
        assert isinstance(pred_score, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert pred_score.shape == y.shape
        tang09_metrics = MultiLabelEval._compute_tang09_error(
            pred_score, y, print_detail=print_detail)
        perf = {'macro_f1': tang09_metrics['macro'],
                'micro_f1': tang09_metrics['micro']}
        return perf

    @staticmethod
    def _preserve_k_top(score, k, axis):
        assert score.ndim == 2
        assert k.shape == (score.shape[1 - axis],)
        index = np.argsort(-score, axis=axis)
        pred = np.zeros_like(score, np.int64)
        for i in range(score.shape[1 - axis]):
            if axis == 0:
                pred[index[:k[i], i], i] = 1
            else:
                pred[i, index[i, :k[i]]] = 1
        return pred

    @staticmethod
    def _compute_tang09_error(score, y, print_detail, label_wise=False):
        """
        Translated from a MATLAB script provided by Tang & Liu. See:
            Relational learning via latent social dimensions. KDD '09.
        """
        assert score.ndim == 2
        assert score.shape == y.shape
        assert y.dtype in (np.int, np.int32, np.int64, np.bool)
        if y.dtype != np.int64:
            y = y.astype(np.int64)
        if label_wise:
            # Assuming the number of samples per label is known,
            # preserve only top-scored samples for each label.
            pred = MultiLabelEval._preserve_k_top(score, np.sum(y, 0), 0)
        else:
            # Assuming the number of labels per sample is known,
            # preserve only top-scored labels for each sample.
            pred = MultiLabelEval._preserve_k_top(score, np.sum(y, 1), 1)
        index = (np.sum(y, axis=0) > 0)  # remove labels with no samples
        y, pred = y[:, index], pred[:, index]
        if print_detail:
            print(sklearn.metrics.classification_report(y, pred),
                  file=sys.stderr)
        results = {
            'micro': sklearn.metrics.f1_score(y, pred, average='micro'),
            'macro': sklearn.metrics.f1_score(y, pred, average='macro')}
        return results


# noinspection PyUnresolvedReferences
def thsprs_from_spsprs(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))


# noinspection PyUnresolvedReferences
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


# noinspection PyUnresolvedReferences
class NeibSampler:
    def __init__(self, graph, nb_size):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb_all[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        print('#popkids / #all = %d/%d' % (len(popkids), n), file=sys.stderr)
        self.g, self.nb_all, self.pk = graph, nb_all, popkids
        self.sample(keep_big_deg=True)

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self, keep_big_deg=False):
        g = self.g
        nb = self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            v_nb = list(g.neighbors(v))
            if keep_big_deg:
                v_nb = sorted(v_nb, key=(lambda x: -g.degree(x)))
                pk_nb[i] = np.array(v_nb[:nb_size], dtype=np.int64)
            else:
                pk_nb[i] = np.random.choice(v_nb, nb_size, replace=False)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


# noinspection PyUnresolvedReferences
class RoutingLayer(nn.Module):
    def __init__(self, cap_sz, out_caps, inp_caps=None):
        super(RoutingLayer, self).__init__()
        if inp_caps is not None:
            self.fc = nn.Linear(cap_sz * inp_caps, cap_sz * out_caps)
        self.d, self.k = cap_sz * out_caps, out_caps
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        if hasattr(self, 'fc'):
            x = fn.relu(self.fc(x))
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, delta_d = self.d, self.k, self.d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)
        u = x.view(n, k, delta_d)  # u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2)  # p = fn.softmax(p / tau, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)


# This is DisenGCN.
class CapsuleNet(nn.Module):
    def __init__(self, nfeat, nclass, hyperpm, jump):
        super(CapsuleNet, self).__init__()
        cur_dim = hyperpm.ncaps * hyperpm.nhidden
        self.pca = SparseInputLinear(nfeat, cur_dim)
        conv_ls = []
        inp_caps, out_caps = None, hyperpm.ncaps
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(hyperpm.nhidden, out_caps, inp_caps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
            cur_dim += out_caps * hyperpm.nhidden
            inp_caps, out_caps = out_caps, max(1, out_caps - hyperpm.dcaps)
        self.conv_ls = conv_ls
        self.jump = jump
        if jump:
            self.mlp = nn.Linear(cur_dim, nclass)
        else:
            self.mlp = nn.Linear(inp_caps * hyperpm.nhidden, nclass)
        self.dropout = hyperpm.dropout
        self.routit = hyperpm.routit

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb):
        nb = nb.view(-1)
        xs = []
        x = self._dropout(fn.relu(self.pca(x)))
        xs.append(x)
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
            xs.append(x)
        if self.jump:
            # noinspection PyUnresolvedReferences
            x = torch.cat(xs, dim=1)
        x = self.mlp(x)
        return x


class EvalHelper:
    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, hyperpm):
        use_cuda = torch.cuda.is_available() and not hyperpm.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        graph, feat, targ = dataset.get_graph_feat_targ()
        if hyperpm.model.startswith('CapsNet'):
            print('normalizing input features...', file=sys.stderr)
            feat = sklearn.preprocessing.normalize(feat)
        nclass = targ.shape[1]
        trn_idx, val_idx, tst_idx = dataset.get_split()
        loss_fn = nn.MultiLabelSoftMarginLoss()
        targ_onehot_np = targ
        if isinstance(loss_fn, nn.MultiLabelMarginLoss):
            # weights
            weights = targ.sum(axis=1)
            assert weights.min() > 0
            assert weights.max() < nclass
            weights = (nclass - weights)  # weights * (nclass - weights)
            weights = nclass / weights.astype(np.float32)
            weights = torch.from_numpy(weights).to(dev)
            # targets
            new_targ = [[] for _ in range(targ.shape[0])]
            for i, j in np.argwhere(targ > 0):
                new_targ[i].append(j)
            for x in new_targ:
                x.extend([-1] * (nclass - len(x)))
            targ = np.array(new_targ, dtype=np.int64)
            targ = torch.from_numpy(targ).to(dev)
        else:
            assert isinstance(loss_fn, nn.MultiLabelSoftMarginLoss)
            weights = targ.sum(axis=0)
            assert weights.min() > 0
            weights = targ.shape[0] / weights.astype(np.float32)
            weights = torch.from_numpy(weights).to(dev)
            loss_fn = nn.MultiLabelSoftMarginLoss(weight=weights)
            targ = torch.from_numpy(targ.astype(np.float32)).to(dev)
        if hasattr(feat, 'tocoo'):
            feat = thsprs_from_spsprs(feat)
        else:
            feat = torch.from_numpy(feat)
        feat = feat.to(dev)
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        self.targ_onehot_np = targ_onehot_np
        self.loss_fn = loss_fn
        self.weights = weights
        self.use_val = hyperpm.val
        self.graph, self.feat, self.targ = graph, feat, targ
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz = None, None
        self.renew_model(hyperpm)
        self.neib_sampler = NeibSampler(graph, hyperpm.nbsz)

    def renew_model(self, hyperpm):
        print('-- Initializing %s --' % hyperpm.model, file=sys.stderr)
        dev = self.feat.device
        nfeat, nclass = self.feat.size(1), self.targ_onehot_np.shape[1]
        if hyperpm.model == 'CapsNet':
            model = CapsuleNet(nfeat, nclass, hyperpm, jump=True)
        elif hyperpm.model == 'CapsNetNJ':
            model = CapsuleNet(nfeat, nclass, hyperpm, jump=False)
        else:
            assert False
        model = model.to(dev)
        optmz = optim.Adam(model.parameters(),
                           lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.model, self.optmz = model, optmz

    def _get_logits(self, resample=False):
        dev = self.feat.device
        if isinstance(self.model, CapsuleNet):
            if self.neib_sampler.nb_all.device != dev:
                self.neib_sampler.to(dev)
            if resample:
                self.neib_sampler.sample()
            logits = self.model(self.feat, self.neib_sampler.nb_all)
        else:
            if not hasattr(self, 'adj'):
                adj = nx.adjacency_matrix(self.graph)
                adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
                adj += scipy.sparse.eye(adj.shape[0])
                rowsum = np.array(adj.sum(1))
                r_inv_sqrt = np.power(rowsum, -0.5).flatten()
                r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
                r_mat_inv_sqrt = scipy.sparse.diags(r_inv_sqrt)
                adj = adj.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
                adj = thsprs_from_spsprs(adj)
                self.adj = adj.to(dev)
            logits = self.model(self.feat, self.adj)
        return logits

    def run_epoch(self, end='\n'):
        self.model.train()
        self.optmz.zero_grad()
        logits = self._get_logits(resample=True)  # = self._get_logits() # TODO
        loss = self.loss_fn(logits[self.trn_idx], self.targ[self.trn_idx])
        loss.backward()
        self.optmz.step()
        print('trn-loss: %.4f' % loss.item(), end=end, file=sys.stderr)
        return loss.item()

    def print_trn_acc(self):
        print('trn-', end='', file=sys.stderr)
        if self.use_val:
            trn_acc = self._print_acc(self.trn_idx, end=' val-')
            val_acc = self._print_acc(self.val_idx)
        else:
            trn_acc = self._print_acc(self.trn_idx)
            val_acc = 0
        return trn_acc, val_acc

    def print_tst_acc(self):
        print('tst-', end='', file=sys.stderr)
        tst_acc = self._print_acc(self.tst_idx, is_final=True)
        return tst_acc

    def _print_acc(self, eval_idx, end='\n', is_final=False):
        self.model.eval()
        logits = self._get_logits()
        prob = logits[eval_idx]
        prob = prob.detach().cpu().numpy()
        targ = self.targ_onehot_np[eval_idx]
        acc = MultiLabelEval.eval(prob, targ, is_final)
        macro_f1 = acc['macro_f1']
        micro_f1 = acc['micro_f1']
        print('macro,micro-f1: %5.2f%%,%5.2f%%' % (
            macro_f1 * 100, micro_f1 * 100), end=end, file=sys.stderr)
        return macro_f1, micro_f1

    def load_checkpoint(self, tmpfile):
        tmpfile.seek(0)
        ckp = torch.load(tmpfile)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optmz.load_state_dict(ckp['optimizer_state_dict'])

    def save_checkpoint(self, tmpfile=None):
        tmpfile = tmpfile or tempfile.TemporaryFile()
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optmz.state_dict()},
                   tmpfile)
        return tmpfile


# noinspection PyUnresolvedReferences
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# noinspection PyUnresolvedReferences
def train_and_eval(datadir, datname, hyperpm, num_trials=0, trial_steps=1):
    if hyperpm.seed >= 0:
        set_rng_seed(hyperpm.seed)
    dataset = DataReader(datname, datadir, use_val=hyperpm.val,
                         trn_ratio=hyperpm.trnr)
    agent = EvalHelper(dataset, hyperpm)
    best_trn_acc = (0, 0)
    model_sav = agent.save_checkpoint()
    tm = time.time()
    for trial_i in range(num_trials + 1):
        if trial_i < num_trials:
            nepoch = trial_steps
        else:
            agent.load_checkpoint(model_sav)
            nepoch = hyperpm.nepoch
        wait_cnt = 0
        for t in range(nepoch):
            print('%3d/%d' % (t, hyperpm.nepoch), end=' ', file=sys.stderr)
            agent.run_epoch(end=' ')
            cur_trn_acc, cur_val_acc = agent.print_trn_acc()
            if hyperpm.val:
                cur_trn_acc = cur_val_acc
            if sum(cur_trn_acc) > sum(best_trn_acc):
                wait_cnt = 0
                best_trn_acc = cur_trn_acc
                model_sav.close()
                model_sav = agent.save_checkpoint()
            elif trial_i == num_trials:
                wait_cnt += 1
                if wait_cnt > hyperpm.early:
                    break
        if trial_i < num_trials:
            agent.renew_model(hyperpm)
    print("time: %.4f sec." % (time.time() - tm), file=sys.stderr)
    agent.load_checkpoint(model_sav)
    return best_trn_acc, agent.print_tst_acc()


def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='PPI',
                        help='PPI/POS/blogcatalog')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed. A non-negative number.')
    parser.add_argument('--model', type=str, default='CapsNet',
                        help='Default: CapsNet. CapsNet is DisenGCN')
    parser.add_argument('--val', action='store_true', default=True,
                        help='To use a validation set. Default: False.')
    parser.add_argument('--nepoch', type=int, default=1000,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=1000,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=5,
                        help='Number of conv layers.')
    parser.add_argument('--ncaps', type=int, default=7,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--dcaps', type=int, default=1,
                        help='Decrease this number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=16,
                        help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=6,
                        help='Number of iterations when routing.')
    parser.add_argument('--nbsz', type=int, default=32,
                        help='Size of the sampled neighborhood.')
    parser.add_argument('--trnr', type=float, default=0.8,
                        help='Ratio of train versus all.')
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    val_acc, tst_acc = train_and_eval(args.datadir, args.datname, args)
    print('(val) macro=%5.2f%% micro=%5.2f%% || '
          '(tst) macro=%5.2f%% micro=%5.2f%%' % (
              val_acc[0] * 100, val_acc[1] * 100,
              tst_acc[0] * 100, tst_acc[1] * 100), file=sys.stderr)
    return val_acc, tst_acc


if __name__ == '__main__':
    print(str(main()))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
