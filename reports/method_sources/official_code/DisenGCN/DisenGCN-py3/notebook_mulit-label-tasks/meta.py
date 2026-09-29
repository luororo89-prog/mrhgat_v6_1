#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import argparse
import subprocess
import sys

import hyperopt

min_y = 0
min_c = None


def flush():
    sys.stdout.flush()
    sys.stderr.flush()


def trial(hyperpm, dryrun=False):
    global min_y, min_c
    cmd = 'python main.py'
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if isinstance(v, str):
            cmd += ' %s' % v
        elif int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
        if k == 'ncaps':
            nhidden = 128 // int(v)  # Total dimension = 128.
            cmd += ' --nhidden %d' % nhidden
    try:
        print('\nval=.....%% tst=.....%% @ %s' % cmd, file=sys.stderr)
        flush()
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        print('...')
        flush()
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    if dryrun:
        print(output.decode(encoding='UTF-8'))
        flush()
        score = 0
    else:
        val, tst = eval(output)
        print('val=%5.2f%%,%5.2f%% tst=%5.2f%%,%5.2f%% @ %s' % (
            val[0] * 100, val[1] * 100, tst[0] * 100, tst[1] * 100, cmd))
        flush()
        score = -(val[0] + val[1])  # micro-f1 + macro-f1
        if score < min_y:
            min_y, min_c = score, cmd
    return {'loss': score, 'status': hyperopt.STATUS_OK}


parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='POS',
                    help='blogcatalog, PPI, POS, youtube, flickr')
parser.add_argument('--model', type=str, default='CapsNet',  # CapsNet=DisenGCN
                    help='CapsNet, GCN, JGCN, GAT, ...')
args = parser.parse_args()
space = {'datname': args.datname,
         'model': args.model,
         'lr': hyperopt.hp.loguniform('lr', -8, 0),
         'reg': hyperopt.hp.loguniform('reg', -8, 0),
         'dropout': hyperopt.hp.quniform('dropout', 1, 19, 1) * 0.05,
         'nlayer': hyperopt.hp.quniform('nlayer', 1, 6, 1),
         'ncaps': 1}
if args.model in ('GAT', 'CapsNet'):
    space['ncaps'] = hyperopt.hp.quniform('ncaps', 4, 16, 4)
if args.model == 'CapsNet':
    space['dcaps'] = hyperopt.hp.quniform('dcaps', 0, 4, 2)
    space['routit'] = hyperopt.hp.quniform('routit', 3, 7, 2)
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
print('>>>>', min_c)
