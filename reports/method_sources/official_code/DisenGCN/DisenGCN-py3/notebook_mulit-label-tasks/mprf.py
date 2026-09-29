#!/usr/bin/env python3
#
# A script to run the model several times and report the average performance.
#
# Some sample output:
#   [mprf] macro-f1=22.06%, micro-f1=26.05% @ python main.py --datname PPI --dcaps 4 --dropout 0.45 --lr 0.0057 --model CapsNet --ncaps 12 --nhidden 10 --nlayer 2 --reg 0.00059 --routit 6
#   [mprf] macro-f1=33.14%, micro-f1=54.77% @ python main.py --datname POS --dcaps 4 --dropout 0.45 --lr 0.078 --model CapsNet --ncaps 12 --nhidden 10 --nlayer 2 --reg 0.00048 --routit 6
#   [mprf] macro-f1=29.19%, micro-f1=41.89% @ python main.py --datname blogcatalog --dcaps 2 --dropout 0.4 --lr 0.0079 --model CapsNet --ncaps 12 --nhidden 10 --nlayer 4 --reg 0.00036 --routit 6
#
# I'm not sure if the hyper-parameters in the examples above are
#   the ones for producing the results reported in the paper or not.
# I was probably using '--nepoch 2000' or '--nepoch 3000' instead of
#   just the default '--nepoch 1000'. I can't find the log now.
#
import subprocess

cmd = 'python main.py --datname PPI --dcaps 4 --dropout 0.45 --lr 0.0057 --model CapsNet --ncaps 12 --nhidden 10 --nlayer 2 --reg 0.00059 --routit 6'
# cmd = 'python main.py --datname POS --dcaps 4 --dropout 0.45 --lr 0.078 --model CapsNet --ncaps 12 --nhidden 10 --nlayer 2 --reg 0.00048 --routit 6'
# cmd = 'python main.py --datname blogcatalog --dcaps 2 --dropout 0.4 --lr 0.0079 --model CapsNet --ncaps 12 --nhidden 10 --nlayer 4 --reg 0.00036 --routit 6'

macro_f1s = []
micro_f1s = []
for _ in range(10):  # Plz use a larger sample.
    output = subprocess.check_output(cmd, shell=True)
    _, (macro_f1, micro_f1) = eval(output)
    macro_f1s.append(macro_f1)
    micro_f1s.append(micro_f1)
macro_f1 = sum(macro_f1s) / len(macro_f1s) * 100
micro_f1 = sum(micro_f1s) / len(micro_f1s) * 100
print('[mprf] macro-f1=%5.2f%%, micro-f1=%5.2f%% @ %s' % (
    macro_f1, micro_f1, cmd))
