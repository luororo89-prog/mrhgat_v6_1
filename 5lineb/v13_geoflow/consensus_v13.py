"""Consensus edge assignment of a model_v13 run: consensus_v10.py with the model_v13 network (MechNet) loaded.

python 5lineb/v13_geoflow/consensus_v13.py --config 5lineb/v13_geoflow/configs/walk_v1_net2_v13_k2_s30.yaml
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v10_edgedisen'))
from model_v13 import MechNet, pressure_features  # noqa: E402
import model_v11  # noqa: E402
from graph import build_graph  # noqa: E402
import consensus_v10  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text(encoding='utf-8'))
    g = build_graph(cfg)
    model_v11.NodeDisen = partial(MechNet, pressure=pressure_features(g) if cfg.get('pressure', True) else None,
                                  attention=cfg.get('attention', True), outflow=cfg.get('outflow', True),
                                  base=cfg.get('base', True), n2v=cfg.get('n2v'),
                                  emb=np.zeros((len(g['ids']), 16)) if cfg.get('n2v') else None,
                                  d=cfg['hid_dim'] // cfg['n_channels'] if cfg.get('hid_dim') else 8, n=len(g['ids']),
                                  **{k: cfg[k] for k in ('proj', 'routing_layers', 'dropout', 'channel_score', 'route_score',
                                                         'route_distance', 'rule_topology', 'rule_origin', 'rule_distance')
                                     if k in cfg})
    consensus_v10.main()
