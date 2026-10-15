"""Line B v13: Base-Compensation-Competition network (model_v13) run through the run_v9 pipeline.
pressure, attention, outflow, base, edge_topology, labels_file, label_weight, label_holdout, holdout_share, holdout_seed and n2v in the
config are passed to train_v13.

python 5lineb/v13_geoflow/run_v13.py --config 5lineb/v13_geoflow/configs/walk_v1_net2_v13_k2_s30.yaml
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v9_disengcn'))
import run_v9  # noqa: E402
from model_v13 import train_v13  # noqa: E402

if __name__ == '__main__':
    run_v9.FIT_KEYS = run_v9.FIT_KEYS + ('pressure', 'attention', 'outflow', 'base', 'edge_topology', 'road_impedance', 'dest_fe', 'labels_file', 'label_weight',
                                         'label_holdout', 'holdout_share', 'holdout_seed', 'n2v', 'gravity_mlp', 'freeze_from', 'fc_head', 'dropout', 'proj', 'hid_dim', 'routing_layers', 'weight_decay',
                                         'val_share', 'early', 'channel_score',
                                         'route_score', 'route_distance', 'rule_topology', 'rule_origin', 'rule_distance',
                                         'consistency_score', 'consistency_sim')
    run_v9.main(train_v13)
