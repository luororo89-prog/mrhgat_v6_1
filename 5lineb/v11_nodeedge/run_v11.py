"""Line B v11: node encoding, edge routing, node aggregation and decoding (model_v11) run through the run_v9 pipeline.

python 5lineb/v11_nodeedge/run_v11.py --config 5lineb/v11_nodeedge/configs/walk_v4_pca5_v11_k3_s20.yaml
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v9_disengcn'))
import run_v9  # noqa: E402
from model_v11 import train_v11  # noqa: E402

if __name__ == '__main__':
    run_v9.main(train_v11)
