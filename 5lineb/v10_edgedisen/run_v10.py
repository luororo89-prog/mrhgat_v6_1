"""Line B v10: edge-property DisenGCN (model_v10) run through the run_v9 pipeline (routing shares, channel alignment,
joint or per-channel decoders, fixed ruler, channel profiles).

python 5lineb/v10_edgedisen/run_v10.py --config 5lineb/v10_edgedisen/configs/walk_v4_pca5_v10_k3_joint.yaml
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'v9_disengcn'))
import run_v9  # noqa: E402
from model_v10 import train_v10  # noqa: E402

if __name__ == '__main__':
    run_v9.main(train_v10)
