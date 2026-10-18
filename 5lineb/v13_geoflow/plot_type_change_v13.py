"""Type change figure of a model_v13 run: plot_type_change_v11.py with the node types of 3node_typing.

Types (basic and effective) come from the ruler of the run (ruler_from, 3node_typing/results, the types of
3node_typing/results/fig_types.png), measured from x and x + mean dx (dx from consensus_community.csv). The component
change (rise or decline of TH, SH and CHC) comes from the varimax rotation of that ruler (--ruler, made by
rotate_ruler_v11.py --src 3node_typing/results --dst results_lineb/walk_sup1013_rot). English names and colours of the
types as in 3node_typing/code/plot_node_types.py. All eight change patterns of the components are shown (MIN_PATTERN 1,
no pooled 'Other' row).

python 5lineb/v13_geoflow/plot_type_change_v13.py \
    --config 5lineb/v13_geoflow/configs/walk_sup1013_net2_v13_k2_s100_hold_norec_nodecay_iters7.yaml \
    --ruler results_lineb/walk_sup1013_rot
"""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'v11_nodeedge'))
import plot_type_change_v11 as ptc  # noqa: E402

TYPE_EN = {'稀缺型（三甲低·二级低·社康低）': 'Scarcity', '三甲二级型（三甲高·二级高·社康中）': 'SH-based',
           '社康主导型（三甲低·二级低·社康高）': 'CHC-based', '三甲社康型（三甲高·二级低·社康高）': 'Abundance'}
COLOR = {'Scarcity': '#3C78B4', 'SH-based': '#F28E2B', 'CHC-based': '#5AA84F', 'Abundance': '#D9473F'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a, _ = ap.parse_known_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    T = ptc.frozen(ROOT / cfg['ruler_from'])['main']
    rot_frozen, measure = ptc.frozen, ptc.measure

    def frozen(run_dir):
        """Rotated ruler (components) carrying the type names of the run's ruler."""
        out = rot_frozen(run_dir)
        out['main']['type_names'] = [TYPE_EN[t] for t in T['type_names']]
        return out

    ptc.frozen = frozen
    ptc.measure = lambda Z, R: measure(Z, T)
    ptc.MIN_PATTERN = 1
    ptc.TYPE_EN_ROT = {**ptc.TYPE_EN_ROT, **{t: t for t in TYPE_EN.values()}}
    ptc.COLOR = {**ptc.COLOR, **COLOR}
    ptc.main()


if __name__ == '__main__':
    main()
