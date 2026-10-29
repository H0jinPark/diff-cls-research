import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/backbone_controls.py'
spec = importlib.util.spec_from_file_location('backbone_controls', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_paired_comparisons_use_matching_seeds():
    rows = [dict(setting='diff_cls', seed=13, dev_weighted_f1=.6, dev_macro_f1=.5, best_epoch=2),
            dict(setting='marked_cls_diff', seed=13, dev_weighted_f1=.5, dev_macro_f1=.4, best_epoch=1),
            dict(setting='marked_cls_diff', seed=42, dev_weighted_f1=.1, dev_macro_f1=.2, best_epoch=1)]
    summary = m.summarize(rows)
    pair = summary['paired_weighted_f1_differences']['diff_cls_minus_marked_cls_diff']
    assert pair['differences_by_seed'] == {'13': pytest.approx(.1)}
    assert pair['mean'] == pytest.approx(.1)
