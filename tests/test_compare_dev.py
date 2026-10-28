import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/compare_dev.py'
spec = importlib.util.spec_from_file_location('compare_dev', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_paired_differences():
    rows = [dict(mode='diff_cls', seed=13, best_dev_macro_f1=.4, best_epoch=3),
            dict(mode='diff_cls', seed=42, best_dev_macro_f1=.6, best_epoch=5),
            dict(mode='cls_cls', seed=13, best_dev_macro_f1=.3, best_epoch=4),
            dict(mode='cls_cls', seed=42, best_dev_macro_f1=.4, best_epoch=5)]
    summary = module.build_summary(rows)
    assert summary['by_mode']['diff_cls']['macro_f1_mean'] == pytest.approx(.5)
    assert summary['by_mode']['diff_cls']['macro_f1_std_sample'] == pytest.approx(.141421356)
    assert summary['paired_differences']['cls_cls']['mean_diff_cls_minus_baseline'] == pytest.approx(.15)
    assert summary['paired_differences']['cls_cls']['n'] == 2


def test_one_seed_has_no_sample_std():
    row = dict(mode='diff_cls', seed=13, best_dev_macro_f1=.4, best_epoch=1)
    assert module.build_summary([row])['by_mode']['diff_cls']['macro_f1_std_sample'] is None
