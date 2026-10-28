import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/paper_comparison.py'
spec = importlib.util.spec_from_file_location('paper_comparison', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_report_aggregates_paper_weighted_scores_by_setting():
    rows = [dict(setting='diff_cls', best_epoch=2, test=dict(weighted_f1=.5, macro_f1=.4, accuracy=.6)),
            dict(setting='diff_cls', best_epoch=3, test=dict(weighted_f1=.7, macro_f1=.6, accuracy=.8))]
    report = module.build_report(rows, {'selection_metric': 'weighted_f1'})
    assert report['by_setting']['diff_cls']['test_weighted_f1_mean'] == pytest.approx(.6)
    assert report['by_setting']['diff_cls']['test_weighted_f1_std_sample'] == pytest.approx(.141421356)
    assert report['by_setting']['diff_cls']['best_epochs'] == [2, 3]
    assert report['reference']['test_weighted_f1'] == .478


def test_default_conditions_are_predeclared():
    settings = module.SETTINGS
    assert len(settings) == 8
    assert len({x[0] for x in settings}) == len(settings)
    assert {x[0] for x in settings if x[2]} == {'diff_cls_cl', 'concat_cls_cl'}
