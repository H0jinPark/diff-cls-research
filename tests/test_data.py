import copy
from pathlib import Path
import pytest
from diffcls.data import read_rows, split_rows, audit_splits, metrics, write_rows

DEMO = Path(__file__).resolve().parents[1] / "data/demo.jsonl"


def test_family_holdout_and_determinism():
    rows = read_rows(DEMO)
    a = split_rows(rows, "family", 42)
    assert a == split_rows(rows, "family", 42)
    report = audit_splits(a)
    assert sum(len(x) for x in a.values()) == len(rows)
    assert all(not v['family'] and not v['parent'] for v in report['overlap'].values())


def test_lexical_components_do_not_leak():
    rows = read_rows(DEMO)
    # Shared modifiers connect every family; cannot honestly form three partitions.
    with pytest.raises(ValueError, match="independent groups"):
        split_rows(rows, "lexical")


def test_duplicate_pairs_rejected(tmp_path):
    rows = read_rows(DEMO)[:1]
    extra = copy.deepcopy(rows[0]); extra['id'] = 'other'
    write_rows(tmp_path/'bad.jsonl', rows + [extra])
    with pytest.raises(ValueError, match="Duplicate parent-child"):
        read_rows(tmp_path/'bad.jsonl')


def test_marker_alignment_rejected(tmp_path):
    row = read_rows(DEMO)[0]
    row['child_suffix'] = 'wrong'
    write_rows(tmp_path/'bad.jsonl', [row])
    with pytest.raises(ValueError, match="reconstruct"):
        read_rows(tmp_path/'bad.jsonl')


def test_cross_split_duplicate():
    row = read_rows(DEMO)[0]
    with pytest.raises(ValueError, match="Duplicate examples"):
        audit_splits(dict(train=[row], dev=[row], test=[]))


def test_metrics_known_confusion():
    m = metrics([0, 0, 1, 1], [0, 1, 1, 1], ['a','b'])
    assert m['confusion_matrix'] == [[1,1],[0,2]]
    assert m['accuracy'] == 0.75
    assert m['macro_f1'] == pytest.approx((2/3 + 0.8)/2)
