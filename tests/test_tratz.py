import json
import pytest
from diffcls.tratz import prepare
from diffcls.data import read_rows


def fixture(tmp_path):
    path = tmp_path / 'source/classification/data/tratz_coarse_grained_lexical'
    path.mkdir(parents=True)
    (path / 'train.tsv').write_text('olive\toil\tsource\nsteel\tbridge\tmaterial\n')
    (path / 'val.tsv').write_text('apple\tjuice\tsource\n')
    (path / 'test.tsv').write_text('iron\tbeam\tmaterial\n')
    return tmp_path / 'source'


def test_preserves_source_splits_labels_and_marker(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / 'out'
    report = prepare(source, output, 'coarse', 'lexical')
    train = read_rows(output / 'train.jsonl')
    assert report['rows'] == {'train': 2, 'dev': 1, 'test': 1}
    assert (train[0]['parent'], train[0]['child_prefix'], train[0]['child_suffix'], train[0]['label']) == ('oil', 'olive', 'oil', 'source')
    assert read_rows(output / 'dev.jsonl')[0]['source_file'] == 'val.tsv'
    assert json.loads((output / 'manifest.json').read_text())['source_split'] == 'lexical'


def test_duplicate_cross_split_is_rejected(tmp_path):
    source = fixture(tmp_path)
    (source / 'classification/data/tratz_coarse_grained_lexical/test.tsv').write_text('olive\toil\tsource\n')
    with pytest.raises(ValueError, match='Duplicate examples'):
        prepare(source, tmp_path / 'out', 'coarse', 'lexical')


def test_bad_row_is_not_silently_dropped(tmp_path):
    source = fixture(tmp_path)
    (source / 'classification/data/tratz_coarse_grained_lexical/test.tsv').write_text('malformed\trow\n')
    with pytest.raises(ValueError, match='expected modifier'):
        prepare(source, tmp_path / 'out', 'coarse', 'lexical')
