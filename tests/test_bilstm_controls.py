import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('bilstm_controls', Path(__file__).parents[1] / 'scripts/bilstm_controls.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def row(prefix, suffix, parent):
    return {'child_prefix': prefix, 'child_suffix': suffix, 'parent': parent, 'label': 'material'}


def test_marker_position_padding_and_base_representation():
    rows = [row('plastic', 'pallet', 'pallet'), row('steel small', 'box', 'box')]
    vocab = b.build_vocab(rows)
    child, parent, positions, labels, _ = b.collate(rows, vocab, {'material': 0}, True)
    assert positions.tolist() == [1, 2]
    assert all(child[i, pos] == vocab['[DIFF]'] for i, pos in enumerate(positions))
    model = b.BiLSTMClassifier(torch.randn(len(vocab), 8), 5, 1, 'diff_minus_base', 0)
    model.eval()
    with torch.no_grad():
        hc, hp = model.representations(child, parent, positions)
        single = b.collate(rows[:1], vocab, {'material': 0}, True)
        alone_hc, alone_hp = model.representations(*single[:3])
    assert hc.shape == hp.shape == (2, 10)
    assert torch.allclose(hc[0], alone_hc[0], atol=1e-6)
    assert torch.allclose(hp[0], alone_hp[0], atol=1e-6)


def test_glove_exact_dimension_and_hash(tmp_path):
    path = tmp_path / 'glove.txt'
    path.write_text('plastic 1 2\npallet 3 4\n')
    vocab = b.build_vocab([row('plastic', 'pallet', 'pallet')])
    weights, found, digest = b.load_glove(path, vocab, 2)
    assert found == {'plastic', 'pallet'}
    assert weights[vocab['pallet']].tolist() == [3, 4]
    assert len(digest) == 64
