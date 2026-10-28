import argparse
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from diffcls.data import read_rows, split_rows, write_rows
from diffcls.engine import train, load_checkpoint
from diffcls.model import contrastive_loss


def test_contrastive_matches_are_better_than_mismatches():
    a = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    b = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    good = contrastive_loss(a, b, ['one', 'two'], .1)
    bad = contrastive_loss(a, b.flip(0), ['one', 'two'], .1)
    assert good < bad
    good.backward()
    assert a.grad is not None and b.grad is not None


def test_duplicate_heads_are_positive_not_negatives():
    a = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    b = a.clone()
    assert contrastive_loss(a, b, ['one', 'one', 'two'], .1) < .001
    assert contrastive_loss(a[:2], b[:2], ['one', 'one'], .1) is None


def test_two_phase_training_saves_encoder_and_restores_checkpoint(tmp_path):
    torch.set_num_threads(1)
    source = Path(__file__).resolve().parents[1] / 'data/demo.jsonl'
    partitions = split_rows(read_rows(source), 'family')
    for name, rows in partitions.items():
        write_rows(tmp_path / f'{name}.jsonl', rows)
    args = argparse.Namespace(train=str(tmp_path/'train.jsonl'), dev=str(tmp_path/'dev.jsonl'),
        output=str(tmp_path/'run'), epochs=1, batch_size=16, lr=1e-3, max_length=32,
        seed=42, device='cpu', tiny=True, model='unused', local_files_only=True,
        mode='diff_cls', dropout=0.1, weight_decay=0.01, command='train',
        contrastive_epochs=1, contrastive_lr=1e-3, contrastive_temperature=.1)
    path = train(args)
    state = load_file(str(tmp_path/'run/contrastive-encoder.safetensors'))
    model, _, config = load_checkpoint(path, torch.device('cpu'))
    assert config['contrastive_epochs'] == 1
    assert any(not torch.equal(state[k], v) for k, v in model.encoder.state_dict().items())
    assert (tmp_path/'run/contrastive-history.json').is_file()


def test_other_modes_are_rejected_before_training(tmp_path):
    args = argparse.Namespace(epochs=1, batch_size=4, lr=1e-3, max_length=32,
                              contrastive_epochs=1, mode='cls_cls')
    with pytest.raises(ValueError, match='requires diff_cls or concat_cls'):
        train(args)
