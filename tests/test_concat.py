from types import SimpleNamespace
import argparse
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from diffcls.data import read_rows, split_rows, write_rows
from diffcls.engine import train, load_checkpoint, run_eval
from diffcls.model import Collator, PairClassifier, feature_vector


def test_concat_order_and_feature_dimensions():
    child = torch.tensor([[1., 2.]])
    parent = torch.tensor([[3., 4.]])
    assert torch.equal(feature_vector(child, parent, 'concat_cls'), torch.tensor([[1., 2., 3., 4.]]))
    class Encoder(torch.nn.Module):
        config = SimpleNamespace(hidden_size=2)
        def forward(self, input_ids):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1, 1, 2))
    model = PairClassifier(Encoder(), 3, 'concat_cls')
    model.head = torch.nn.Identity()
    result = model({'input_ids': torch.tensor([[1, 8]])}, {'input_ids': torch.tensor([[2, 7]])}, torch.tensor([1]))
    assert torch.equal(result, torch.tensor([[8., 8., 2., 2.]]))


def test_pair_interaction_order_and_model_shape():
    child = torch.tensor([[1., 2.]])
    parent = torch.tensor([[3., 4.]])
    assert torch.equal(feature_vector(child, parent, 'pair_interaction'),
                       torch.tensor([[1., 2., 3., 4., 2., 2., 3., 8.]]))
    class Encoder(torch.nn.Module):
        config = SimpleNamespace(hidden_size=2)
        def forward(self, input_ids):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1, 1, 2))
    model = PairClassifier(Encoder(), 3, 'pair_interaction')
    assert model.head[1].in_features == 8
    model.head = torch.nn.Identity()
    result = model({'input_ids': torch.tensor([[1, 8]])}, {'input_ids': torch.tensor([[2, 7]])}, torch.tensor([1]))
    assert torch.equal(result, torch.tensor([[8., 8., 2., 2., 6., 6., 16., 16.]]))


@pytest.mark.parametrize('mode,cl_epochs', [('concat_cls', 0), ('concat_cls', 1), ('pair_interaction', 0)])
def test_concat_training_checkpoint_and_evaluation(tmp_path, mode, cl_epochs):
    torch.set_num_threads(1)
    source = Path(__file__).resolve().parents[1] / 'data/demo.jsonl'
    splits = split_rows(read_rows(source), 'family')
    for name, rows in splits.items():
        write_rows(tmp_path / f'{name}.jsonl', rows)
    args = argparse.Namespace(train=str(tmp_path/'train.jsonl'), dev=str(tmp_path/'dev.jsonl'),
        output=str(tmp_path/'run'), epochs=1, batch_size=16, lr=1e-3, max_length=32,
        seed=42, device='cpu', tiny=True, model='unused', local_files_only=True,
        mode=mode, dropout=0.1, weight_decay=0.01, command='train',
        contrastive_epochs=cl_epochs, contrastive_lr=1e-3, contrastive_temperature=.1)
    checkpoint = train(args)
    model, tok, config = load_checkpoint(checkpoint, torch.device('cpu'))
    assert model.head[1].in_features == (128 if mode == 'pair_interaction' else 64)
    assert config['mode'] == mode
    loader = DataLoader(splits['test'], batch_size=16,
        collate_fn=Collator(tok, {v:i for i,v in enumerate(config['labels'])}, mode, 32))
    result, records = run_eval(model, loader, torch.device('cpu'), config['labels'])
    assert len(records) == len(splits['test'])
    assert 0 <= result['macro_f1'] <= 1
