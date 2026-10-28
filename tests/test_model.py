from types import SimpleNamespace
import torch
import pytest
from transformers import BertTokenizerFast
from diffcls.model import PairClassifier, Collator, feature_vector


@pytest.fixture
def tokenizer(tmp_path):
    path=tmp_path/'vocab.txt'
    path.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nplastic\npallet\n')
    t = BertTokenizerFast(vocab_file=str(path))
    t.add_special_tokens({'additional_special_tokens':['[DIFF]']})
    return t


def row():
    return dict(id='1',parent='pallet',child='plastic pallet',child_prefix='plastic',child_suffix='pallet',modifier='plastic',label='material')


def test_marker_is_single_token_and_position(tokenizer):
    c,p,pos,y,_=Collator(tokenizer,{'material':0},'diff_cls')([row()])
    assert tokenizer.encode('[DIFF]',add_special_tokens=False)==[tokenizer.convert_tokens_to_ids('[DIFF]')]
    assert pos.tolist()==[2]
    assert p['input_ids'][0,0].item()==tokenizer.cls_token_id


def test_no_silent_truncation(tokenizer):
    with pytest.raises(ValueError,match='exceeds'):
        Collator(tokenizer,{'material':0},'diff_cls',4)([row()])


def test_subtraction_direction():
    assert torch.equal(feature_vector(torch.tensor([3.]),torch.tensor([1.]),'diff_cls'),torch.tensor([2.]))


def test_actual_marker_and_cls_selection():
    class Encoder(torch.nn.Module):
        config=SimpleNamespace(hidden_size=2)
        def forward(self,input_ids):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1,1,2))
    m=PairClassifier(Encoder(),2,'diff_cls')
    m.head=torch.nn.Identity()
    actual=m({'input_ids':torch.tensor([[1,8,4]])},{'input_ids':torch.tensor([[2,7]])},torch.tensor([1]))
    assert torch.equal(actual,torch.tensor([[6.,6.]]))
