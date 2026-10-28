import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from diffcls.data import read_rows, split_rows, write_rows
from diffcls.engine import train, load_checkpoint, run_eval
from diffcls.model import Collator


def test_offline_training_save_load_predict(tmp_path):
    torch.set_num_threads(1)
    source=Path(__file__).resolve().parents[1]/'data/demo.jsonl'
    partitions=split_rows(read_rows(source),'family')
    for name,rows in partitions.items():
        write_rows(tmp_path/f'{name}.jsonl',rows)
    args=argparse.Namespace(train=str(tmp_path/'train.jsonl'),dev=str(tmp_path/'dev.jsonl'),
        output=str(tmp_path/'run'),epochs=1,batch_size=16,lr=1e-3,max_length=32,
        seed=42,device='cpu',tiny=True,model='unused',local_files_only=True,
        mode='diff_cls',dropout=0.1,weight_decay=0.01,command='train')
    path=train(args)
    model,tok,cfg=load_checkpoint(path,torch.device('cpu'))
    loader=DataLoader(partitions['test'],batch_size=16,collate_fn=Collator(tok,{v:i for i,v in enumerate(cfg['labels'])},cfg['mode'],32))
    a,records=run_eval(model,loader,torch.device('cpu'),cfg['labels'])
    model2,tok2,cfg2=load_checkpoint(path,torch.device('cpu'))
    b,records2=run_eval(model2,loader,torch.device('cpu'),cfg2['labels'])
    assert a==b and records==records2
    assert len(records)==len(partitions['test'])
    assert all(abs(sum(r['probabilities_uncalibrated'].values())-1)<1e-5 for r in records)
