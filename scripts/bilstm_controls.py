"""Run GloVe/BiLSTM marker and contrastive controls, using validation only."""
import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader

from diffcls.data import audit_splits, fingerprint, metrics, read_rows, write_json
from diffcls.model import contrastive_loss

CASES = ('compound_mean', 'diff_only', 'diff_minus_base', 'diff_minus_base_cl')


def words(text):
    return re.findall(r"[\w]+(?:['-][\w]+)*|[^\w\s]", text.casefold())


def build_vocab(rows):
    vocab = {'<pad>': 0, '<unk>': 1, '[DIFF]': 2}
    for row in rows:
        for key in ('child_prefix', 'child_suffix', 'parent'):
            for token in words(row[key]):
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def load_glove(path, vocab, dimension):
    weights = torch.randn(len(vocab), dimension, generator=torch.Generator().manual_seed(0)) * .05
    weights[0].zero_()
    found, digest = set(), hashlib.sha256()
    with open(path, 'rb') as stream:
        for raw in stream:
            digest.update(raw)
            token, sep, values = raw.rstrip().partition(b' ')
            try:
                token = token.decode('utf-8')
            except UnicodeDecodeError:
                continue
            if sep and token in vocab and vocab[token] >= 3:
                vector = torch.tensor([float(x) for x in values.split()])
                if vector.numel() != dimension or not torch.isfinite(vector).all():
                    raise ValueError(f'Invalid GloVe vector: {token}')
                weights[vocab[token]] = vector
                found.add(token)
    if not found:
        raise ValueError('No words found in GloVe file')
    return weights, found, digest.hexdigest()


def collate(rows, vocab, labels, marked):
    children, parents, positions = [], [], []
    for row in rows:
        prefix, suffix, parent = (words(row[k]) for k in ('child_prefix', 'child_suffix', 'parent'))
        if not suffix or not parent:
            raise ValueError('Empty tokenized suffix or parent')
        tokens = prefix + (['[DIFF]'] if marked else []) + suffix
        children.append(torch.tensor([vocab.get(t, 1) for t in tokens]))
        parents.append(torch.tensor([vocab.get(t, 1) for t in parent]))
        positions.append(len(prefix))
    return (pad_sequence(children, batch_first=True), pad_sequence(parents, batch_first=True),
            torch.tensor(positions), torch.tensor([labels[r['label']] for r in rows]),
            [r['parent'].strip().casefold() for r in rows])


class BiLSTMClassifier(nn.Module):
    def __init__(self, weights, hidden, labels, mode, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights.clone(), freeze=False, padding_idx=0)
        self.encoder = nn.LSTM(weights.shape[1], hidden, bidirectional=True, batch_first=True)
        self.mode = mode
        width = hidden * 2
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, width), nn.GELU(), nn.Linear(width, labels))

    def encode(self, tokens):
        lengths = tokens.ne(0).sum(1)
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=tokens.shape[1])
        return hidden, lengths

    def representations(self, child, parent, positions):
        hidden, lengths = self.encode(child)
        if self.mode == 'compound_mean':
            hc = (hidden * child.ne(0).unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        else:
            hc = hidden[torch.arange(len(hidden), device=hidden.device), positions]
        hp = None
        if self.mode in ('diff_minus_base', 'diff_minus_base_cl'):
            hidden, lengths = self.encode(parent)
            hp = (hidden * parent.ne(0).unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        return hc, hp

    def forward(self, child, parent, positions):
        hc, hp = self.representations(child, parent, positions)
        return self.head(hc - hp if hp is not None else hc)


def run(args):
    if min(args.epochs, args.cl_epochs, args.batch_size, args.hidden, args.dim) < 1 or args.batch_size < 2:
        raise ValueError('Invalid dimensions, epochs or batch size')
    if min(args.lr, args.cl_lr, args.temperature) <= 0 or not 0 <= args.dropout < 1:
        raise ValueError('Invalid hyperparameters')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cases)) != len(args.cases):
        raise ValueError('Seeds and cases must be unique')
    out, splits = Path(args.output), Path(args.splits)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory must be empty: {out}')
    train, dev = (read_rows(splits / f'{name}.jsonl') for name in ('train', 'dev'))
    audit = audit_splits({'train': train, 'dev': dev, 'test': []})
    names = sorted({r['label'] for r in train})
    labels = {name: i for i, name in enumerate(names)}
    if any(r['label'] not in labels for r in dev):
        raise ValueError('Unseen dev label')
    # Read unlabelled dev text to match pretrained GloVe vocabulary; never read test.
    vocab = build_vocab(train + dev)
    weights, found, glove_hash = load_glove(args.glove, vocab, args.dim)
    def coverage(rows):
        counts = Counter(t for row in rows for key in ('child_prefix', 'child_suffix', 'parent') for t in words(row[key]))
        return {'occurrences': sum(n for t, n in counts.items() if t in found) / sum(counts.values()),
                'unique': len(counts), 'covered_unique': sum(t in found for t in counts)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'protocol.json', {**vars(args), 'glove_sha256': glove_hash,
        'data_sha256': {n: fingerprint(splits/f'{n}.jsonl') for n in ('train', 'dev')},
        'selection_metric':'weighted_f1', 'test_used':False, 'vocab_uses_unlabeled_dev_text':True,
        'glove_coverage':{'train':coverage(train), 'dev':coverage(dev)}, 'labels':names})
    write_json(out/'data-audit.json', audit)
    write_json(out/'vocab.json', vocab)
    results, device = [], torch.device(args.device)
    for seed in args.seeds:
        for case in args.cases:
            print(f'\n=== {case} / seed {seed} ===', flush=True)
            random.seed(seed); torch.manual_seed(seed)
            if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
            model = BiLSTMClassifier(weights, args.hidden, len(names), case, args.dropout).to(device)
            fn = lambda rows: collate(rows, vocab, labels, case != 'compound_mean')
            loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed), collate_fn=fn)
            val = DataLoader(dev, batch_size=args.batch_size, collate_fn=fn)
            folder = out/f'{case}-seed{seed}'
            folder.mkdir()
            def move(batch):
                a, b, p, y, keys = batch
                return a.to(device), b.to(device), p.to(device), y.to(device), keys
            if case.endswith('_cl'):
                opt = torch.optim.AdamW(list(model.embedding.parameters()) + list(model.encoder.parameters()), lr=args.cl_lr)
                for epoch in range(1, args.cl_epochs+1):
                    model.train(); total = used = skipped = 0
                    for batch in loader:
                        child, parent, pos, _, keys = move(batch)
                        opt.zero_grad(set_to_none=True)
                        hc, hp = model.representations(child, parent, pos)
                        loss = contrastive_loss(hc, hp, keys, args.temperature)
                        if loss is None:
                            skipped += 1; continue
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite CL loss')
                        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                        total += loss.item(); used += 1
                    if not used: raise ValueError('No contrastive batches with distinct heads')
                    print(json.dumps({'phase':'contrastive', 'epoch':epoch, 'contrastive_loss':total/used,
                                      'batches_used':used, 'batches_skipped':skipped}), flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            best, history = -1., []
            for epoch in range(1, args.epochs+1):
                model.train(); total = count = 0
                for batch in loader:
                    child, parent, pos, y, _ = move(batch)
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(child, parent, pos), y)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    total += loss.item()*len(y); count += len(y)
                model.eval(); gold, pred = [], []
                with torch.no_grad():
                    for batch in val:
                        child, parent, pos, y, _ = move(batch)
                        gold.extend(y.tolist()); pred.extend(model(child, parent, pos).argmax(-1).tolist())
                score = metrics(gold, pred, names)
                row = {'epoch':epoch, 'train_loss':total/count, 'dev':score}
                history.append(row)
                print(json.dumps({'epoch':epoch, 'train_loss':total/count,
                                  'dev_weighted_f1':score['weighted_f1'], 'dev_macro_f1':score['macro_f1']}), flush=True)
                if score['weighted_f1'] > best:
                    best = score['weighted_f1']; best_row = row
                    torch.save(model.state_dict(), folder/'best.pt')
                write_json(folder/'history.json', history)
            results.append({'setting':case, 'seed':seed, 'best_epoch':best_row['epoch'],
                            'dev_weighted_f1':best, 'dev_macro_f1':best_row['dev']['macro_f1'],
                            'dev_accuracy':best_row['dev']['accuracy'], 'checkpoint':str(folder/'best.pt')})
            summaries, pairs = {}, {}
            for name in args.cases:
                scores = [r['dev_weighted_f1'] for r in results if r['setting'] == name]
                if scores:
                    summaries[name] = {'n':len(scores), 'dev_weighted_f1_mean':statistics.mean(scores),
                                       'dev_weighted_f1_std_sample':statistics.stdev(scores) if len(scores)>1 else None}
            for left,right in [('diff_only','compound_mean'), ('diff_minus_base','diff_only'), ('diff_minus_base_cl','diff_minus_base')]:
                a = {x['seed']:x['dev_weighted_f1'] for x in results if x['setting']==left}
                b = {x['seed']:x['dev_weighted_f1'] for x in results if x['setting']==right}
                delta = {str(s):a[s]-b[s] for s in sorted(a.keys() & b.keys())}
                if delta: pairs[left+'_minus_'+right] = {'by_seed':delta, 'mean':statistics.mean(delta.values())}
            write_json(out/'summary.json', {'conditions':summaries, 'paired_differences':pairs, 'runs':results, 'test_used':False})
    for name, score in summaries.items():
        print(f"{name:22} dev weighted F1: {score['dev_weighted_f1_mean']:.4f} | n={score['n']}")
    print(f'Summary: {out / "summary.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', default='data/splits/tratz-coarse-lexical')
    p.add_argument('--glove', required=True, help='Uncompressed GloVe text file')
    p.add_argument('--output', required=True)
    p.add_argument('--dim', type=int, default=300)
    p.add_argument('--hidden', type=int, default=150, help='Per-direction hidden size')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--cl-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--cl-lr', type=float, default=.001)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--seeds', nargs='+', type=int, default=[13, 42, 77])
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--device', choices=('cpu','cuda','mps'), default='cuda')
    run(p.parse_args())


if __name__ == '__main__':
    main()
