"""Matched-backbone marker and feature controls. Never evaluate on the test split."""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from diffcls.data import audit_splits, fingerprint, read_rows, write_json

CASES = {
    'compound_cls': ('compound_cls', False),
    'marked_cls': ('marked_cls', False),
    'diff_only': ('diff_only', False),
    'marked_cls_diff': ('marked_cls_diff', False),
    'diff_cls': ('diff_cls', False),
    'diff_cls_cl': ('diff_cls', True),
}
PAIRS = [('marked_cls', 'compound_cls'), ('diff_only', 'marked_cls'),
         ('diff_cls', 'marked_cls_diff'), ('diff_cls', 'diff_only'), ('diff_cls_cl', 'diff_cls')]


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['setting'], []).append(row)
    summary = {}
    for name, values in grouped.items():
        scores = [x['dev_weighted_f1'] for x in values]
        summary[name] = {'n': len(scores), 'dev_weighted_f1_mean': statistics.mean(scores),
                         'dev_weighted_f1_std_sample': statistics.stdev(scores) if len(scores) > 1 else None,
                         'dev_macro_f1_mean': statistics.mean(x['dev_macro_f1'] for x in values),
                         'best_epochs': [x['best_epoch'] for x in values]}
    pairs = {}
    for a, b in PAIRS:
        left = {x['seed']: x['dev_weighted_f1'] for x in grouped.get(a, [])}
        right = {x['seed']: x['dev_weighted_f1'] for x in grouped.get(b, [])}
        both = sorted(left.keys() & right.keys())
        if both:
            diffs = {str(k): left[k] - right[k] for k in both}
            pairs[f'{a}_minus_{b}'] = {'differences_by_seed': diffs, 'mean': statistics.mean(diffs.values())}
    return {'conditions': summary, 'paired_weighted_f1_differences': pairs, 'runs': rows,
            'note': 'Validation only; no test scores. Head inputs and/or training stages differ as documented.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', default='data/splits/tratz-coarse-lexical')
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='google-bert/bert-large-uncased')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[13, 42, 77])
    p.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max-length', type=int, default=32)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    p.add_argument('--contrastive-epochs', type=int, default=3)
    p.add_argument('--contrastive-lr', type=float, default=2e-5)
    p.add_argument('--contrastive-temperature', type=float, default=.1)
    p.add_argument('--tiny', action='store_true', help='Random tiny BERT for engineering smoke tests only')
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or len(args.cases) != len(set(args.cases)):
        p.error('seeds and cases must be unique')
    if min(args.epochs, args.batch_size, args.contrastive_epochs) < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.lr, args.contrastive_lr, args.contrastive_temperature)):
        p.error('epochs, batch-size, and learning rates must be positive')
    splits, output = Path(args.splits), Path(args.output)
    if output.exists() and any(output.iterdir()):
        p.error(f'Output must be empty: {output}')
    train = read_rows(splits/'train.jsonl')
    dev = read_rows(splits/'dev.jsonl')
    audit = audit_splits({'train': train, 'dev': dev, 'test': []})
    hashes = {x: fingerprint(splits/f'{x}.jsonl') for x in ('train', 'dev')}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', {**vars(args), 'data_sha256': hashes, 'selection_metric': 'weighted_f1',
                                      'test_used': False, 'warning': 'tiny random encoder' if args.tiny else None})
    write_json(output/'data-audit.json', audit)
    results = []
    for seed in args.seeds:
        for name in args.cases:
            mode, use_cl = CASES[name]
            run = output/f'{name}-seed{seed}'
            cmd = [sys.executable, '-m', 'diffcls.cli', 'train', '--train', str(splits/'train.jsonl'),
                   '--dev', str(splits/'dev.jsonl'), '--output', str(run), '--model', args.model,
                   '--mode', mode, '--epochs', str(args.epochs), '--seed', str(seed),
                   '--batch-size', str(args.batch_size), '--lr', str(args.lr), '--max-length', str(args.max_length),
                   '--device', args.device, '--selection-metric', 'weighted_f1']
            if use_cl:
                cmd += ['--contrastive-epochs', str(args.contrastive_epochs), '--contrastive-lr', str(args.contrastive_lr),
                        '--contrastive-temperature', str(args.contrastive_temperature)]
            if args.tiny:
                cmd.append('--tiny')
            if args.local_files_only:
                cmd.append('--local-files-only')
            print(f'\n=== {name} / seed {seed} ===', flush=True)
            subprocess.run(cmd, check=True)
            history = json.loads((run/'history.json').read_text())
            best = max(history, key=lambda r: r['dev']['weighted_f1'])
            results.append({'setting': name, 'seed': seed, 'mode': mode, 'contrastive': use_cl,
                            'best_epoch': best['epoch'], 'dev_weighted_f1': best['dev']['weighted_f1'],
                            'dev_macro_f1': best['dev']['macro_f1'], 'dev_accuracy': best['dev']['accuracy'],
                            'checkpoint': str(run/'best')})
            write_json(output/'summary.json', summarize(results))
    for name, data in summarize(results)['conditions'].items():
        print(f"{name:18} dev weighted F1: {data['dev_weighted_f1_mean']:.4f} | n={data['n']}")
    print(f'Summary: {output / "summary.json"}')


if __name__ == '__main__':
    main()
