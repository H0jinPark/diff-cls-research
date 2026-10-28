"""Predeclared Tratz experiment: weighted-F1 selection and final test evaluation."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from diffcls.data import audit_splits, fingerprint, read_rows, write_json

EXPECTED = {
    'train': ('train.tsv', '70cc6072814b37aff7d0e03164fca864917fd18b3eed2bef97b380ac8fe8d4a2'),
    'dev': ('val.tsv', 'ebebef9b165633bdc0c7216651eea441a8ee828d4947e318f2997854da968124'),
    'test': ('test.tsv', 'a89395fab79b4594a08f7cb853fcdc313b73592d329dfdd8b4c01d01d49b579f'),
}
SETTINGS = [
    ('diff_cls', 'diff_cls', False), ('cls_cls', 'cls_cls', False),
    ('concat_cls', 'concat_cls', False), ('pair_interaction', 'pair_interaction', False),
    ('diff_only', 'diff_only', False), ('modifier_only', 'modifier_only', False),
    ('diff_cls_cl', 'diff_cls', True), ('concat_cls_cl', 'concat_cls', True),
]


def verify_data(splits, source, allow_demo):
    data = {n: read_rows(splits / f'{n}.jsonl') for n in ('train', 'dev', 'test')}
    audit_splits(data)
    if allow_demo:
        return data
    source = source / 'classification/data/tratz_coarse_grained_lexical'
    for name, (file_name, expected_hash) in EXPECTED.items():
        path = source / file_name
        if fingerprint(path) != expected_hash:
            raise ValueError(f'Source file does not match pinned PANIC data: {path}')
        with path.open(encoding='utf-8', newline='') as f:
            original = list(csv.reader(f, delimiter='\t'))
        mapped = data[name]
        if len(original) != len(mapped):
            raise ValueError(f'{name} size mismatch: {len(original)} vs {len(mapped)}')
        for n, (record, row) in enumerate(zip(original, mapped), 1):
            if len(record) != 3 or tuple(record) != (row['modifier'], row['parent'], row['label']):
                raise ValueError(f'{name} line {n} differs from pinned PANIC data')
            if row['child'] != f"{row['modifier']} {row['parent']}" or row['child_prefix'] != row['modifier'] or row['child_suffix'] != row['parent']:
                raise ValueError(f'{name} line {n} has altered input formatting')
    return data


def build_report(results, settings):
    grouped = {}
    for r in results:
        grouped.setdefault(r['setting'], []).append(r)
    out = {}
    for key, rows in grouped.items():
        values = [r['test']['weighted_f1'] for r in rows]
        out[key] = {'n': len(rows), 'test_weighted_f1_mean': statistics.mean(values),
                    'test_weighted_f1_std_sample': statistics.stdev(values) if len(values) > 1 else None,
                    'test_macro_f1_mean': statistics.mean(r['test']['macro_f1'] for r in rows),
                    'test_accuracy_mean': statistics.mean(r['test']['accuracy'] for r in rows),
                    'best_epochs': [r['best_epoch'] for r in rows]}
    return {'protocol': settings, 'by_setting': out, 'runs': results,
            'reference': {'paper': 'Shwartz & Waterson (2018), Tratz coarse Lexfull, Int-NC',
                          'test_weighted_f1': 0.478,
                          'note': 'Published 2018 result, not a verified current SOTA. Comparison uses same split and weighted F1; original model, early stopping and resources differ.',
                          'paper_url': 'https://aclanthology.org/N18-2035/',
                          'code_url': 'https://github.com/tensorflow/models/blob/archive/research/lexnet_nc/lexnet_common.py'}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', default='data/splits/tratz-coarse-lexical')
    p.add_argument('--source', default='data/benchmarks/panic')
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='google-bert/bert-large-uncased')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[13, 42, 77])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max-length', type=int, default=32)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    p.add_argument('--contrastive-epochs', type=int, default=3)
    p.add_argument('--contrastive-lr', type=float, default=2e-5)
    p.add_argument('--contrastive-temperature', type=float, default=.1)
    p.add_argument('--tiny', action='store_true', help='Random tiny model for testing only')
    p.add_argument('--allow-demo', action='store_true', help='Skip PANIC verification; engineering test only')
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.epochs, args.batch_size, args.contrastive_epochs) < 1:
        p.error('Seeds must be unique; epochs and batch-size must be positive')
    if not math.isfinite(args.lr) or not math.isfinite(args.contrastive_lr) or not math.isfinite(args.contrastive_temperature) or min(args.lr, args.contrastive_lr, args.contrastive_temperature) <= 0:
        p.error('Learning rates and temperature must be finite positive numbers')
    output, splits = Path(args.output), Path(args.splits)
    if output.exists() and any(output.iterdir()):
        p.error(f'Output directory must be new or empty: {output}')
    verify_data(splits, Path(args.source), args.allow_demo)
    output.mkdir(parents=True, exist_ok=True)
    protocol = vars(args).copy()
    protocol.update(selection_metric='weighted_f1', original_source_sha256={k: v[1] for k,v in EXPECTED.items()} if not args.allow_demo else None,
                    split_sha256={n: fingerprint(splits / f'{n}.jsonl') for n in ('train','dev','test')},
                    warning='Demo / non-benchmark' if args.allow_demo or args.tiny else None)
    write_json(output / 'protocol.json', protocol)
    completed = []
    # All model and seed choices are fixed before any test scores are computed.
    for seed in args.seeds:
        for name, mode, cl in SETTINGS:
            run = output / f'{name}-seed{seed}'
            cmd = [sys.executable, '-m', 'diffcls.cli', 'train', '--train', str(splits/'train.jsonl'),
                   '--dev', str(splits/'dev.jsonl'), '--output', str(run), '--model', args.model,
                   '--mode', mode, '--epochs', str(args.epochs), '--seed', str(seed),
                   '--batch-size', str(args.batch_size), '--lr', str(args.lr), '--max-length', str(args.max_length),
                   '--device', args.device, '--selection-metric', 'weighted_f1']
            if cl:
                cmd += ['--contrastive-epochs', str(args.contrastive_epochs), '--contrastive-lr', str(args.contrastive_lr),
                        '--contrastive-temperature', str(args.contrastive_temperature)]
            if args.tiny:
                cmd.append('--tiny')
            print(f'\n=== TRAIN {name} / seed {seed} ===', flush=True)
            subprocess.run(cmd, check=True)
            history = json.loads((run/'history.json').read_text())
            best = max(history, key=lambda x: x['dev']['weighted_f1'])
            completed.append({'setting': name, 'mode': mode, 'seed': seed, 'contrastive': cl,
                              'best_epoch': best['epoch'], 'dev': best['dev'], 'run_dir': str(run)})
            write_json(output/'train-progress.json', {'completed': completed, 'expected': len(args.seeds)*len(SETTINGS)})
    results = []
    for row in completed:
        run = Path(row['run_dir'])
        print(f"\n=== TEST {row['setting']} / seed {row['seed']} ===", flush=True)
        subprocess.run([sys.executable, '-m', 'diffcls.cli', 'evaluate', '--checkpoint', str(run/'best'),
                        '--data', str(splits/'test.jsonl'), '--output', str(run/'paper-test'),
                        '--device', args.device, '--batch-size', str(args.batch_size)], check=True)
        result = json.loads((run/'paper-test/metrics.json').read_text())
        if result['data_sha256'] != protocol['split_sha256']['test']:
            raise ValueError('Test dataset changed during run')
        results.append({**row, 'test': result})
        write_json(output/'paper-results.json', build_report(results, protocol))
    report = build_report(results, protocol)
    print('\n=== Test weighted F1 (mean ± sample standard deviation; predeclared settings) ===')
    for name, row in report['by_setting'].items():
        sd = row['test_weighted_f1_std_sample']
        print(f"{name:17} {row['test_weighted_f1_mean']:.4f} ± {sd:.4f}" if sd is not None else f"{name:17} {row['test_weighted_f1_mean']:.4f} (one seed)")
    print(f"Report: {output / 'paper-results.json'}")


if __name__ == '__main__':
    main()
