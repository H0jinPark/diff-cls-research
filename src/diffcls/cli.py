import argparse
import json
from pathlib import Path
from .data import read_rows, write_json, write_rows, split_rows, audit_splits, fingerprint


def main():
    parser = argparse.ArgumentParser(description="DIFF-minus-CLS semantic refinement experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate canonical CSV/JSONL data")
    validate.add_argument("data")
    split = sub.add_parser("split", help="Create reproducible partitions and overlap audit")
    split.add_argument("data")
    split.add_argument("--output", required=True)
    split.add_argument("--mode", choices=["random", "family", "head", "modifier", "lexical"], default="family")
    split.add_argument("--seed", type=int, default=42)
    tratz = sub.add_parser("prepare-tratz", help="Convert PANIC Tratz data preserving published splits")
    tratz.add_argument("--source", default="data/benchmarks/panic")
    tratz.add_argument("--output", required=True)
    tratz.add_argument("--granularity", choices=["coarse", "fine"], default="coarse")
    tratz.add_argument("--split-type", choices=["random", "lexical"], default="lexical")
    train_parser = sub.add_parser("train")
    for flag in ("train", "dev", "output"):
        train_parser.add_argument("--" + flag, required=True)
    train_parser.add_argument("--model", default="google-bert/bert-base-multilingual-cased")
    train_parser.add_argument("--mode", choices=["diff_cls", "concat_cls", "pair_interaction", "cls_cls", "diff_only", "modifier_only"], default="diff_cls")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--max-length", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train_parser.add_argument("--tiny", action="store_true", help="Random tiny BERT, offline engineering test ONLY")
    train_parser.add_argument("--local-files-only", action="store_true")
    train_parser.add_argument("--contrastive-epochs", type=int, default=0, help="Stage 1 alignment epochs; 0 disables")
    train_parser.add_argument("--contrastive-lr", type=float, default=2e-5)
    train_parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    for name in ("evaluate", "predict"):
        p = sub.add_parser(name)
        for flag in ("checkpoint", "data", "output"):
            p.add_argument("--" + flag, required=True)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            rows = read_rows(args.data)
            print(json.dumps({"rows": len(rows), "labels": sorted({r['label'] for r in rows})}, ensure_ascii=False))
        elif args.command == "split":
            output = Path(args.output)
            if output.exists() and any(output.iterdir()):
                raise ValueError("Split output directory must be empty")
            rows = read_rows(args.data)
            partitions = split_rows(rows, args.mode, args.seed)
            report = audit_splits(partitions)
            report.update(mode=args.mode, seed=args.seed, source_sha256=fingerprint(args.data))
            for name, part in partitions.items():
                write_rows(output / f"{name}.jsonl", part)
            write_json(output / "manifest.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "prepare-tratz":
            from .tratz import prepare
            print(json.dumps(prepare(args.source, args.output, args.granularity, args.split_type), ensure_ascii=False, indent=2))
        elif args.command == "train":
            from .engine import train
            print(train(args))
        else:
            from .engine import evaluate
            evaluate(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
