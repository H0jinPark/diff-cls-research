"""Run all modes with identical partitions/seeds. Test is evaluated only after dev selection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--splits", required=True)
p.add_argument("--output", required=True)
p.add_argument("--model", default="google-bert/bert-base-multilingual-cased")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 77])
p.add_argument("--tiny", action="store_true")
args = p.parse_args()
summary = []
for seed in args.seeds:
    for mode in ("diff_cls", "cls_cls", "diff_only", "modifier_only"):
        run = Path(args.output) / f"{mode}-seed{seed}"
        cmd = [sys.executable, "-m", "diffcls.cli", "train", "--train", str(Path(args.splits)/"train.jsonl"),
               "--dev", str(Path(args.splits)/"dev.jsonl"), "--output", str(run), "--mode", mode,
               "--model", args.model, "--seed", str(seed), "--epochs", str(args.epochs)]
        if args.tiny:
            cmd += ["--tiny", "--device", "cpu"]
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, "-m", "diffcls.cli", "evaluate", "--checkpoint", str(run/"best"),
                        "--data", str(Path(args.splits)/"test.jsonl"), "--output", str(run/"test")], check=True)
        metric = json.loads((run/"test"/"metrics.json").read_text())
        summary.append(dict(mode=mode, seed=seed, accuracy=metric["accuracy"], macro_f1=metric["macro_f1"]))
Path(args.output, "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
