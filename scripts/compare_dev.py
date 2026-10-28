"""Compare multiple seeds on a fixed train/dev split. Never reads test.jsonl."""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

MODES = ("diff_cls", "concat_cls", "pair_interaction", "cls_cls", "diff_only", "modifier_only")


def build_summary(results):
    by_mode = {}
    for row in results:
        by_mode.setdefault(row["mode"], []).append(row)
    summaries = {}
    for mode, rows in by_mode.items():
        values = [r["best_dev_macro_f1"] for r in rows]
        summaries[mode] = {
            "runs": len(rows), "macro_f1_mean": statistics.mean(values),
            "macro_f1_std_sample": statistics.stdev(values) if len(values) > 1 else None,
            "seeds": [r["seed"] for r in rows], "best_epochs": [r["best_epoch"] for r in rows],
        }
    paired = {}
    proposed = {r["seed"]: r["best_dev_macro_f1"] for r in by_mode.get("diff_cls", [])}
    for mode, rows in by_mode.items():
        if mode == "diff_cls":
            continue
        diffs = {str(r["seed"]): proposed[r["seed"]] - r["best_dev_macro_f1"]
                 for r in rows if r["seed"] in proposed}
        if diffs:
            paired[mode] = {"n": len(diffs), "mean_diff_cls_minus_baseline": statistics.mean(diffs.values()),
                            "differences_by_seed": diffs}
    return {"runs": results, "by_mode": summaries, "paired_differences": paired,
            "note": "Development only. Best epoch selected independently per run. Test set not accessed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="data/splits/tratz-coarse-lexical")
    parser.add_argument("--output", required=True, help="New or empty output directory")
    parser.add_argument("--model", default="prajjwal1/bert-tiny")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 77])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["diff_cls", "concat_cls"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--tiny", action="store_true", help="Random tiny model for engineering tests only")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--contrastive-epochs", type=int, default=0,
                        help="Alignment epochs for diff_cls and concat_cls; other baselines remain single-stage")
    parser.add_argument("--contrastive-lr", type=float, default=2e-5)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("Seeds and modes must be unique")
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    if args.contrastive_epochs < 0 or args.contrastive_lr <= 0 or args.contrastive_temperature <= 0:
        parser.error("contrastive epochs must be nonnegative; learning rate and temperature must be positive")
    splits, output = Path(args.splits), Path(args.output)
    for name in ("train.jsonl", "dev.jsonl"):
        if not (splits / name).is_file():
            parser.error(f"Missing {splits / name}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            run = output / f"{mode}-seed{seed}"
            cmd = [sys.executable, "-m", "diffcls.cli", "train", "--train", str(splits / "train.jsonl"),
                   "--dev", str(splits / "dev.jsonl"), "--output", str(run), "--model", args.model,
                   "--mode", mode, "--seed", str(seed), "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                   "--max-length", str(args.max_length), "--device", args.device]
            if args.tiny:
                cmd.append("--tiny")
            if args.local_files_only:
                cmd.append("--local-files-only")
            if mode in ("diff_cls", "concat_cls") and args.contrastive_epochs:
                cmd += ["--contrastive-epochs", str(args.contrastive_epochs),
                        "--contrastive-lr", str(args.contrastive_lr),
                        "--contrastive-temperature", str(args.contrastive_temperature)]
            print(f"\n=== {mode} / seed {seed} ===", flush=True)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"Run failed; completed results remain in {output}. Use a new output directory to retry.", file=sys.stderr)
                raise
            history = json.loads((run / "history.json").read_text())
            best = max(history, key=lambda r: r["dev"]["macro_f1"])
            results.append({"mode": mode, "seed": seed, "best_epoch": best["epoch"],
                            "best_dev_macro_f1": best["dev"]["macro_f1"],
                            "best_dev_accuracy": best["dev"]["accuracy"],
                            "best_dev_weighted_f1": best["dev"].get("weighted_f1"),
                            "last_epoch": history[-1]["epoch"],
                            "last_dev_macro_f1": history[-1]["dev"]["macro_f1"],
                            "checkpoint": str(run / "best")})
            (output / "summary.json").write_text(json.dumps(build_summary(results), ensure_ascii=False, indent=2) + "\n")
    summary = build_summary(results)
    print("\n=== Development results (sample standard deviation) ===")
    for mode, s in summary["by_mode"].items():
        sd = "n/a" if s["macro_f1_std_sample"] is None else f"{s['macro_f1_std_sample']:.4f}"
        print(f"{mode:14s} macro-F1 {s['macro_f1_mean']:.4f} ± {sd} | best epochs {s['best_epochs']}")
    print(f"Results: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
