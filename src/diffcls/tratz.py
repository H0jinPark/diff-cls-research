"""Convert PANIC's three-column Tratz release without changing its published splits."""
import csv
import hashlib
from pathlib import Path

from .data import audit_splits, fingerprint, read_rows, write_json, write_rows


def prepare(source, output, granularity, split_type):
    source, output = Path(source), Path(output)
    if granularity not in ("coarse", "fine") or split_type not in ("random", "lexical"):
        raise ValueError("Expected coarse/fine and random/lexical")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    source_dir = source / "classification" / "data" / f"tratz_{granularity}_grained_{split_type}"
    partitions, input_hashes = {}, {}
    for name, filename in (("train", "train.tsv"), ("dev", "val.tsv"), ("test", "test.tsv")):
        path = source_dir / filename
        input_hashes[filename] = fingerprint(path)
        rows = []
        with path.open(encoding="utf-8", newline="") as handle:
            for n, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if len(fields) != 3 or not all(x.strip() for x in fields):
                    raise ValueError(f"{path}:{n}: expected modifier, head, label")
                modifier, head, label = (v.strip() for v in fields)
                # Source fields are modifier/head, not evidence of subclass semantics.
                key = hashlib.sha256(f"{modifier}\0{head}".encode()).hexdigest()[:16]
                rows.append({"id": f"tratz-{granularity}-{key}", "parent": head,
                             "child": f"{modifier} {head}", "child_prefix": modifier,
                             "child_suffix": head, "modifier": modifier, "family": head,
                             "label": label, "source": "PANIC Tratz-derived release",
                             "source_file": filename, "source_line": n,
                             "source_granularity": granularity, "source_split": split_type})
        partitions[name] = rows
    report = audit_splits(partitions)
    train_labels = {r["label"] for r in partitions["train"]}
    missing = {r["label"] for k in ("dev", "test") for r in partitions[k]} - train_labels
    if missing:
        raise ValueError(f"Non-training labels: {sorted(missing)}")
    report.update(dataset="PANIC Tratz-derived noun compound data", granularity=granularity,
                  source_split=split_type, source_files_sha256=input_hashes,
                  note="Published splits preserved; parent=head is an experimental text representation, not an asserted ontology subclass.")
    for name, rows in partitions.items():
        write_rows(output / f"{name}.jsonl", rows)
        read_rows(output / f"{name}.jsonl")
    write_json(output / "manifest.json", report)
    return {"output": str(output), "rows": {n: len(r) for n, r in partitions.items()},
            "labels": len(train_labels), "overlap_counts": {
                pair: {field: len(values) for field, values in overlaps.items()}
                for pair, overlaps in report["overlap"].items()}}
