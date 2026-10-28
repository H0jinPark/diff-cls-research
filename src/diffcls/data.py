"""Explicit lexical alignment; no guessed substring decomposition or silent truncation."""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

REQUIRED = ("id", "parent", "child", "child_prefix", "child_suffix", "modifier", "label", "family")


def read_rows(path, labeled=True):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, pairs = set(), set()
    for n, row in enumerate(rows, 1):
        for key in REQUIRED:
            if key == "label" and not labeled:
                continue
            if key not in row or not isinstance(row[key], str):
                raise ValueError(f"Row {n}: missing/string field {key}")
        for key in ("id", "parent", "child", "modifier", "family") + (("label",) if labeled else ()):
            if not row[key].strip():
                raise ValueError(f"Row {n}: empty {key}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate id: {row['id']}")
        seen.add(row["id"])
        pair = (row["parent"].strip().casefold(), row["child"].strip().casefold())
        if pair in pairs:
            raise ValueError(f"Duplicate parent-child pair: {pair}")
        pairs.add(pair)
        if any("[DIFF]" in row[k] for k in ("parent", "child", "child_prefix", "child_suffix", "modifier")):
            raise ValueError("Input must not contain reserved [DIFF] token")
        reconstructed = " ".join((row["child_prefix"] + " " + row["child_suffix"]).split())
        if reconstructed != " ".join(row["child"].split()):
            raise ValueError(f"Row {n}: prefix + suffix must reconstruct child")
        if not row["child_suffix"].strip():
            raise ValueError(f"Row {n}: child_suffix must be nonempty")
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_rows(rows, mode="family", seed=42, fractions=(0.7, 0.15, 0.15)):
    """Group-disjoint splitting; never move rows after grouping to improve label balance."""
    if mode not in ("random", "family", "head", "modifier", "lexical"):
        raise ValueError(f"Unknown split mode: {mode}")
    groups = {}
    if mode == "lexical":
        # Connected components prevent either a shared head OR modifier crossing splits.
        roots = list(range(len(rows)))
        def find(i):
            while roots[i] != i:
                roots[i] = roots[roots[i]]
                i = roots[i]
            return i
        owners = {}
        for i, r in enumerate(rows):
            for key in (r["parent"].casefold().strip(), r["modifier"].casefold().strip()):
                if key in owners:
                    roots[find(i)] = find(owners[key])
                owners[key] = i
        for i, r in enumerate(rows):
            groups.setdefault(find(i), []).append(r)
    else:
        field = {"random": "id", "family": "family", "head": "parent", "modifier": "modifier"}[mode]
        for r in rows:
            groups.setdefault(r[field].casefold().strip(), []).append(r)
    if len(groups) < 3:
        raise ValueError("Need at least 3 independent groups; lexical connectivity may prevent splitting")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    nt = min(n - 2, max(1, round(n * fractions[0])))
    nv = min(n - nt - 1, max(1, round(n * fractions[1])))
    chunks = (keys[:nt], keys[nt:nt+nv], keys[nt+nv:])
    result = {name: [r for k in chunk for r in groups[k]] for name, chunk in zip(("train", "dev", "test"), chunks)}
    known = {r["label"] for r in result["train"]}
    unseen = {r["label"] for name in ("dev", "test") for r in result[name]} - known
    if unseen:
        raise ValueError(f"Labels absent from training: {sorted(unseen)}. Add data or explicitly choose a different seed.")
    return result


def audit_splits(splits):
    report = {name: {"rows": len(rows), "labels": dict(Counter(r["label"] for r in rows))} for name, rows in splits.items()}
    overlaps = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps[f"{a}:{b}"] = {}
        for field in ("id", "parent", "modifier", "family"):
            left = {r[field].casefold().strip() for r in splits[a]}
            right = {r[field].casefold().strip() for r in splits[b]}
            overlaps[f"{a}:{b}"][field] = sorted(left & right)
        left = {(r["parent"].casefold().strip(), r["child"].casefold().strip()) for r in splits[a]}
        right = {(r["parent"].casefold().strip(), r["child"].casefold().strip()) for r in splits[b]}
        if left & right or overlaps[f"{a}:{b}"]["id"]:
            raise ValueError(f"Duplicate examples across {a}/{b}")
    report["overlap"] = overlaps
    return report


def metrics(gold, predictions, labels):
    matrix = [[0 for _ in labels] for _ in labels]
    for g, p in zip(gold, predictions):
        matrix[g][p] += 1
    by_label = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    weighted_f1 = sum(v["f1"] * v["support"] for v in by_label.values()) / len(gold)
    return dict(accuracy=sum(g == p for g, p in zip(gold, predictions)) / len(gold), weighted_f1=weighted_f1,
                macro_f1=sum(v["f1"] for v in by_label.values()) / len(labels),
                per_label=by_label, confusion_matrix=matrix, label_order=labels)
