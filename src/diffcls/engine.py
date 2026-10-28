import json
import random
import platform
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizerFast

from .data import read_rows, write_json, write_rows, metrics, audit_splits, fingerprint
from .model import PairClassifier, Collator


def device_for(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    child, parent, pos, targets, ids = batch
    return ({k: v.to(device) for k, v in child.items()},
            {k: v.to(device) for k, v in parent.items()}, pos.to(device),
            targets.to(device) if targets is not None else None, ids)


def run_eval(model, loader, device, labels):
    model.eval()
    gold, predicted, records = [], [], []
    with torch.no_grad():
        for batch in loader:
            child, parent, pos, targets, ids = move(batch, device)
            probs = model(child, parent, pos).softmax(-1).cpu()
            predictions = probs.argmax(-1).tolist()
            predicted.extend(predictions)
            if targets is not None:
                gold.extend(targets.cpu().tolist())
            for i, p, scores in zip(ids, predictions, probs.tolist()):
                records.append({"id": i, "prediction": labels[p], "probabilities_uncalibrated": dict(zip(labels, scores))})
    return (metrics(gold, predicted, labels) if gold else None), records


def save_checkpoint(path, model, tokenizer, config):
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "tokenizer")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
    write_json(path / "experiment.json", config)


def load_checkpoint(path, device):
    path = Path(path)
    config = json.loads((path / "experiment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(path / "encoder", local_files_only=True))
    model = PairClassifier(encoder, len(config["labels"]), config["mode"], config["dropout"])
    model.load_state_dict(load_file(str(path / "model.safetensors")))
    return model.to(device), tokenizer, config


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.max_length < 4:
        raise ValueError("epochs/batch_size/lr/max_length must be positive and max_length >= 4")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output must be empty to avoid overwriting a run: {out}")
    train_rows, dev_rows = read_rows(args.train), read_rows(args.dev)
    audit = audit_splits({"train": train_rows, "dev": dev_rows, "test": []})
    labels = sorted({r["label"] for r in train_rows})
    if any(r["label"] not in labels for r in dev_rows):
        raise ValueError("Development labels must all occur in training")
    seed_everything(args.seed)
    device = device_for(args.device)
    out.mkdir(parents=True, exist_ok=True)
    if args.tiny:
        # Random, local BERT: engineering smoke test only, not a language model baseline.
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        vocab += sorted({word.casefold() for r in train_rows for k in ("parent", "child", "modifier") for word in r[k].split()} - set(vocab))
        vocab_path = out / "tiny-vocab.txt"
        vocab_path.write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
        encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=64,
                                      max_position_embeddings=args.max_length))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        encoder = AutoModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    if encoder.config.model_type != "bert":
        raise ValueError("This experiment requires a BERT encoder; choose a BERT checkpoint")
    if args.max_length > encoder.config.max_position_embeddings:
        raise ValueError("max_length exceeds the encoder position limit")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[DIFF]"]})
    encoder.resize_token_embeddings(len(tokenizer))
    model = PairClassifier(encoder, len(labels), args.mode, args.dropout).to(device)
    collate = Collator(tokenizer, {v: i for i, v in enumerate(labels)}, args.mode, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    dev_loader = DataLoader(dev_rows, batch_size=args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(labels=labels, device_used=str(device), torch_version=torch.__version__,
                  transformers_version=transformers.__version__, python_version=platform.python_version(),
                  train_sha256=fingerprint(args.train), dev_sha256=fingerprint(args.dev))
    write_json(out / "data-audit.json", audit)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            child, parent, pos, targets, _ = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(child, parent, pos), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        result, _ = run_eval(model, dev_loader, device, labels)
        history.append(dict(epoch=epoch, train_loss=total / count, dev=result))
        print(json.dumps({"epoch": epoch, "train_loss": total / count, "dev_macro_f1": result["macro_f1"]}), flush=True)
        if result["macro_f1"] > best:
            best = result["macro_f1"]
            config["best_epoch"] = epoch
            save_checkpoint(out / "best", model, tokenizer, config)
        write_json(out / "history.json", history)
    return out / "best"


def evaluate(args):
    device = device_for(args.device)
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    rows = read_rows(args.data, labeled=args.command == "evaluate")
    label_map = {v: i for i, v in enumerate(config["labels"])}
    if any("label" in r and r["label"] not in label_map for r in rows):
        raise ValueError("Unknown label in evaluation data")
    collate = Collator(tokenizer, label_map, config["mode"], config["max_length"])
    loader = DataLoader(rows, batch_size=args.batch_size, collate_fn=collate)
    result, records = run_eval(model, loader, device, config["labels"])
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output must be empty: {out}")
    if result is not None:
        result.update(data_sha256=fingerprint(args.data), checkpoint=str(Path(args.checkpoint).resolve()))
        write_json(out / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    write_rows(out / "predictions.jsonl", records)
