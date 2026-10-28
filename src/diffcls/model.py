import torch
from torch import nn

MODES = ("diff_cls", "cls_cls", "diff_only", "modifier_only")


def feature_vector(child, parent, mode):
    if mode in ("diff_cls", "cls_cls"):
        return child - parent
    return child


class PairClassifier(nn.Module):
    def __init__(self, encoder, num_labels, mode="diff_cls", dropout=0.1):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.encoder = encoder
        self.mode = mode
        size = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, size), nn.GELU(), nn.Linear(size, num_labels))

    def forward(self, child, parent, positions):
        hidden = self.encoder(**child).last_hidden_state
        index = positions if self.mode in ("diff_cls", "diff_only") else torch.zeros_like(positions)
        hc = hidden[torch.arange(hidden.size(0), device=hidden.device), index]
        hp = self.encoder(**parent).last_hidden_state[:, 0] if self.mode in ("diff_cls", "cls_cls") else None
        return self.head(feature_vector(hc, hp, self.mode))


class Collator:
    def __init__(self, tokenizer, labels, mode, max_length=128):
        self.tokenizer, self.labels, self.mode, self.max_length = tokenizer, labels, mode, max_length
        if tokenizer.cls_token_id is None or tokenizer.padding_side != "right":
            raise ValueError("Requires a CLS-based tokenizer with right padding")

    def __call__(self, rows):
        if self.mode == "modifier_only":
            text = [r["modifier"] for r in rows]
        elif self.mode == "cls_cls":
            text = [r["child"] for r in rows]
        else:
            text = [r["child_prefix"] + " [DIFF] " + r["child_suffix"] for r in rows]
        def encode(strings):
            batch = self.tokenizer(strings, padding=True, truncation=False, return_tensors="pt")
            if batch["input_ids"].shape[1] > self.max_length:
                raise ValueError("Input exceeds max_length: shorten explicitly; silent truncation is disabled")
            if not torch.all(batch["input_ids"][:, 0] == self.tokenizer.cls_token_id):
                raise ValueError("CLS must be the first token")
            return dict(batch)
        child = encode(text)
        parent = encode([r["parent"] for r in rows])
        positions = torch.zeros(len(rows), dtype=torch.long)
        if self.mode in ("diff_cls", "diff_only"):
            mask = child["input_ids"] == self.tokenizer.convert_tokens_to_ids("[DIFF]")
            if not torch.all(mask.sum(1) == 1):
                raise ValueError("Each child requires exactly one registered DIFF token")
            positions = mask.long().argmax(1)
        targets = torch.tensor([self.labels[r["label"]] for r in rows]) if all("label" in r for r in rows) else None
        return child, parent, positions, targets, [r["id"] for r in rows]
