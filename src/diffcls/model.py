import torch
from torch import nn

MODES = ("diff_cls", "concat_cls", "pair_interaction", "cls_cls", "diff_only", "modifier_only", "compound_cls", "marked_cls", "marked_cls_diff")


def feature_vector(child, parent, mode):
    if mode in ("diff_cls", "cls_cls", "marked_cls_diff"):
        return child - parent
    if mode == "concat_cls":
        return torch.cat((child, parent), dim=-1)
    if mode == "pair_interaction":
        return torch.cat((child, parent, torch.abs(child - parent), child * parent), dim=-1)
    return child


class PairClassifier(nn.Module):
    def __init__(self, encoder, num_labels, mode="diff_cls", dropout=0.1):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.encoder = encoder
        self.mode = mode
        size = encoder.config.hidden_size
        input_size = {"concat_cls": 2 * size, "pair_interaction": 4 * size}.get(mode, size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_size, size), nn.GELU(), nn.Linear(size, num_labels))

    def representations(self, child, parent, positions):
        hidden = self.encoder(**child).last_hidden_state
        index = positions if self.mode in ("diff_cls", "concat_cls", "pair_interaction", "diff_only") else torch.zeros_like(positions)
        hc = hidden[torch.arange(hidden.size(0), device=hidden.device), index]
        hp = self.encoder(**parent).last_hidden_state[:, 0] if self.mode in ("diff_cls", "concat_cls", "pair_interaction", "cls_cls", "marked_cls_diff") else None
        return hc, hp

    def forward(self, child, parent, positions):
        hc, hp = self.representations(child, parent, positions)
        return self.head(feature_vector(hc, hp, self.mode))


def contrastive_loss(child, parent, parent_keys, temperature):
    """Symmetric multi-positive InfoNCE: every occurrence of the same head is positive.

    Returns None if there are no negative heads in the batch. Logits use L2-normalized
    raw representations; downstream features use the unnormalized states.
    """
    if len(parent_keys) != len(child) or temperature <= 0:
        raise ValueError("Invalid contrastive inputs")
    if len(set(parent_keys)) < 2:
        return None
    logits = torch.nn.functional.normalize(child, dim=-1) @ torch.nn.functional.normalize(parent, dim=-1).T
    logits = logits / temperature
    mask = torch.tensor([[a == b for b in parent_keys] for a in parent_keys], dtype=torch.bool, device=logits.device)
    def loss_axis(scores, positives):
        return (torch.logsumexp(scores, dim=1) - torch.logsumexp(scores.masked_fill(~positives, float('-inf')), dim=1)).mean()
    return (loss_axis(logits, mask) + loss_axis(logits.T, mask.T)) / 2


class Collator:
    def __init__(self, tokenizer, labels, mode, max_length=128):
        self.tokenizer, self.labels, self.mode, self.max_length = tokenizer, labels, mode, max_length
        if tokenizer.cls_token_id is None or tokenizer.padding_side != "right":
            raise ValueError("Requires a CLS-based tokenizer with right padding")

    def __call__(self, rows):
        if self.mode == "modifier_only":
            text = [r["modifier"] for r in rows]
        elif self.mode in ("cls_cls", "compound_cls"):
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
        if self.mode in ("diff_cls", "concat_cls", "pair_interaction", "diff_only"):
            mask = child["input_ids"] == self.tokenizer.convert_tokens_to_ids("[DIFF]")
            if not torch.all(mask.sum(1) == 1):
                raise ValueError("Each child requires exactly one registered DIFF token")
            positions = mask.long().argmax(1)
        targets = torch.tensor([self.labels[r["label"]] for r in rows]) if all("label" in r for r in rows) else None
        return child, parent, positions, targets, [r["id"] for r in rows]
