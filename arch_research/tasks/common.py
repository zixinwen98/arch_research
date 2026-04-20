from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from arch_research.data.batch import TaskBatch


def pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int = 0) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        if seq:
            out[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def build_causal_lm_batch_from_token_lists(
    token_lists: Sequence[Sequence[int]],
    *,
    metric_masks: Sequence[Sequence[bool]] | None = None,
    metadata: Dict[str, object] | None = None,
) -> TaskBatch:
    inputs = [tokens[:-1] for tokens in token_lists]
    labels = [tokens[1:] for tokens in token_lists]
    input_ids = pad_sequences(inputs, pad_id=0)
    label_tensor = torch.full_like(input_ids, -100)
    attention_mask = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)

    for idx, seq in enumerate(labels):
        if seq:
            label_tensor[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[idx, : len(seq)] = 1
            position_ids[idx, : len(seq)] = torch.arange(len(seq), dtype=torch.long)

    metric_mask_tensor = None
    if metric_masks is not None:
        metric_mask_tensor = torch.zeros_like(input_ids, dtype=torch.bool)
        for idx, seq in enumerate(metric_masks):
            if seq:
                metric_mask_tensor[idx, : len(seq)] = torch.tensor(seq, dtype=torch.bool)

    return TaskBatch(
        input_ids=input_ids,
        labels=label_tensor,
        attention_mask=attention_mask,
        position_ids=position_ids,
        metadata={} if metadata is None else metadata,
        loss_mask=None,
    ) if metric_mask_tensor is None else TaskBatch(
        input_ids=input_ids,
        labels=label_tensor,
        attention_mask=attention_mask,
        position_ids=position_ids,
        metadata={**({} if metadata is None else metadata), "metric_mask": metric_mask_tensor},
        loss_mask=None,
    )


def compute_masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    active = mask & (labels != -100)
    denom = int(active.sum().item())
    if denom == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = ((preds == labels) & active).sum().item()
    return float(correct) / float(denom)


def compute_sequence_exact_match(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    active = mask & (labels != -100)
    if active.ndim != 2:
        return 0.0
    preds = logits.argmax(dim=-1)
    valid_rows = 0
    exact = 0
    for row in range(active.shape[0]):
        row_mask = active[row]
        if int(row_mask.sum().item()) == 0:
            continue
        valid_rows += 1
        if bool(torch.equal(preds[row][row_mask], labels[row][row_mask])):
            exact += 1
    if valid_rows == 0:
        return 0.0
    return float(exact) / float(valid_rows)


def answer_region_mask(tokens: Sequence[str], *, end_markers: Sequence[str], query_prefixes: Sequence[str] = ()) -> List[bool]:
    mask = [False] * len(tokens)
    in_answer = False
    for idx, tok in enumerate(tokens):
        if tok == "<ans>":
            in_answer = True
            continue
        if tok in end_markers or any(tok.startswith(prefix) for prefix in query_prefixes):
            in_answer = False
        if in_answer:
            mask[idx] = True
    return mask

