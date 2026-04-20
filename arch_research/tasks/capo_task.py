from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import torch

from . import capo
from arch_research.data.batch import TaskBatch
from arch_research.data.tokenizers import SimpleWhitespaceTokenizer, build_gpt2_subset_tokenizer
from arch_research.tasks.base import BaseTask
from arch_research.tasks.common import compute_masked_token_accuracy


class CapoTask(BaseTask):
    name = "capo"

    def __init__(self, **task_args):
        super().__init__(**task_args)
        self.scan_entries = int(self.task_args.get("scan_entries", 2000))
        self.tokenizer_name = str(self.task_args.get("tokenizer", "ws"))
        self.n_names = int(self.task_args.get("n_names", 100))

    def build_tokenizer(self, rng):
        self.db = capo.build_capo_database(rng, n_names=self.n_names)
        labels = ["birthdate", "birth_place", "school", "company", "major"]
        if self.tokenizer_name == "gpt2":
            sample_texts = [
                capo.generate_bio_entry(rng, self.db, shuffle_attrs=True)[0]
                for _ in range(self.scan_entries)
            ]
            return build_gpt2_subset_tokenizer(sample_texts, extra_texts=labels)

        tokens = set(labels) | {"<unk>"}
        collections = [
            self.db.first_names,
            self.db.last_names,
            self.db.birthdates,
            self.db.birth_places,
            self.db.schools,
            self.db.companies,
            self.db.majors,
        ]
        for values in collections:
            for value in values:
                tokens.update(str(value).split())
        return SimpleWhitespaceTokenizer.from_tokens(tokens)

    def _build_packed_batch(self, rng: random.Random, tokenizer, num_entries: int) -> TaskBatch:
        pieces: List[str] = []
        segments: List[Tuple[str, int, int]] = []
        idx = 0
        for _ in range(num_entries):
            text, _row = capo.generate_bio_entry(rng, self.db, shuffle_attrs=True)
            toks = text.split()
            pieces.extend(toks)
            start = idx
            idx += len(toks)
            segments.append((text, start, idx))

        full_text = " ".join(pieces)
        labels = ["birthdate", "birth_place", "school", "company", "major"]
        attr_masks_tokens: Dict[str, List[bool]] = {lbl: [False] * len(pieces) for lbl in labels}
        label_set = set(labels)
        for _txt, start, end in segments:
            current = None
            for i in range(start, end):
                tok = pieces[i]
                if tok in label_set:
                    current = tok
                    continue
                if current is not None:
                    attr_masks_tokens[current][i] = True

        if hasattr(tokenizer, "encode_with_offsets"):
            ids, offsets = tokenizer.encode_with_offsets(full_text)
            piece_spans: List[Tuple[int, int]] = []
            pos = 0
            for i, piece in enumerate(pieces):
                piece_spans.append((pos, pos + len(piece)))
                pos += len(piece)
                if i != len(pieces) - 1:
                    pos += 1

            def overlaps(a, b) -> bool:
                return (a[0] < b[1]) and (b[0] < a[1])

            token_level_masks: Dict[str, List[bool]] = {lbl: [False] * len(ids) for lbl in labels}
            for tok_idx, span in enumerate(offsets):
                for lbl in labels:
                    for piece_idx, flag in enumerate(attr_masks_tokens[lbl]):
                        if flag and overlaps(span, piece_spans[piece_idx]):
                            token_level_masks[lbl][tok_idx] = True
                            break
            attr_masks_tokens = token_level_masks
        else:
            ids = tokenizer.encode(full_text)

        input_ids = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
        labels_tensor = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        attr_masks = {
            lbl: torch.tensor(mask[1:], dtype=torch.bool).unsqueeze(0) for lbl, mask in attr_masks_tokens.items()
        }
        ttt_window = int(self.task_args.get("window", 128))
        ttt_config = []
        for start in range(0, input_ids.shape[1], ttt_window):
            end = min(start + ttt_window, input_ids.shape[1])
            ttt_config.append(("apply_then_update", start, end))

        return TaskBatch(
            input_ids=input_ids,
            labels=labels_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata={"attr_masks": attr_masks, "ttt_config": ttt_config},
        )

    def build_batch(self, rng, tokenizer, batch_size: int, split: str, runtime_cfg) -> TaskBatch:
        return self._build_packed_batch(rng, tokenizer, batch_size)

    def compute_eval_metrics(self, batch: TaskBatch, logits, tokenizer) -> Dict[str, float]:
        attr_masks = batch.metadata.get("attr_masks", {})
        metrics: Dict[str, float] = {}
        total_acc = []
        for label, mask in attr_masks.items():
            acc = compute_masked_token_accuracy(logits, batch.labels, mask)
            metrics[f"eval/{label}_accuracy"] = acc
            total_acc.append(acc)
        metrics["eval/value_token_accuracy"] = float(sum(total_acc) / len(total_acc)) if total_acc else 0.0
        return metrics
