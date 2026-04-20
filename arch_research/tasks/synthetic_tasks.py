from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch

from . import brevo, depo, lano, mano
from arch_research.data.batch import TaskBatch
from arch_research.data.tokenizers import SimpleWhitespaceTokenizer
from arch_research.tasks.base import BaseTask
from arch_research.tasks.common import (
    answer_region_mask,
    build_causal_lm_batch_from_token_lists,
    compute_masked_token_accuracy,
    compute_sequence_exact_match,
)


class SyntheticWhitespaceTask(BaseTask):
    answer_metric_name = "answer_token_accuracy"
    exact_metric_name = "exact_match"

    def build_tokenizer(self, rng):
        return SimpleWhitespaceTokenizer.from_tokens(self.collect_vocab_tokens())

    def collect_vocab_tokens(self) -> Sequence[str]:
        raise NotImplementedError

    def sample_text(self, rng: random.Random, split: str) -> str:
        raise NotImplementedError

    def metric_mask_from_tokens(self, tokens: Sequence[str]) -> List[bool]:
        raise NotImplementedError

    def build_batch(self, rng, tokenizer, batch_size: int, split: str, runtime_cfg) -> TaskBatch:
        token_lists = []
        metric_masks = []
        for _ in range(batch_size):
            text = self.sample_text(rng, split)
            tokens = text.split()
            token_lists.append(tokenizer.encode(text))
            answer_mask = self.metric_mask_from_tokens(tokens)
            metric_masks.append(answer_mask[1:])
        return build_causal_lm_batch_from_token_lists(
            token_lists,
            metric_masks=metric_masks,
        )

    def compute_eval_metrics(self, batch: TaskBatch, logits, tokenizer) -> Dict[str, float]:
        metric_mask = batch.metadata["metric_mask"]
        return {
            f"eval/{self.answer_metric_name}": compute_masked_token_accuracy(logits, batch.labels, metric_mask),
            f"eval/{self.exact_metric_name}": compute_sequence_exact_match(logits, batch.labels, metric_mask),
        }


class DepoTask(SyntheticWhitespaceTask):
    name = "depo"
    answer_metric_name = "retrieval_token_accuracy"
    exact_metric_name = "query_exact_match"

    def collect_vocab_tokens(self) -> Sequence[str]:
        variant = str(self.task_args.get("variant", "depo1"))
        k = int(self.task_args.get("K", 8))
        base = set(depo.depo_vocab(variant)) | {"<bos>", "<ans>"}
        for i in range(1, k + 1):
            base.add(f"<query {i}>")
        return sorted(base)

    def sample_text(self, rng: random.Random, split: str) -> str:
        sample = depo.build_sample(
            N=int(self.task_args.get("N", 50)),
            K=int(self.task_args.get("K", 8)),
            ctx=int(self.task_args.get("ctx", 2048)),
            variant=str(self.task_args.get("variant", "depo1")),
            schedule=str(self.task_args.get("schedule", "uniform")),
            rng=rng,
            sample_seed=rng.randint(0, 2**31 - 1),
        )
        return sample.text

    def metric_mask_from_tokens(self, tokens: Sequence[str]) -> List[bool]:
        return answer_region_mask(tokens, end_markers=(), query_prefixes=("<query ",))


class ManoTask(SyntheticWhitespaceTask):
    name = "mano"
    answer_metric_name = "answer_accuracy"
    exact_metric_name = "answer_exact_match"

    def collect_vocab_tokens(self) -> Sequence[str]:
        max_l = int(self.task_args.get("L", 16))
        tag = str(self.task_args.get("len_token", "query"))
        base = {"<bos>", "<ans>", "+", "-", "*"} | {str(i) for i in range(23)}
        for i in range(1, max_l + 1):
            base.add(f"<{tag}_{i}>")
        return sorted(base)

    def sample_text(self, rng: random.Random, split: str) -> str:
        l_value = int(self.task_args.get("L", 16))
        text, _meta = mano.build_mano_sample(
            rng,
            l_value,
            int(self.task_args.get("ctx", 1024)),
            str(self.task_args.get("len_token", "query")),
        )
        return text

    def metric_mask_from_tokens(self, tokens: Sequence[str]) -> List[bool]:
        return answer_region_mask(tokens, end_markers=())


class BrevoTask(SyntheticWhitespaceTask):
    name = "brevo"
    answer_metric_name = "ancestor_token_accuracy"
    exact_metric_name = "ancestor_exact_match"

    def collect_vocab_tokens(self) -> Sequence[str]:
        variant = str(self.task_args.get("variant", "brevo1"))
        if variant == "brevo1":
            vocab = {f"v{i}" for i in range(1, int(self.task_args.get("V", 1000)) + 1)}
        else:
            vocab = {"a", "b", "c", "d"}
        vocab |= {"<bos>", "<query>", "<ans>", "<eos>"}
        return sorted(vocab)

    def sample_text(self, rng: random.Random, split: str) -> str:
        n = self.task_args.get("N")
        if n is None:
            n = 110 if str(self.task_args.get("variant", "brevo1")) == "brevo1" else 50
        text, _meta = brevo.build_brevo_sample(
            N=int(n),
            ctx=int(self.task_args.get("ctx", 1024 if str(self.task_args.get("variant", "brevo1")) == "brevo1" else 1536)),
            variant=str(self.task_args.get("variant", "brevo1")),
            schedule=str(self.task_args.get("schedule", "sqrt")),
            V=int(self.task_args.get("V", 1000)),
            rng=rng,
            sample_seed=rng.randint(0, 2**31 - 1),
        )
        return text

    def metric_mask_from_tokens(self, tokens: Sequence[str]) -> List[bool]:
        return answer_region_mask(tokens, end_markers=("<eos>",))


class LanoTask(SyntheticWhitespaceTask):
    name = "lano"
    answer_metric_name = "token_accuracy"
    exact_metric_name = "sequence_accuracy"

    def collect_vocab_tokens(self) -> Sequence[str]:
        return ["<unk>", "<bos>", "1", "2", "3"]

    def sample_text(self, rng: random.Random, split: str) -> str:
        levels = self.task_args.get("levels")
        if levels is None:
            variant = str(self.task_args.get("variant", "cfg3f"))
            levels = {"cfg3f": 3, "cfg3k": 4, "cfg3j": 2}[variant]
        item = lano.build_lano_sample(rng, levels=int(levels), ctx=int(self.task_args.get("ctx", 1024)))
        return item["text"]

    def metric_mask_from_tokens(self, tokens: Sequence[str]) -> List[bool]:
        mask = [False] * len(tokens)
        for idx in range(1, len(tokens)):
            mask[idx] = True
        return mask
