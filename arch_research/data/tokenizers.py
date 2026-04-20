from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


class SimpleWhitespaceTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.tok2id = dict(vocab)
        self.id2tok = {i: t for t, i in self.tok2id.items()}
        self.unk = self.tok2id.get("<unk>")

    @classmethod
    def from_tokens(cls, tokens: Iterable[str]) -> "SimpleWhitespaceTokenizer":
        ordered = sorted(set(tokens) | {"<unk>"})
        return cls({tok: i for i, tok in enumerate(ordered)})

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in text.split():
            if tok in self.tok2id:
                ids.append(self.tok2id[tok])
            elif self.unk is not None:
                ids.append(self.unk)
            else:
                raise KeyError(f"Unknown token {tok!r}")
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.id2tok.get(i, "<unk>") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tok2id)


class GPT2SubsetTokenizer:
    def __init__(self, base_tok, used_ids: List[int]):
        self.base = base_tok
        self.used_ids = list(sorted(set(used_ids)))
        self.orig2sub = {tid: i for i, tid in enumerate(self.used_ids)}
        self.sub2orig = {i: tid for i, tid in enumerate(self.used_ids)}

    @property
    def vocab_size(self) -> int:
        return len(self.used_ids)

    def encode(self, text: str) -> List[int]:
        ids = self.base.encode(text, add_special_tokens=False)
        return [self.orig2sub[i] for i in ids]

    def decode(self, ids: Sequence[int]) -> str:
        orig = [self.sub2orig[i] for i in ids]
        return self.base.decode(orig)

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        enc = self.base(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        ids = [self.orig2sub[i] for i in enc["input_ids"]]
        offsets = [(int(a), int(b)) for (a, b) in enc["offset_mapping"]]
        return ids, offsets


def build_gpt2_subset_tokenizer(
    sample_texts: Iterable[str],
    *,
    extra_texts: Sequence[str] | None = None,
) -> GPT2SubsetTokenizer:
    try:
        from transformers import GPT2TokenizerFast  # type: ignore
    except Exception as e:
        raise RuntimeError("transformers not installed; install transformers to use the gpt2 tokenizer") from e

    base = GPT2TokenizerFast.from_pretrained("gpt2")
    used: set[int] = set()
    for text in sample_texts:
        used.update(base.encode(text, add_special_tokens=False))
    for text in extra_texts or ():
        used.update(base.encode(text, add_special_tokens=False))
    return GPT2SubsetTokenizer(base, sorted(used))

