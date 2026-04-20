#!/usr/bin/env python3
"""
Train GPT_TTT on online-generated CAPO biography entries.

Approach
  - Build a fixed CAPO database (deterministic seed).
  - In each training step, sample 128 entries online from the fixed DB.
  - Concatenate them into a single packed sequence with <bos>/<eos> per entry.
  - Build a ttt_config that applies-then-updates the fast weights per entry so
    that later entries in the same batch see updated fast weights.

This script uses a tiny whitespace tokenizer derived from the database values,
so it is self-contained (no external tokenizer dependency).
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from tttlab.models.gpt_ttt import Config as TTTConfig, GPT_TTT
from tttlab.tasks.capo import build_capo_database, generate_bio_entry


# ------------------------------ Tokenizer ---------------------------------- #


class WhiteSpaceTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.tok2id = dict(vocab)
        self.id2tok = {i: t for t, i in self.tok2id.items()}
        self.unk = self.tok2id.get("<unk>", None)

    @classmethod
    def from_capo_db(cls, db) -> "WhiteSpaceTokenizer":
        # Build vocabulary from all candidate tokens (split on whitespace)
        specials = ["<bos>", "<eos>", ",", ".", "<unk>"]
        labels = ["birthdate", "birth_place", "school", "company", "major"]

        def add_all(container: List[str], acc: set[str]):
            for item in container:
                for tok in str(item).split():
                    acc.add(tok)

        acc: set[str] = set(labels + specials)
        add_all(db.first_names, acc)
        add_all(db.last_names, acc)
        add_all(db.birthdates, acc)
        add_all(db.birth_places, acc)
        add_all(db.schools, acc)
        add_all(db.companies, acc)
        add_all(db.majors, acc)

        vocab_list = list(sorted(acc))
        vocab = {tok: i for i, tok in enumerate(vocab_list)}
        return cls(vocab)

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for tok in text.split():
            if tok in self.tok2id:
                out.append(self.tok2id[tok])
            elif self.unk is not None:
                out.append(self.unk)
            else:
                raise KeyError(f"OOV token without <unk>: {tok}")
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.id2tok.get(i, "<unk>") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tok2id)


# ------------------------------- Utilities --------------------------------- #


class GPT2SubsetTokenizer:
    """GPT-2 tokenizer wrapped to a compact subset of tokens used by CAPO.

    - Builds a mapping from original GPT-2 ids -> compact ids [0..K-1].
    - Provides encode/decode and encode_with_offsets (for value-mask alignment).
    """

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


def build_gpt2_subset_tokenizer(rng: random.Random, db, *, scan_entries: int = 2000) -> GPT2SubsetTokenizer:
    try:
        from transformers import GPT2TokenizerFast  # type: ignore
    except Exception as e:
        raise RuntimeError("transformers not installed; pip install transformers to use --tokenizer gpt2") from e

    base = GPT2TokenizerFast.from_pretrained("gpt2")

    used: set[int] = set()
    # Scan synthetic entries from fixed DB to collect used token ids
    for _ in range(scan_entries):
        text, _ = generate_bio_entry(rng, db, shuffle_attrs=True)
        used.update(base.encode(text, add_special_tokens=False))

    # Always include labels and specials
    for lbl in ["birthdate", "birth_place", "school", "company", "major"]:
        used.update(base.encode(lbl, add_special_tokens=False))

    return GPT2SubsetTokenizer(base, sorted(used))

def build_packed_batch(
    rng: random.Random,
    *,
    db,
    tokenizer: WhiteSpaceTokenizer,
    num_entries: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, int, int]], Dict[str, torch.Tensor]]:
    """Generate a packed training sample of `num_entries` bios.

    Returns
    - input_ids: [1, T-1]
    - targets:   [1, T-1]
    - segments: list of (text, start_idx, end_idx) indices into the unshifted token stream
    """
    pieces: List[str] = []
    segments: List[Tuple[str, int, int]] = []

    idx = 0
    for _ in range(num_entries):
        text, _row = generate_bio_entry(rng, db, shuffle_attrs=True)
        # Keep dataset clean: only person name, attribute name, and value tokens
        toks = text.split()
        pieces.extend(toks)
        start = idx
        idx += len(toks)
        end = idx
        # Store the segment bounds; text is optional here
        segments.append((text, start, end))

    full_text = " ".join(pieces)

    # Build attribute value masks per token position in the unshifted stream (piece-level first)
    labels = ["birthdate", "birth_place", "school", "company", "major"]
    N = len(pieces)
    attr_masks_tokens: Dict[str, List[bool]] = {lbl: [False] * N for lbl in labels}
    label_set = set(labels)
    for _txt, s, e in segments:
        current: str | None = None
        for i in range(s, e):
            tok = pieces[i]
            if tok in label_set:
                current = tok
                continue
            # dataset is clean; no special punctuation tokens to skip
            # If new attribute label occurs later, current will switch there; for now, mark
            if current is not None:
                attr_masks_tokens[current][i] = True

    # Encode and, if available, get offsets for mask alignment
    if hasattr(tokenizer, "encode_with_offsets"):
        ids, offsets = tokenizer.encode_with_offsets(full_text)  # type: ignore[attr-defined]
        # Map piece indices to character spans in the concatenated text
        piece_spans: List[Tuple[int, int]] = []
        pos = 0
        for i, p in enumerate(pieces):
            piece_spans.append((pos, pos + len(p)))
            pos += len(p)
            if i != len(pieces) - 1:
                pos += 1  # the space separator

        def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
            return (a[0] < b[1]) and (b[0] < a[1])

        token_level_masks: Dict[str, List[bool]] = {lbl: [False] * len(ids) for lbl in labels}
        for t_idx, span in enumerate(offsets):
            for lbl in labels:
                # mark True if this token overlaps any value piece span for this label
                for i, flag in enumerate(attr_masks_tokens[lbl]):
                    if not flag:
                        continue
                    if overlaps(span, piece_spans[i]):
                        token_level_masks[lbl][t_idx] = True
                        break
        attr_masks_tokens = token_level_masks
    else:
        ids = tokenizer.encode(full_text)

    # Next-token prediction: shift by one
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)

    # Shift masks to align with targets y (which correspond to tokens[1:])
    attr_masks_tgt: Dict[str, torch.Tensor] = {
        lbl: torch.tensor(m[1:], dtype=torch.bool) for lbl, m in attr_masks_tokens.items()
    }
    return x, y, segments, attr_masks_tgt


def build_ttt_config_from_segments(
    segments: List[Tuple[str, int, int]],
    T: int,
    window: int,
) -> List[Tuple[str, int, int]]:
    """Create a TTT schedule that updates fast weights every `window` tokens.

    Ignores entry boundaries and chunks the packed sequence into contiguous
    windows of length `window` (last chunk may be shorter).
    """
    if window <= 0:
        # Single chunk over whole sequence
        return [("apply_then_update", 0, T)]
    cfg: List[Tuple[str, int, int]] = []
    for start in range(0, T, window):
        end = min(start + window, T)
        cfg.append(("apply_then_update", start, end))
    return cfg


# -------------------------------- Training --------------------------------- #


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT_TTT on CAPO biographies")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20, help="Training steps (small for test)")
    p.add_argument("--batch_entries", type=int, default=128, help="# of entries per packed batch")
    p.add_argument("--window", type=int, default=128, help="Fixed sliding window size (e.g., 128)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tokenizer", choices=["ws", "gpt2"], default="ws", help="Tokenizer: whitespace or GPT-2 subset")
    p.add_argument("--scan_entries", type=int, default=2000, help="#entries to scan for GPT-2 subset vocab")
    p.add_argument("--muon", action="store_true", help="Enable Muon-style TTT fast-weight update")
    p.add_argument("--two_phase", action="store_true", help="First half steps: optimize model; second half: TTT-only updates (no optimizer)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv or [])
    torch.manual_seed(ns.seed)
    rng = random.Random(ns.seed)

    # Fixed database + tokenizer
    db = build_capo_database(rng)
    if ns.tokenizer == "ws":
        tokenizer = WhiteSpaceTokenizer.from_capo_db(db)
    else:
        tokenizer = build_gpt2_subset_tokenizer(rng, db, scan_entries=ns.scan_entries)
    print(f"Tokenizer: {ns.tokenizer} | vocab size: {tokenizer.vocab_size}")

    # Preview a few data samples at the beginning of training
    print("Sample data (first 3 bios):")
    for i in range(3):
        txt, _ = generate_bio_entry(rng, db, shuffle_attrs=True)
        ids = []
        try:
            ids = getattr(tokenizer, 'encode')(txt)  # type: ignore[attr-defined]
        except Exception:
            pass
        print(f"  {i+1}. {txt}")
        if ids:
            print(f"     ids: {ids[:32]}{' ...' if len(ids)>32 else ''}")

    # Tiny model suitable for CPU quick test
    cfg = TTTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=6,
        n_heads=8,
        d_head=32,
        d_model=256,
        d_mlp=256,
        d_ttt=0,
        sliding_window=512,
        tie_output=True,
        use_mlp=True,
        ttt_momentum=0.0,
    )
    model = GPT_TTT(cfg).to(ns.device)
    opt = torch.optim.Adam(model.parameters(), lr=ns.lr)
    # Cosine LR schedule over optimizer steps (first half if two_phase)
    opt_steps = ns.steps if not ns.two_phase else max(1, ns.steps // 2)
    try:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(opt, T_max=opt_steps, eta_min=1e-6)
    except Exception:
        scheduler = None

    # ---- Parameter report (including fast weights) ----
    def report_params(model: GPT_TTT):
        # Count nn.Parameter trainables
        nn_params = sum(p.numel() for p in model.parameters())
        # Embedding and head
        emb_params = model.tok.weight.numel()
        head_params = model.out.weight.numel() + (0 if getattr(model.out, 'bias', None) is None else model.out.bias.numel())
        other_params = nn_params - emb_params - head_params

        # Fast weights across all LaCT layers
        fw = 0
        fwm = 0
        for m in model.modules():
            if m.__class__.__name__ == 'LaCTLayer':
                if hasattr(m, 'w1') and hasattr(m, 'w2') and hasattr(m, 'w3'):
                    fw += m.w1.numel() + m.w2.numel() + m.w3.numel()
                if hasattr(m, 'v1') and hasattr(m, 'v2') and hasattr(m, 'v3'):
                    fwm += m.v1.numel() + m.v2.numel() + m.v3.numel()

        print("Parameters (counts):")
        print(f"- token_embedding: {emb_params}")
        print(f"- output_head:    {head_params}")
        print(f"- other_params:   {other_params}")
        print(f"- nn.Parameters total: {nn_params}")
        print(f"- fast_weights (w1+w2+w3): {fw}")
        if fwm > 0:
            print(f"- fast_momentum (v1+v2+v3): {fwm}")
        print(f"- grand_total (params + fast_weights): {nn_params + fw}")

        # Per-block breakdown
        print("Per-block parameter breakdown:")
        for idx, blk in enumerate(model.blocks):
            attn_ct = 0
            mlp_ct = 0
            fast_ct = 0
            fast_mom_ct = 0
            # Attention-related params in LaCT + ln1
            lact = blk.lact
            for name in ["qkv_w", "qkv_b", "o_proj_w", "o_proj_b"]:
                t = getattr(lact, name, None)
                if t is not None:
                    attn_ct += t.numel()
            # lr_w/lr_b (controller for fast-weights): count them under attention or fast? we count under attention-ctrl
            for name in ["lr_w", "lr_b"]:
                t = getattr(lact, name, None)
                if t is not None:
                    attn_ct += t.numel()
            # ln1 params
            for p in blk.ln1.parameters():
                attn_ct += p.numel()
            # MLP-related params (if present) + ln2
            if getattr(blk, "use_mlp", False):
                if hasattr(blk, "mlp"):
                    for p in blk.mlp.parameters():
                        mlp_ct += p.numel()
                if hasattr(blk, "ln2"):
                    for p in blk.ln2.parameters():
                        mlp_ct += p.numel()
            # Fast-weights tensors (non-Parameter)
            for name in ["w1", "w2", "w3"]:
                t = getattr(lact, name, None)
                if t is not None:
                    fast_ct += t.numel()
            for name in ["v1", "v2", "v3"]:
                t = getattr(lact, name, None)
                if t is not None:
                    fast_mom_ct += t.numel()
            msg = (
                f"  block {idx}: attention={attn_ct} | mlp={mlp_ct} | "
                f"fast_weights={fast_ct}"
            )
            if fast_mom_ct:
                msg += f" | fast_momentum={fast_mom_ct}"
            print(msg)

    report_params(model)

    for step in range(ns.steps):
        x, y, segs, attr_masks = build_packed_batch(rng, db=db, tokenizer=tokenizer, num_entries=ns.batch_entries)
        x = x.to(ns.device)
        y = y.to(ns.device)

        T = x.shape[1]
        # Fixed sliding window as requested
        dyn_w = int(ns.window)
        model.cfg.sliding_window = dyn_w
        attn_mask = torch.ones(1, T, dtype=torch.int64, device=ns.device)
        # position_ids reset to 0 at each <bos>; compute from tokens
        # We can derive from text boundaries; for simplicity, set absolute 0..T-1.
        # Causal mask handles direction; varlen not required on CPU test path.
        position_ids = torch.arange(T, dtype=torch.int64, device=ns.device).unsqueeze(0)

        # Build TTT schedule by fixed-size windows
        ttt_config = build_ttt_config_from_segments(segs, T, 128)

        # Two-phase option: first half optimize model; second half TTT-only (no optimizer/backward)
        ttt_only = ns.two_phase and (step >= (ns.steps // 2))
        if ttt_only:
            with torch.no_grad():
                logits = model(
                    x,
                    attn_mask_1d=attn_mask,
                    position_ids=position_ids,
                    ttt_config=ttt_config,
                    use_muon=ns.muon,
                )
                loss_val = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
                # Metrics: per-attribute value-only losses
                per_pos = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none")
                per_pos = per_pos.view(x.shape[0], x.shape[1])
                metrics = {}
                attr_order = ["birthdate", "birth_place", "school", "company", "major"]
                vals = []
                for lbl in attr_order:
                    m = attr_masks[lbl].to(per_pos.device)
                    m_b = m.unsqueeze(0).expand(per_pos.shape[0], -1)
                    denom = m_b.sum().clamp_min(1)
                    val = (per_pos * m_b.float()).sum() / denom
                    metrics[lbl] = float(val.item())
                    vals.append(val)
                attr_avg = torch.stack(vals).mean().item()
                # Non-value loss: complement of union of all value masks
                union = torch.zeros_like(per_pos, dtype=torch.bool)
                for lbl in attr_order:
                    m = attr_masks[lbl].to(per_pos.device)
                    union |= m.unsqueeze(0).expand(per_pos.shape[0], -1)
                non_mask = ~union
                non_denom = non_mask.sum().clamp_min(1)
                non_value_loss = (per_pos * non_mask.float()).sum() / non_denom
                non_value_loss = float(non_value_loss.item())
        else:
            logits = model(
                x,
                attn_mask_1d=attn_mask,
                position_ids=position_ids,
                ttt_config=ttt_config,
                use_muon=ns.muon,
            )
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            # Metrics: per-attribute value-only losses
            with torch.no_grad():
                per_pos = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none")
                per_pos = per_pos.view(x.shape[0], x.shape[1])
                metrics = {}
                attr_order = ["birthdate", "birth_place", "school", "company", "major"]
                vals = []
                for lbl in attr_order:
                    m = attr_masks[lbl].to(per_pos.device)
                    m_b = m.unsqueeze(0).expand(per_pos.shape[0], -1)
                    denom = m_b.sum().clamp_min(1)
                    val = (per_pos * m_b.float()).sum() / denom
                    metrics[lbl] = float(val.item())
                    vals.append(val)
                attr_avg = torch.stack(vals).mean().item()
                # Non-value loss
                union = torch.zeros_like(per_pos, dtype=torch.bool)
                for lbl in attr_order:
                    m = attr_masks[lbl].to(per_pos.device)
                    union |= m.unsqueeze(0).expand(per_pos.shape[0], -1)
                non_mask = ~union
                non_denom = non_mask.sum().clamp_min(1)
                non_value_loss = (per_pos * non_mask.float()).sum() / non_denom
                non_value_loss = float(non_value_loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

        if (step + 1) % 1 == 0:
            loss_to_print = loss_val if ttt_only else float(loss.item())
            phase = "ttt_only" if ttt_only else "opt"
            # Pull current LR from optimizer
            lr_cur = opt.param_groups[0]['lr'] if not ttt_only else 0.0
            msg = [
                f"step {step+1}/{ns.steps}",
                f"phase {phase}",
                f"loss {loss_to_print:.4f}",
                f"value_loss {attr_avg:.4f}",
                f"non_value_loss {non_value_loss:.4f}",
                f"lr {lr_cur:.2e}",
            ]
            for lbl in ["birthdate", "birth_place", "school", "company", "major"]:
                msg.append(f"{lbl} {metrics[lbl]:.4f}")
            msg.append(f"T {T}")
            msg.append(f"window {dyn_w}")
            print(" | ".join(msg))

    print("Training run finished.")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
