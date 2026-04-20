#!/usr/bin/env python3
"""
LANO dataset generator (Hierarchical Language Structure, CFG-based)

Generates sequences from a small probabilistic context-free grammar as
described (cfg3f baseline and two variants cfg3k/cfg3j). Sequences are
composed entirely of terminal symbols {1,2,3}. Text uses a single <bos>
token followed by terminals.

Grammar summary (core idea):
  - Start symbol (root NT): 22
  - 22 expands uniformly into one of four rules (each with prob 1/4):
      22 -> 20 21
      22 -> 20 19 21
      22 -> 21 19 19
      22 -> 20 20
  - Non-terminals {21,20} further expand with uniformly chosen rules that
    mirror the structure above. NT 19 is the base category and collapses
    to a terminal. After reaching the configured expansion depth (levels),
    remaining NTs collapse to 19, then every 19 is replaced by a terminal
    token in {1,2,3} sampled uniformly.

Variants (depth/levels):
  - cfg3f: levels=3  (baseline)
  - cfg3k: levels=4  (deeper, longer, harder)
  - cfg3j: levels=2  (shallower, shorter)

CLI examples:
  # JSONL streaming with paper-like cfg3f settings
  python lano.py --variant cfg3f --num 1000 > lano3f.jsonl

  # Single JSON array with cfg3k
  python lano.py --variant cfg3k --num 200 --format json --output lano3k.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Iterable, List, Sequence


ROOT = 22
NT_21 = 21
NT_20 = 20
NT_19 = 19


def choose_root_expansion(rng: random.Random) -> List[int]:
    # Four options with prob 1/4 each
    opts = [
        [NT_20, NT_21],
        [NT_20, NT_19, NT_21],
        [NT_21, NT_19, NT_19],
        [NT_20, NT_20],
    ]
    return list(rng.choice(opts))


def choose_21_expansion(rng: random.Random) -> List[int]:
    # Mirror the root patterns but bias toward reducing to 19s
    opts = [
        [NT_20, NT_21],
        [NT_19, NT_21],
        [NT_21, NT_19, NT_19],
        [NT_20, NT_20],
    ]
    return list(rng.choice(opts))


def choose_20_expansion(rng: random.Random) -> List[int]:
    # Reduce further toward 19s; still allow some branching
    opts = [
        [NT_19, NT_20],
        [NT_20, NT_19],
        [NT_19, NT_19, NT_20],
        [NT_19, NT_19],
    ]
    return list(rng.choice(opts))


def expand_cfg(rng: random.Random, levels: int) -> List[int]:
    """Expand from ROOT to a sequence of NT ids; stop at depth=levels.

    At max depth, collapse any further NTs to NT_19. Returns a flat list of
    NT ids (typically many 19s).
    """
    def rec(symbol: int, depth: int) -> List[int]:
        if depth >= levels:
            # Collapse to 19 at max depth to ensure termination
            return [NT_19]
        if symbol == ROOT:
            seq = choose_root_expansion(rng)
        elif symbol == NT_21:
            seq = choose_21_expansion(rng)
        elif symbol == NT_20:
            seq = choose_20_expansion(rng)
        elif symbol == NT_19:
            return [NT_19]
        else:
            # Unknown symbol; treat as 19
            return [NT_19]

        out: List[int] = []
        for s in seq:
            out.extend(rec(s, depth + 1))
        return out

    return rec(ROOT, 0)


def terminalize(rng: random.Random, nts: List[int]) -> List[str]:
    """Replace each NT (mostly 19) with a terminal token '1'|'2'|'3'."""
    return [str(rng.randint(1, 3)) for _ in nts]


def build_lano_sample(rng: random.Random, *, levels: int, ctx: int) -> dict:
    # Expand, then terminalize
    nts = expand_cfg(rng, levels)
    terms = terminalize(rng, nts)

    # Pack: ensure total tokens including <bos> fit in ctx; otherwise resample
    toks = ["<bos>"] + terms
    if len(toks) > ctx:
        # simple resample loop with a smaller effective depth if needed
        for shrink in range(levels, 1, -1):
            nts = expand_cfg(rng, shrink)
            terms = terminalize(rng, nts)
            toks = ["<bos>"] + terms
            if len(toks) <= ctx:
                break
        if len(toks) > ctx:
            raise RuntimeError(f"Unable to fit LANO sample in ctx={ctx}; consider increasing ctx or reducing levels")

    text = " ".join(toks)
    return {
        "text": text,
        "meta": {
            "variant": None,  # filled by caller
            "levels": levels,
            "ctx": ctx,
        },
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LANO CFG dataset (.jsonl or a single .json array)")
    p.add_argument("--variant", choices=["cfg3f", "cfg3k", "cfg3j"], default="cfg3f")
    p.add_argument("--levels", type=int, default=None, help="Override number of expansion levels; defaults per variant")
    p.add_argument("--ctx", type=int, default=1024, help="Context/token cap (default 1024)")
    p.add_argument("--num", type=int, default=1, help="# of samples to generate")
    p.add_argument("--seed", type=int, default=0, help="Base seed for RNG")
    p.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Output format")
    p.add_argument("--output", type=str, default="-", help="Output path or '-' for stdout")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)

    if ns.levels is None:
        defaults = {"cfg3f": 3, "cfg3k": 4, "cfg3j": 2}
        levels = defaults[ns.variant]
    else:
        levels = ns.levels

    base_seed = ns.seed
    out_fp = sys.stdout if ns.output == "-" else open(ns.output, "w", encoding="utf-8")
    close_out = ns.output != "-"

    try:
        if ns.format == "jsonl":
            for i in range(ns.num):
                sample_seed = (base_seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFF
                rng = random.Random(sample_seed)
                item = build_lano_sample(rng, levels=levels, ctx=ns.ctx)
                item["meta"]["variant"] = ns.variant
                item["meta"]["seed"] = sample_seed
                out_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_fp.flush()
        else:
            items: List[dict] = []
            for i in range(ns.num):
                sample_seed = (base_seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFF
                rng = random.Random(sample_seed)
                item = build_lano_sample(rng, levels=levels, ctx=ns.ctx)
                item["meta"]["variant"] = ns.variant
                item["meta"]["seed"] = sample_seed
                items.append(item)
            out_fp.write(json.dumps(items, ensure_ascii=False))
            out_fp.flush()
    finally:
        if close_out:
            out_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
