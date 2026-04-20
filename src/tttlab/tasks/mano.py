#!/usr/bin/env python3
"""
mano / MANO dataset generator (Knowledge Manipulation)

Task: generate arithmetic prefix expressions with ℓ binary operations over
operands in [0,22], using operators {+, -, *}, and compute the result modulo 23.

Text format (space-delimited):
  <bos> <query_ℓ> expr_tokens... <ans> value

Example (ℓ=3):
  <bos> <query_3> + * 7 5 - 12 4 <ans> 17

Spec highlights (from paper):
  - Sample ℓ uniformly from [1..L] (L is max ops)
  - Operators: +, -, * (each is its own token)
  - Operands: integers 0..22 (each a single token)
  - All arithmetic is mod 23
  - Expressions generated recursively in prefix: choose op, split ℓ-1 into ℓ' and ℓ-1-ℓ',
    recursively build sub-expressions
  - Special tokens: <bos>, <ans>, and <query_ℓ> for ℓ ∈ [L]

Output JSON objects:
  {"text": "...", "meta": {"L": int, "ell": int, "ctx": int, "seed": int,
                              "len_token": "query|len", "mod": 23}}

CLI examples:
  # JSONL streaming
  python mano_dataset.py --Lset 16,13,10 --num 1000 > mano.jsonl

  # Single JSON array
  python mano_dataset.py --L 16 --num 100 --format json --output mano.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import List, Sequence, Tuple

MOD = 23
OPS = ['+', '-', '*']


def eval_op(op: str, a: int, b: int) -> int:
    if op == '+':
        return (a + b) % MOD
    if op == '-':
        return (a - b) % MOD
    if op == '*':
        return (a * b) % MOD
    raise ValueError(op)


def gen_expr_prefix(rng: random.Random, ops_remaining: int) -> Tuple[List[str], int]:
    """Generate a prefix expression with exactly ops_remaining binary operations.

    Returns (tokens, value_mod23).
    Base case ops_remaining == 0 yields a single operand token.
    """
    if ops_remaining == 0:
        x = rng.randint(0, 22)
        return [str(x)], x % MOD

    op = rng.choice(OPS)
    left_ops = rng.randint(0, ops_remaining - 1)
    right_ops = ops_remaining - 1 - left_ops
    ltoks, lval = gen_expr_prefix(rng, left_ops)
    rtoks, rval = gen_expr_prefix(rng, right_ops)
    val = eval_op(op, lval, rval)
    return [op] + ltoks + rtoks, val


def build_mano_sample(rng: random.Random, L: int, ctx: int, len_token: str) -> Tuple[str, dict]:
    if L < 1:
        raise ValueError("L must be >= 1")

    # Sample ℓ uniformly from [1..L]
    ell = rng.randint(1, L)
    expr_tokens, value = gen_expr_prefix(rng, ell)

    # Assemble tokens
    tag = 'query' if len_token == 'query' else 'len'
    toks: List[str] = ["<bos>", f"<{tag}_{ell}>"]
    toks.extend(expr_tokens)
    toks.append("<ans>")
    toks.append(str(value))

    # Ensure packing within ctx (left-aligned)
    if len(toks) > ctx:
        # Resample smaller ell until it fits; lower ell tends to reduce length 2*ell+5
        # Try up to L attempts, then raise
        for _ in range(L):
            ell = rng.randint(1, max(1, min(L, (ctx - 5) // 2)))
            expr_tokens, value = gen_expr_prefix(rng, ell)
            toks = ["<bos>", f"<{tag}_{ell}>"] + expr_tokens + ["<ans>", str(value)]
            if len(toks) <= ctx:
                break
        if len(toks) > ctx:
            raise RuntimeError(f"Unable to fit sample in ctx={ctx}; consider increasing ctx or reducing L")

    text = " ".join(toks)
    meta = {
        "L": L,
        "ell": ell,
        "ctx": ctx,
        "seed": None,  # filled by caller per instance seed
        "len_token": len_token,
        "mod": MOD,
    }
    return text, meta


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mano/MANO dataset (.jsonl or a single .json array)")
    p.add_argument("--L", type=int, default=None, help="Max operations per expression (>=1). If omitted and --Lset not provided, defaults to 16")
    p.add_argument("--Lset", type=str, default=None, help="Comma-separated options for L; sample one per instance (e.g., '16,13,10')")
    p.add_argument("--ctx", type=int, default=1024, help="Context/token cap (default 1024 as in paper)")
    p.add_argument("--num", type=int, default=1, help="# of samples to generate")
    p.add_argument("--seed", type=int, default=0, help="Base seed for RNG")
    p.add_argument("--len-token", choices=["query", "len"], default="query", help="Use '<query_ℓ>' or '<len_ℓ>' marker")
    p.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Output format")
    p.add_argument("--output", type=str, default="-", help="Output path or '-' for stdout")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)

    # Determine L choice set and default
    L_options: List[int] | None = None
    if ns.Lset:
        try:
            L_options = [int(x.strip()) for x in ns.Lset.split(',') if x.strip()]
        except ValueError:
            raise SystemExit("--Lset must be a comma-separated list of integers, e.g., 16,13,10")
        if not L_options:
            raise SystemExit("--Lset produced an empty list")
    if ns.L is None and L_options is None:
        ns.L = 16

    base_seed = ns.seed
    out_fp = sys.stdout if ns.output == '-' else open(ns.output, 'w', encoding='utf-8')
    close_out = ns.output != '-'

    try:
        if ns.format == 'jsonl':
            for i in range(ns.num):
                sample_seed = (base_seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFF
                rng = random.Random(sample_seed)
                L_this = rng.choice(L_options) if L_options else ns.L
                text, meta = build_mano_sample(rng, L_this, ns.ctx, ns.len_token)
                meta["seed"] = sample_seed
                out_fp.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False) + "\n")
                out_fp.flush()
        else:
            items: List[dict] = []
            for i in range(ns.num):
                sample_seed = (base_seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFF
                rng = random.Random(sample_seed)
                L_this = rng.choice(L_options) if L_options else ns.L
                text, meta = build_mano_sample(rng, L_this, ns.ctx, ns.len_token)
                meta["seed"] = sample_seed
                items.append({"text": text, "meta": meta})
            out_fp.write(json.dumps(items, ensure_ascii=False))
            out_fp.flush()
    finally:
        if close_out:
            out_fp.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

