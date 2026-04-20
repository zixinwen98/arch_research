#!/usr/bin/env python3
"""
DEPO dataset generator

Generates space-delimited text instances per spec and streams .jsonl lines.

Instance format:
  <bos> x1 y1 x2 y2 ... xn yn <query k1> q1 <ans> a1 ... <query kt> qt <ans> at

Where:
  - Single directed cycle over n nodes, edges shuffled before emission
  - t = min(10, n) queries, each q uniform over nodes, k uniform in [1..K], a = pi^k(q)
  - Specials: <bos>, <ans>, and <query k> (k=1..K)
  - Variants:
      depo1: vocab 50 (t0..t49); node name = 1–2 tokens; unique per node
      depo2: vocab 4 (a,b,c,d); node name = 5–7 tokens; unique per node
  - Packing: left-align; cap to --ctx tokens (default 2048). Ensure full edge list; drop trailing Q/A triples if needed.

CLI usage examples:
  # JSONL streaming
  python depo.py --variant depo1 --N 50 --K 8 --num 1000 > data.jsonl
  python depo.py --variant depo2 --N 80 --K 4 --schedule sqrt --seed 42 --output data.jsonl

  # Single JSON array file
  python depo.py --variant depo1 --N 50 --K 8 --num 1000 --format json --output data.json

Output lines:
  {"text": "...", "meta": {"variant": "depo1|depo2", "N": int, "K": int, "n": int, "t": int, "ctx": int, "seed": int, "schedule": "uniform|sqrt"}}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


# --------------------------- Tokenization helpers --------------------------- #


def depo_vocab(variant: str) -> List[str]:
    if variant == "depo1":
        return [f"t{i}" for i in range(50)]
    if variant == "depo2":
        return ["a", "b", "c", "d"]
    raise ValueError(f"Unknown variant: {variant}")


def sample_node_name_tokens(rng: random.Random, variant: str) -> List[str]:
    """Sample a single node name (as a list of tokens) for the given variant."""
    vocab = depo_vocab(variant)
    if variant == "depo1":
        length = rng.choice([1, 2])
    elif variant == "depo2":
        length = rng.choice([5, 6, 7])
    else:
        raise ValueError(variant)
    return [rng.choice(vocab) for _ in range(length)]


def sample_unique_node_names(
    rng: random.Random, variant: str, n: int, max_tries: int = 10000
) -> List[List[str]]:
    """Create n unique node names (each a list of tokens). Uniqueness is by exact token sequence.

    Retries up to max_tries samples total; raises on failure.
    """
    names: List[List[str]] = []
    seen: set[Tuple[str, ...]] = set()
    tries = 0
    while len(names) < n:
        if tries > max_tries:
            raise RuntimeError(
                f"Unable to sample {n} unique node names for {variant} within {max_tries} tries"
            )
        tries += 1
        nm = sample_node_name_tokens(rng, variant)
        key = tuple(nm)
        if key in seen:
            continue
        seen.add(key)
        names.append(nm)
    return names


# ----------------------------- Graph construction --------------------------- #


def build_cycle_edges(nodes: List[List[str]]) -> List[Tuple[List[str], List[str]]]:
    """Given a list of node token-sequences, construct a single directed cycle edges list.

    Returns a list of (src_tokens, dst_tokens) pairs of length n, in natural cycle order.
    """
    n = len(nodes)
    edges: List[Tuple[List[str], List[str]]] = []
    for i in range(n):
        j = (i + 1) % n
        edges.append((nodes[i], nodes[j]))
    return edges


def shuffle_pairs_in_place(rng: random.Random, pairs: List[Tuple[List[str], List[str]]]) -> None:
    rng.shuffle(pairs)


def node_index_map(nodes: List[List[str]]) -> dict[Tuple[str, ...], int]:
    return {tuple(tok_seq): i for i, tok_seq in enumerate(nodes)}


def advance_k_steps(nodes: List[List[str]], idx: int, k: int) -> List[str]:
    n = len(nodes)
    return nodes[(idx + k) % n]


# ------------------------------- Schedules ---------------------------------- #


def sample_n(
    rng: random.Random, N: int, schedule: str
) -> int:
    """Sample n in [3..N] using the chosen schedule.

    - uniform: P(n) = 1/(N-2)
    - sqrt:    P(n) ∝ 1/sqrt(N + n)
    """
    if N < 3:
        raise ValueError("N must be >= 3")
    if schedule == "uniform":
        return rng.randint(3, N)
    if schedule == "sqrt":
        values = list(range(3, N + 1))
        weights = [1.0 / math.sqrt(N + n) for n in values]
        total = sum(weights)
        thresh = rng.random() * total
        acc = 0.0
        for n, w in zip(values, weights):
            acc += w
            if acc >= thresh:
                return n
        return values[-1]
    raise ValueError(f"Unknown schedule: {schedule}")


# ------------------------------- Packing ------------------------------------ #


def tokens_len(seq: Sequence[str]) -> int:
    return len(seq)


def edges_tokens(edges: List[Tuple[List[str], List[str]]]) -> List[str]:
    out: List[str] = []
    for src, dst in edges:
        out.extend(src)
        out.extend(dst)
    return out


def qa_triplet_tokens(k: int, q: List[str], a: List[str]) -> List[str]:
    toks: List[str] = [f"<query {k}>"]
    toks.extend(q)
    toks.append("<ans>")
    toks.extend(a)
    return toks


# ------------------------------ Sample builder ------------------------------ #


@dataclass
class Sample:
    text: str
    meta: dict


def build_sample(
    *,
    N: int,
    K: int,
    ctx: int,
    variant: str,
    schedule: str,
    rng: random.Random,
    sample_seed: int,
    max_resample: int = 100,
) -> Sample:
    """Construct a single sample adhering to packing rules.

    Ensures full edge list fits within ctx; if not, resamples n and node names.
    """
    attempts = 0
    while True:
        attempts += 1
        n = sample_n(rng, N, schedule)
        t = min(10, n)

        # Build node names and a random cycle ordering
        nodes = sample_unique_node_names(rng, variant, n)
        rng.shuffle(nodes)

        edges = build_cycle_edges(nodes)
        shuffle_pairs_in_place(rng, edges)

        # Edges tokenization with <bos>
        edge_tok_seq = ["<bos>"] + edges_tokens(edges)
        edge_tok_len = tokens_len(edge_tok_seq)

        if edge_tok_len > ctx:
            if attempts >= max_resample:
                raise RuntimeError(
                    f"Edge list alone exceeds ctx={ctx} for N={N}, variant={variant}. Consider smaller N or larger ctx."
                )
            # Try again with a different n
            continue

        # Pack within ctx: keep full edges; drop trailing QA triples as needed
        remaining = ctx - edge_tok_len
        # Build QA triples to make packing easy
        qa_triples: List[List[str]] = []
        idx_map = node_index_map(nodes)
        for _ in range(t):
            q = rng.choice(nodes)
            q_idx = idx_map[tuple(q)]
            k = rng.randint(1, K)
            a = advance_k_steps(nodes, q_idx, k)
            qa_triples.append(qa_triplet_tokens(k, q, a))

        packed_qa = []
        used = 0
        for trip in qa_triples:
            L = len(trip)
            if used + L <= remaining:
                packed_qa.extend(trip)
                used += L
            else:
                break

        text_tokens = edge_tok_seq + packed_qa
        text = " ".join(text_tokens)

        meta = {
            "variant": variant,
            "N": N,
            "K": K,
            "n": n,
            "t": t,
            "ctx": ctx,
            "seed": sample_seed,
            "schedule": schedule,
        }

        return Sample(text=text, meta=meta)


# --------------------------------- CLI -------------------------------------- #


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DEPO dataset .jsonl")
    p.add_argument("--variant", choices=["depo1", "depo2"], default="depo1")
    p.add_argument("--N", type=int, default=50, help="Max nodes per graph (>=3)")
    p.add_argument("--K", type=int, default=8, help="Max k for queries (>=1)")
    p.add_argument("--ctx", type=int, default=2048, help="Context/token cap")
    p.add_argument("--schedule", choices=["uniform", "sqrt"], default="uniform")
    p.add_argument("--num", type=int, default=1, help="Number of samples to generate")
    p.add_argument("--seed", type=int, default=0, help="Base seed for RNG")
    p.add_argument("--output", type=str, default="-", help="Output path or '-' for stdout")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)

    base_rng = random.Random(ns.seed)

    # Open output stream
    if ns.output == "-":
        out = sys.stdout
        close_out = False
    else:
        out = open(ns.output, "w", encoding="utf-8")
        close_out = True

    try:
        for i in range(ns.num):
            # Derive per-sample seed deterministically for reproducibility
            sample_seed = (ns.seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFF
            rng = random.Random(sample_seed)
            sample = build_sample(
                N=ns.N,
                K=ns.K,
                ctx=ns.ctx,
                variant=ns.variant,
                schedule=ns.schedule,
                rng=rng,
                sample_seed=sample_seed,
            )
            line = {"text": sample.text, "meta": sample.meta}
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()
    finally:
        if close_out:
            out.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
