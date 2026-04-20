#!/usr/bin/env python3
"""
BREVO dataset generator (Mental Reasoning Breadth)

Generates instances of a random DAG and a query requiring listing all recursive
dependencies (ancestors) of the query vertex in topological order, starting
from the leaves, per the spec.

Text format (space-delimited):
  <bos> x1 y1 x2 y2 ... xm ym <query> q <ans> a1 a2 ... ap <eos>

Where (edges): xi -> yi means yi depends on xi.
The answer a1..ap are all vertices that q depends on (its ancestors), ordered
topologically such that if u -> v -> ... -> q, then u appears before v.

Variants:
  - brevo1: each vertex name is a single unique token from a vocabulary of size V
            (tokens are v1..vV). Unique across nodes.
  - brevo2: each vertex name is 2–4 tokens sampled from a 4-token vocab {a,b,c,d},
            unique by token sequence across nodes.

Sampling n in {3..N} uses a schedule in {uniform, sqrt}, where sqrt uses
P(n) ∝ 1/sqrt(N+n) as in DEPO.

Output lines:
  {"text": "...", "meta": {"variant": "brevo1|brevo2", "N": int, "n": int,
                              "m": int, "ctx": int, "seed": int,
                              "schedule": "uniform|sqrt"}}

CLI usage examples:
  # JSONL streaming (defaults match paper):
  #   brevo1: N∈{110,90,70}, ctx=1024
  #   brevo2: N∈{50,40,30},  ctx=1536
  python -m arch_research.tasks.brevo --variant brevo1 --Nset 110,90,70 --num 100 > brevo1.jsonl
  python -m arch_research.tasks.brevo --variant brevo2 --Nset 50,40,30  --num 100 > brevo2.jsonl

  # Single JSON array
  python -m arch_research.tasks.brevo --variant brevo2 --Nset 50,40,30 --num 100 --format json --output brevo.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


# ------------------------------- Schedules ---------------------------------- #


def sample_n(rng: random.Random, N: int, schedule: str) -> int:
    if N < 3:
        raise ValueError("N must be >= 3")
    if schedule == "uniform":
        return rng.randint(3, N)
    if schedule == "sqrt":
        values = list(range(3, N + 1))
        weights = [1.0 / math.sqrt(N + n) for n in values]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for n, w in zip(values, weights):
            acc += w
            if acc >= r:
                return n
        return values[-1]
    raise ValueError(f"Unknown schedule: {schedule}")


# ------------------------------- Names -------------------------------------- #


def brevo_vocab(variant: str, V: int) -> List[str]:
    if variant == "brevo1":
        return [f"v{i}" for i in range(1, V + 1)]
    if variant == "brevo2":
        return ["a", "b", "c", "d"]
    raise ValueError(variant)


def sample_node_name_tokens(rng: random.Random, variant: str, V: int) -> List[str]:
    if variant == "brevo1":
        # Single token name from v1..vV
        return [rng.choice(brevo_vocab(variant, V))]
    elif variant == "brevo2":
        vocab = brevo_vocab(variant, V)
        length = rng.choice([2, 3, 4])
        return [rng.choice(vocab) for _ in range(length)]
    else:
        raise ValueError(variant)


def sample_unique_node_names(rng: random.Random, variant: str, n: int, V: int, max_tries: int = 10000) -> List[List[str]]:
    names: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()
    tries = 0
    while len(names) < n:
        if tries > max_tries:
            raise RuntimeError(f"Unable to sample {n} unique node names for {variant} within {max_tries} tries")
        tries += 1
        nm = sample_node_name_tokens(rng, variant, V)
        key = tuple(nm)
        if key in seen:
            continue
        seen.add(key)
        names.append(nm)
    return names


# ------------------------------- DAG ---------------------------------------- #


def build_random_dag(rng: random.Random, n: int) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Build a DAG following the protocol.

    Returns (edges, topo_order) where edges are (u->v) pairs over vertex ids 0..n-1,
    and topo_order is the left-to-right order used during construction.
    """
    verts = list(range(n))
    rng.shuffle(verts)

    # Leaves
    L = rng.randint(1, (n // 4) + 1)

    outdeg = {v: 0 for v in verts}
    indeg = {v: 0 for v in verts}
    edges: List[Tuple[int, int]] = []

    # Process each vertex from L onwards, connect with 1..4 previous vertices
    for idx in range(L, n):
        v = verts[idx]
        candidates = [u for u in verts[:idx] if outdeg[u] <= 3]
        if not candidates:
            # Fallback: connect to at least one previous vertex ignoring the outdeg cap
            candidates = verts[:idx]
        kmax = min(4, len(candidates))
        k = rng.randint(1, kmax)
        parents = rng.sample(candidates, k)
        for u in parents:
            edges.append((u, v))
            outdeg[u] += 1
            indeg[v] += 1

    topo_order = verts[:]  # Left-to-right construction order is a topological order
    return edges, topo_order


def ancestors_of(q: int, edges: List[Tuple[int, int]]) -> Set[int]:
    # Build reverse adjacency
    rev: Dict[int, List[int]] = {}
    for u, v in edges:
        rev.setdefault(v, []).append(u)
    # DFS/BFS upstream
    anc: Set[int] = set()
    stack = list(rev.get(q, []))
    while stack:
        u = stack.pop()
        if u in anc:
            continue
        anc.add(u)
        stack.extend(rev.get(u, []))
    return anc


# ------------------------------ Text emission -------------------------------- #


def edges_tokens(edges: List[Tuple[int, int]], names: List[List[str]]) -> List[str]:
    toks: List[str] = []
    for u, v in edges:
        toks.extend(names[u])
        toks.extend(names[v])
    return toks


def build_brevo_sample(*, N: int, ctx: int, variant: str, schedule: str, V: int, rng: random.Random, sample_seed: int, max_resample: int = 100) -> Tuple[str, dict]:
    attempts = 0
    while True:
        attempts += 1
        n = sample_n(rng, N, schedule)

        edges, topo = build_random_dag(rng, n)
        m = len(edges)

        # Choose query from the last quarter; avoid the very last if possible
        start_idx = (3 * n) // 4
        pool = topo[start_idx:n - 1] if n - 1 > start_idx else topo[start_idx:]
        if not pool:
            pool = [topo[-1]]
        q = rng.choice(pool)

        # Compute ancestors and their topological order (by topo order)
        anc = ancestors_of(q, edges)
        pos = {v: i for i, v in enumerate(topo)}
        anc_sorted = sorted(anc, key=lambda v: pos[v])

        # Assign names (reshuffle mapping implicitly by shuffling vertex list before naming)
        names = sample_unique_node_names(rng, variant, n, V)

        # Map vertex id -> name tokens: use a random permutation to "reshuffle the vertices"
        perm = list(range(n))
        rng.shuffle(perm)
        id_to_name = [None] * n  # type: ignore
        for vid, slot in enumerate(perm):
            id_to_name[vid] = names[slot]

        # Emit tokens
        tok_seq: List[str] = ["<bos>"]
        tok_seq.extend(edges_tokens(edges, id_to_name))
        tok_seq.append("<query>")
        tok_seq.extend(id_to_name[q])
        tok_seq.append("<ans>")
        for v in anc_sorted:
            tok_seq.extend(id_to_name[v])
        tok_seq.append("<eos>")

        if len(tok_seq) > ctx:
            if attempts >= max_resample:
                raise RuntimeError(
                    f"Sample exceeds ctx={ctx} with N={N}, variant={variant}. Consider increasing ctx or reducing N."
                )
            continue

        text = " ".join(tok_seq)
        meta = {
            "variant": variant,
            "N": N,
            "n": n,
            "m": m,
            "ctx": ctx,
            "seed": sample_seed,
            "schedule": schedule,
        }
        return text, meta


# --------------------------------- CLI -------------------------------------- #


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BREVO dataset (.jsonl or a single .json array)")
    p.add_argument("--variant", choices=["brevo1", "brevo2"], default="brevo1")
    p.add_argument("--N", type=int, default=None, help="Max nodes per graph (>=3). If omitted and --Nset not provided, defaults to 110 (brevo1) or 50 (brevo2)")
    p.add_argument("--Nset", type=str, default=None, help="Comma-separated options for N; sample one per instance (e.g., '110,90,70')")
    p.add_argument("--ctx", type=int, default=None, help="Context/token cap; defaults to 1024 (brevo1) or 1536 (brevo2)")
    p.add_argument("--schedule", choices=["uniform", "sqrt"], default="sqrt")
    p.add_argument("--num", type=int, default=1, help="# of samples to generate")
    p.add_argument("--seed", type=int, default=0, help="Base seed for RNG")
    p.add_argument("--V", type=int, default=1000, help="Vocab size for brevo1 (v1..vV); ignored for brevo2")
    p.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Output format")
    p.add_argument("--output", type=str, default="-", help="Output path or '-' for stdout")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)

    if ns.variant == "brevo2":
        V = 4
    else:
        V = ns.V

    # Variant-based defaults per paper
    if ns.ctx is None:
        ns.ctx = 1024 if ns.variant == "brevo1" else 1536

    # Parse N set or set default N
    N_options: List[int] | None = None
    if ns.Nset:
        try:
            N_options = [int(x.strip()) for x in ns.Nset.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("--Nset must be a comma-separated list of integers, e.g., 110,90,70")
        if not N_options:
            raise SystemExit("--Nset produced an empty list")
    if ns.N is None and N_options is None:
        ns.N = 110 if ns.variant == "brevo1" else 50

    base_seed = ns.seed

    # Output handling
    out_fp = sys.stdout if ns.output == "-" else open(ns.output, "w", encoding="utf-8")
    close_out = ns.output != "-"

    try:
        if ns.format == "jsonl":
            for i in range(ns.num):
                sample_seed = (base_seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFF
                rng = random.Random(sample_seed)
                N_this = rng.choice(N_options) if N_options else ns.N
                text, meta = build_brevo_sample(
                    N=N_this,
                    ctx=ns.ctx,
                    variant=ns.variant,
                    schedule=ns.schedule,
                    V=V,
                    rng=rng,
                    sample_seed=sample_seed,
                )
                out_fp.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False) + "\n")
                out_fp.flush()
        else:  # json array
            items: List[dict] = []
            for i in range(ns.num):
                sample_seed = (base_seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFF
                rng = random.Random(sample_seed)
                N_this = rng.choice(N_options) if N_options else ns.N
                text, meta = build_brevo_sample(
                    N=N_this,
                    ctx=ns.ctx,
                    variant=ns.variant,
                    schedule=ns.schedule,
                    V=V,
                    rng=rng,
                    sample_seed=sample_seed,
                )
                items.append({"text": text, "meta": meta})
            out_fp.write(json.dumps(items, ensure_ascii=False))
            out_fp.flush()
    finally:
        if close_out:
            out_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
