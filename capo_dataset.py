#!/usr/bin/env python3
"""
CAPO dataset: small biographies generator and simple table export.

Each entry has a random full name and five attributes with values:
  - birthdate (YYYY-MM-DD)
  - birth_place (city)
  - school (university)
  - company (employer)
  - major (field of study)

Text entry format (space-delimited, suitable for toy tokenizers):
  <name> , birthdate <date> birth_place <place> school <school> company <company> major <major>

Notes
  - Order of the attributes is optionally shuffled per entry.
  - A fixed attribute/value database can be constructed and then sampled from.
  - Exposes helpers to build the database, generate entries, and export a table (CSV/JSON).
  - All attribute names and values are exactly one whitespace token; person name
    is also exactly one token (we normalize by replacing spaces with underscores).
  - All attributes (birthdate, birth_place, school, company, major) are
    balanced to have the same number of candidate options.

CLI examples:
  # Generate a CSV table with 1000 rows
  python capo_dataset.py --num 1000 --csv capo.csv

  # Stream JSONL bios (one per line)
  python capo_dataset.py --num 100 --jsonl capo.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# ----------------------------- Data structures ----------------------------- #


Attribute = str


@dataclass
class CapoDatabase:
    first_names: List[str]
    last_names: List[str]
    birthdates: List[str]
    birth_places: List[str]
    schools: List[str]
    companies: List[str]
    majors: List[str]

    @property
    def attributes(self) -> List[Attribute]:
        return ["birthdate", "birth_place", "school", "company", "major"]


# ----------------------------- Database builder ---------------------------- #


def _maybe_import_faker():
    try:
        from faker import Faker  # type: ignore
        return Faker
    except Exception:
        return None


def _normalize_token(s: str) -> str:
    """Normalize to a single whitespace token by collapsing spaces to underscores
    and removing common punctuation that would create multiple tokens in text.
    """
    s = s.strip()
    # Replace any whitespace with underscores
    s = "_".join(s.split())
    # Drop commas; replace slashes with hyphen
    s = s.replace(",", "").replace("/", "-")
    return s


def build_capo_database(rng: random.Random, *, n_names: int = 100) -> CapoDatabase:
    """Build a fixed attribute/value database.

    If `faker` is available, sample realistic names and dates; otherwise, fall back to
    small built-in lists deterministically.
    """
    Faker = _maybe_import_faker()
    if Faker is not None:
        fk = Faker()
        fk.seed_instance(rng.randint(0, 2**31 - 1))
        first_names = [_normalize_token(fk.first_name()) for _ in range(n_names)]
        last_names = [_normalize_token(fk.last_name()) for _ in range(n_names)]
        # Use a modest set of places/schools/companies/majors to encourage reuse
        birthdates = [str(fk.date_of_birth(minimum_age=18, maximum_age=80)) for _ in range(400)]
        birth_places = list({_normalize_token(fk.city()) for _ in range(200)})
        schools = list({_normalize_token(fk.company() + " University") for _ in range(120)})
        companies = list({_normalize_token(fk.company()) for _ in range(200)})
        majors = [
            _normalize_token(x)
            for x in [
                "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
                "Economics", "History", "Philosophy", "Sociology", "Psychology",
                "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
                "Linguistics", "Statistics", "Art History", "Political Science",
            ]
        ]
    else:
        # Fallback deterministic mini database
        first_names = [
            "Alex", "Jamie", "Taylor", "Jordan", "Casey", "Riley", "Avery", "Quinn",
            "Morgan", "Parker", "Cameron", "Drew", "Reese", "Rowan", "Emerson",
        ]
        last_names = [
            "Smith", "Johnson", "Brown", "Williams", "Jones", "Miller",
            "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez",
        ]
        # YYYY-MM-DD simple range
        birthdates = [f"19{y:02d}-{m:02d}-{d:02d}" for y in range(60, 100) for m in (1, 6, 12) for d in (5, 15, 25)]
        birth_places = [
            "New_York", "Los_Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San_Antonio", "San_Diego", "Dallas", "San_Jose",
        ]
        schools = [
            "Northbridge_University", "Westfield_Institute", "Lakeview_College",
            "Riverside_University", "Hillside_College",
        ]
        companies = [
            "Globex", "Initech", "Hooli", "Umbrella", "Stark_Industries",
            "Wayne_Enterprises", "Wonka", "Aperture", "Soylent", "Oceanic",
        ]
        majors = [
            "Computer_Science", "Mathematics", "Physics", "History", "Economics",
            "Biology", "Chemistry", "Philosophy", "Statistics",
        ]

    # Equalize option counts across all attributes by sampling down to the minimum size
    def _sample_to_k(lst: List[str], k: int) -> List[str]:
        if len(lst) <= k:
            return list(lst)[:]
        return rng.sample(lst, k)

    k_min = min(len(birthdates), len(birth_places), len(schools), len(companies), len(majors))
    birthdates = _sample_to_k(birthdates, k_min)
    birth_places = _sample_to_k(birth_places, k_min)
    schools = _sample_to_k(schools, k_min)
    companies = _sample_to_k(companies, k_min)
    majors = _sample_to_k(majors, k_min)

    return CapoDatabase(
        first_names=first_names,
        last_names=last_names,
        birthdates=birthdates,
        birth_places=birth_places,
        schools=schools,
        companies=companies,
        majors=majors,
    )


# ----------------------------- Entry generation ---------------------------- #


def sample_name(rng: random.Random, db: CapoDatabase) -> str:
    # Single-token name via underscore join
    return _normalize_token(f"{rng.choice(db.first_names)}_{rng.choice(db.last_names)}")


def generate_bio_entry(rng: random.Random, db: CapoDatabase, *, shuffle_attrs: bool = True) -> Tuple[str, Dict[str, str]]:
    """Generate one biography string and the underlying attribute dict.

    Returns
    - text: '<name> , attr1 val1 attr2 val2 ... attr5 val5'
    - data: {'name': ..., 'birthdate': ..., 'birth_place': ..., 'school': ..., 'company': ..., 'major': ...}
    """
    name = sample_name(rng, db)
    # Sample values
    vals = {
        "birthdate": _normalize_token(rng.choice(db.birthdates)),
        "birth_place": _normalize_token(rng.choice(db.birth_places)),
        "school": _normalize_token(rng.choice(db.schools)),
        "company": _normalize_token(rng.choice(db.companies)),
        "major": _normalize_token(rng.choice(db.majors)),
    }
    order = db.attributes[:]
    if shuffle_attrs:
        rng.shuffle(order)
    # Emit only relevant tokens: person name, attribute name, attribute value
    parts = [name]
    for key in order:
        parts.append(key)
        parts.append(str(vals[key]))
    text = " ".join(parts)
    row = {"name": name, **vals}
    return text, row


def generate_table(rng: random.Random, db: CapoDatabase, n: int, *, shuffle_attrs: bool = True) -> Tuple[List[str], List[Dict[str, str]]]:
    texts: List[str] = []
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        text, row = generate_bio_entry(rng, db, shuffle_attrs=shuffle_attrs)
        texts.append(text)
        rows.append(row)
    return texts, rows


# --------------------------------- CLI ------------------------------------- #


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CAPO biographies as text and a table")
    p.add_argument("--num", type=int, default=1000, help="Number of entries to generate")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--no-shuffle", action="store_true", help="Do not shuffle attribute order in entries")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path for the table")
    p.add_argument("--json", type=str, default=None, help="Optional single JSON file path for the table")
    p.add_argument("--jsonl", type=str, default=None, help="Optional JSONL output path of text entries with meta")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv or [])
    # Enforce Faker usage for dataset generation from the CLI
    if _maybe_import_faker() is None:
        raise SystemExit(
            "Faker is required for CAPO dataset generation. Please install it: pip install faker"
        )
    rng = random.Random(ns.seed)
    db = build_capo_database(rng)
    texts, rows = generate_table(rng, db, ns.num, shuffle_attrs=(not ns.no_shuffle))

    # Write CSV/JSON table
    if ns.csv:
        fieldnames = ["name", "birthdate", "birth_place", "school", "company", "major"]
        with open(ns.csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as fp:
            json.dump(rows, fp, ensure_ascii=False, indent=2)
    if ns.jsonl:
        with open(ns.jsonl, "w", encoding="utf-8") as fp:
            for i, text in enumerate(texts):
                fp.write(json.dumps({"text": text, "meta": {"seed": ns.seed, "idx": i}}, ensure_ascii=False) + "\n")

    # If nothing was written, print a small preview to stdout
    if not any([ns.csv, ns.json, ns.jsonl]):
        for i in range(min(5, len(texts))):
            print(texts[i])

    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
