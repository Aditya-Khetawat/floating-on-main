"""Shared paths, constants and helpers for the study scripts."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, ROOT)  # make the local pincheck package importable without installing

# Directories treated as vendored copies of loading libraries or environments;
# call sites inside them are excluded from the analysis (Section IV).
VENDORED = {"transformers", "diffusers", "sentence_transformers", "peft", "site-packages",
            "third_party", "thirdparty", "vendor", "external", "_vendor"}

# Repository filters applied after sampling (Section IV, "Sampling").
MAX_SIZE_KB = 80_000
EXCLUDED_REPOS = {"UKPLab/sentence-transformers"}
EXCLUDED_OWNERS = {"huggingface"}

COLLECTION_DATE = "2026-10-20"


def data_path(*parts: str) -> str:
    return os.path.join(DATA, *parts)


def out_path(*parts: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, *parts)


def is_vendored(file_path: str) -> bool:
    parts = file_path.split("/")
    return bool(set(parts[:-1]) & VENDORED) or any(p.startswith("transformers-") for p in parts)


def load_results(path: str | None = None):
    """Load per-repository results and drop vendored and local call sites."""
    path = path or data_path("results.jsonl")
    repos = [json.loads(line) for line in open(path, encoding="utf-8")]
    for r in repos:
        r["calls"] = [c for c in r["calls"] if not is_vendored(c["file"]) and c["id_class"] != "local"]
    return repos


def keep_candidate(c: dict) -> bool:
    owner = c["full_name"].split("/")[0].lower()
    return (c["size"] < MAX_SIZE_KB and owner not in EXCLUDED_OWNERS
            and c["full_name"] not in EXCLUDED_REPOS)
