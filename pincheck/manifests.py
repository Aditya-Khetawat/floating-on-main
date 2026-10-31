"""Dependency-manifest and MLOps-practice signals for a repository."""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, Optional, Tuple

KEY_PACKAGES = ("transformers", "sentence-transformers", "huggingface-hub", "torch")
LOCKFILES = frozenset({"poetry.lock", "uv.lock", "Pipfile.lock", "pdm.lock", "conda-lock.yml"})
SKIP_DIRS = ("site-packages/", "/venv/", "/.venv/", "node_modules/", "/env/lib/")


def is_manifest(basename: str) -> bool:
    return ((basename.startswith("requirements") and basename.endswith((".txt", ".in")))
            or basename in ("pyproject.toml", "setup.py", "setup.cfg", "Pipfile")
            or (basename.startswith("environment") and basename.endswith((".yml", ".yaml"))))


def dependency_spec(text: str, package: str) -> Optional[str]:
    """Strongest declaration of ``package`` in manifest text: "exact", "range", "unpinned" or None."""
    pattern = re.compile(r"(?im)^[\s\"'\-]*" + re.escape(package).replace("\\-", "[-_]")
                         + r"(\[[^\]]*\])?\s*([=<>~!]=?[^;#\"'\n,]*)?")
    rank = {"exact": 3, "range": 2, "unpinned": 1}
    best: Optional[str] = None
    for m in pattern.finditer(text):
        spec = (m.group(2) or "").strip()
        cls = "exact" if spec.startswith("==") else ("range" if spec else "unpinned")
        if best is None or rank[cls] > rank[best]:
            best = cls
    return best


def path_signals(rel_path: str) -> Tuple[bool, bool, bool, bool]:
    """(ci, docker, tests, lockfile) signals contributed by one repository-relative path."""
    base = os.path.basename(rel_path)
    slashed = "/" + rel_path
    ci = rel_path.startswith(".github/workflows/")
    docker = base.startswith("Dockerfile") or base in ("docker-compose.yml", "compose.yaml")
    tests = base.startswith("test_") or "/tests/" in slashed or base.endswith("_test.py")
    lock = base in LOCKFILES
    return ci, docker, tests, lock


def scan_repository_metadata(files: Iterable[Tuple[str, Optional[str]]]) -> Dict:
    """Aggregate practice signals and key-package declarations.

    ``files`` yields (repository-relative path, text or None) pairs; text is only
    needed for dependency manifests.
    """
    sig = {"ci": False, "docker": False, "tests": False, "lockfile": False}
    manifest_text = ""
    for rel, text in files:
        ci, docker, tests, lock = path_signals(rel)
        if ci:
            sig["ci"] = True
            continue
        sig["docker"] |= docker
        sig["tests"] |= tests
        sig["lockfile"] |= lock
        if text is not None and is_manifest(os.path.basename(rel)):
            manifest_text += "\n" + text
    sig["deps"] = {p: dependency_spec(manifest_text, p) for p in KEY_PACKAGES}
    return sig
