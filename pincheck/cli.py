"""Command-line interface: ``pincheck [PATH ...] [--fail-on RULES]``.

Exit status: 0 if no violations of the selected rules, 1 if violations were
found, 2 on usage errors.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from typing import Iterable, List, Optional

from . import __version__
from .core import Finding, analyze_notebook, analyze_source
from .manifests import SKIP_DIRS

RULES = {
    "unpinned": "hub load (literal or indirect identifier) without any revision",
    "remote-code": "trust_remote_code=True without a commit-hash revision",
    "default-pipeline": "pipeline(task) without an explicit model",
    "named-ref": "revision is a mutable branch or tag name rather than a commit hash",
}
DEFAULT_EXCLUDES = ["*/site-packages/*", "*/.venv/*", "*/venv/*", "*/node_modules/*", "*/.git/*"]
IGNORE_MARKER = "pincheck: ignore"
REVISION_LABELS = {"absent": "none", "indirect": "configurable", "named_ref": "named-ref", "hash": "hash"}


def violations(f: Finding) -> List[str]:
    """Rules violated by one finding (local loads never violate anything)."""
    if f.id_class == "local":
        return []
    out = []
    if f.pipeline_default:
        out.append("default-pipeline")
    elif f.revision == "absent":
        out.append("unpinned")
    if f.revision == "named_ref":
        out.append("named-ref")
    if f.trust_remote_code and f.revision != "hash":
        out.append("remote-code")
    return out


def iter_files(paths: Iterable[str], notebooks: bool, excludes: List[str]):
    exts = (".py", ".ipynb") if notebooks else (".py",)
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if not d.startswith(".") or d == ".github"]
            for name in sorted(files):
                full = os.path.join(root, name)
                norm = "/" + full.replace(os.sep, "/")
                if not name.endswith(exts) or any(s in norm for s in SKIP_DIRS):
                    continue
                if any(fnmatch.fnmatch(norm, pat) for pat in excludes):
                    continue
                yield full


def scan(paths: Iterable[str], notebooks: bool = True, excludes: Optional[List[str]] = None):
    """Yield (finding, rules, ignored) for every call site; errors go to stderr."""
    excludes = DEFAULT_EXCLUDES + (excludes or [])
    for path in iter_files(paths, notebooks, excludes):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
        except OSError as exc:
            print(f"pincheck: cannot read {path}: {exc}", file=sys.stderr)
            continue
        found = analyze_notebook(raw, path) if path.endswith(".ipynb") else analyze_source(raw, path)
        if found is None:
            print(f"pincheck: skipped {path} (could not parse)", file=sys.stderr)
            continue
        lines = raw.splitlines() if path.endswith(".py") else []
        for f in sorted(found, key=lambda x: x.line):
            ignored = bool(lines) and 0 < f.line <= len(lines) and IGNORE_MARKER in lines[f.line - 1]
            yield f, violations(f), ignored


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pincheck",
        description="Detect unpinned Hugging Face model dependencies in Python code.",
        epilog="Rules: " + "; ".join(f"{k}: {v}" for k, v in RULES.items()))
    ap.add_argument("paths", nargs="*", default=["."], help="files or directories (default: .)")
    ap.add_argument("--fail-on", default="unpinned,remote-code,default-pipeline",
                    help="comma-separated rules that cause exit status 1 "
                         "(default: unpinned,remote-code,default-pipeline; use 'none' to never fail)")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--no-notebooks", action="store_true", help="skip .ipynb files")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="exclude paths matching GLOB (repeatable)")
    ap.add_argument("--all", action="store_true", help="also list call sites without violations")
    ap.add_argument("--version", action="version", version=f"pincheck {__version__}")
    args = ap.parse_args(argv)

    fail_on = set() if args.fail_on.strip().lower() == "none" else {
        r.strip() for r in args.fail_on.split(",") if r.strip()}
    unknown = fail_on - set(RULES)
    if unknown:
        ap.error(f"unknown rule(s): {', '.join(sorted(unknown))}; choose from {', '.join(RULES)}")

    records, n_sites, n_fail = [], 0, 0
    for f, rules, ignored in scan(args.paths, not args.no_notebooks, args.exclude):
        n_sites += 1
        failing = [] if ignored else [r for r in rules if r in fail_on]
        n_fail += bool(failing)
        if args.format == "json":
            records.append(dict(f.to_dict(), violations=rules, failing=failing, ignored=ignored))
        elif failing or (args.all and not ignored):
            model = f.model_id or f"<{f.id_class}>"
            tag = ",".join(failing) if failing else "ok"
            print(f"{f.file}:{f.line}: [{tag}] {f.api}({model}) revision={REVISION_LABELS[f.revision]}")

    if args.format == "json":
        json.dump({"call_sites": n_sites, "failing": n_fail, "findings": records}, sys.stdout, indent=2)
        print()
    else:
        print(f"pincheck: {n_sites} model-loading call site(s), {n_fail} failing "
              f"[{', '.join(sorted(fail_on)) or 'no rules'}]", file=sys.stderr)
    return 1 if n_fail else 0


def entry() -> None:
    try:
        sys.exit(main())
    except BrokenPipeError:  # e.g. output piped into `head`
        sys.stderr.close()
        sys.exit(1)


if __name__ == "__main__":
    entry()
