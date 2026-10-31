"""Step 2: download each candidate repository and run PinCheck on it.

Streams the default-branch archive from codeload.github.com, keeps Python files,
notebooks, dependency manifests, CI workflows, Dockerfiles and test files, and
writes one JSON line per repository to output/results.jsonl. The run is
resumable. The data used in the paper (collected 20 October 2026) is
data/results.jsonl.

Usage: python study/02_analyze_repos.py [--candidates data/candidates.json] [--workers 6]
"""
import argparse
import concurrent.futures as cf
import json
import os
import tarfile

import requests

from common import data_path, keep_candidate, out_path
from pincheck.core import LOAD_APIS, analyze_source, notebook_to_source
from pincheck.manifests import KEY_PACKAGES, SKIP_DIRS, dependency_spec, is_manifest, path_signals

MAX_FILE_BYTES = 3_000_000


def process(repo: dict) -> dict:
    name = repo["full_name"]
    url = f"https://codeload.github.com/{name}/tar.gz/{repo['default_branch']}"
    res = {"repo": name, "stars": repo["stargazers_count"], "size_kb": repo["size"],
           "created_at": repo["created_at"], "calls": [], "py_files": 0, "nb_files": 0,
           "lockfile": False, "ci": False, "docker": False, "tests": False, "error": None}
    manifest_text = ""
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            res["error"] = f"http{r.status_code}"
            res["deps"] = {k: None for k in KEY_PACKAGES}
            return res
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                if any(s in "/" + rel for s in SKIP_DIRS):
                    continue
                ci, docker, tests, lock = path_signals(rel)
                if ci:  # workflow files only count as CI (matches the original pipeline)
                    res["ci"] = True
                    continue
                res["docker"] |= docker; res["tests"] |= tests; res["lockfile"] |= lock
                if lock:
                    continue
                base = os.path.basename(rel)
                manifest = is_manifest(base)
                is_py, is_nb = base.endswith(".py"), base.endswith(".ipynb")
                if not (manifest or is_py or is_nb) or m.size > MAX_FILE_BYTES:
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                raw = fh.read().decode("utf-8", "ignore")
                if manifest:
                    manifest_text += "\n" + raw
                if is_py or is_nb:
                    src = raw if is_py else notebook_to_source(raw)
                    if src is None:
                        continue
                    res["py_files" if is_py else "nb_files"] += 1
                    if not any(api in src for api in LOAD_APIS):
                        continue
                    found = analyze_source(src, rel)
                    if found:
                        res["calls"].extend(f.to_dict() for f in found)
    except Exception as exc:  # network or archive errors are recorded, not fatal
        res["error"] = type(exc).__name__ + ":" + str(exc)[:100]
    res["deps"] = {k: dependency_spec(manifest_text, k) for k in KEY_PACKAGES}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=data_path("candidates.json"))
    ap.add_argument("--out", default=out_path("results.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cands = [c for c in json.load(open(args.candidates)) if keep_candidate(c)]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["repo"] for line in open(args.out)}
    todo = [c for c in cands if c["full_name"] not in done]
    print(f"{len(todo)} of {len(cands)} repositories to analyze", flush=True)
    with open(args.out, "a") as out, cf.ThreadPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(process, todo), 1):
            out.write(json.dumps(res) + "\n")
            out.flush()
            if i % 25 == 0:
                print(i, res["repo"], len(res["calls"]), res["error"], flush=True)


if __name__ == "__main__":
    main()
