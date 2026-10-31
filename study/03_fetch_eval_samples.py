"""Step 3: download the source files of the RQ1 evaluation samples.

The development sample (60 repositories, random seed 2026) and the disjoint
held-out sample (60 repositories, seed 7331) were drawn from the
model-consuming repositories of the 5 October 2026 collection. The frozen
repository lists, the file paths, and a SHA-256 hash of every file as analyzed
in the paper are stored in data/eval_samples.json.

This script re-downloads those files into output/evalfiles/{dev,test}/. Because
repositories keep changing, some files may differ from the analyzed versions;
the script reports how many files still match the recorded hashes.
"""
import hashlib
import json
import os
import tarfile

import requests

from common import data_path, out_path

MAX_FILE_BYTES = 2_000_000


def local_name(split: str, repo: str, rel: str) -> str:
    return os.path.join(out_path("evalfiles", split), f"{repo.replace('/', '__')}__{rel.replace('/', '__')}")


def main():
    spec = json.load(open(data_path("eval_samples.json")))
    branches = {c["full_name"]: c["default_branch"] for c in json.load(open(data_path("candidates.json")))}
    for split in ("dev", "test"):
        os.makedirs(out_path("evalfiles", split), exist_ok=True)
        wanted = {}
        for f in spec[split]["files"]:
            wanted.setdefault(f["repo"], {})[f["path"]] = f["sha256"]
        same = changed = missing = 0
        for repo in spec[split]["repos"]:
            paths = wanted.get(repo, {})
            got = set()
            try:
                r = requests.get(f"https://codeload.github.com/{repo}/tar.gz/{branches[repo]}", stream=True, timeout=120)
                r.raw.decode_content = True
                with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                    for m in tf:
                        if not m.isfile() or m.size > MAX_FILE_BYTES:
                            continue
                        rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                        if rel not in paths:
                            continue
                        raw = tf.extractfile(m).read().decode("utf-8", "ignore")
                        open(local_name(split, repo, rel), "w", encoding="utf-8").write(raw)
                        got.add(rel)
                        if hashlib.sha256(raw.encode("utf-8")).hexdigest() == paths[rel]:
                            same += 1
                        else:
                            changed += 1
            except Exception as exc:
                print("error", repo, exc)
            missing += len(set(paths) - got)
        print(f"{split}: {same} files identical to the paper's snapshot, {changed} changed, {missing} missing")


if __name__ == "__main__":
    main()
